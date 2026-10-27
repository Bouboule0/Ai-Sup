from __future__ import annotations

import json
import os
from typing import Any
from google import genai
from google.genai import types
from pydantic import ValidationError

from orchestrator.planner.base import BasePlanner
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import Plan
from orchestrator.utils.parsing import extract_json_block, format_validation_error

PLANNER_SYSTEM_PROMPT = """Tu es l'Architecte Planificateur d'un système Plan-then-Execute.
Ton unique rôle est de décomposer l'objectif logiciel de l'utilisateur en un plan d'exécution JSON strictement conforme au schéma `Plan`.
Règles strictes :
1. Ne produis JAMAIS de code applicatif final complet dans tes descriptions : décris précisément ce que l'Exécuteur devra coder ou vérifier.
2. Découpe le travail en étapes atomiques, ordonnées et vérifiables (`Step`).
3. N'utilise que les outils disponibles : `read_file`, `write_file`, `list_dir`, `run_command`.
4. Assure-toi que chaque étape définit un `success_criteria` concret et mesurable.
"""


class ArgonPlanner(BasePlanner):
    """Planificateur concret utilisant un modèle Google Gemini (ex: Gemini 4 Argon) via google-genai."""

    def __init__(self, cost_tracker: CostTracker, model_key: str = "planner") -> None:
        """Initialise le client google-genai et associe le CostTracker.

        Args:
            cost_tracker: Instance de suivi budgétaire.
            model_key: Clé du modèle dans models.yaml (par défaut 'planner').
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("La variable d'environnement GEMINI_API_KEY est absente.")
        self._client = genai.Client(api_key=api_key)
        self._cost_tracker = cost_tracker
        self._model_key = model_key
        self._model_id = cost_tracker.get_model_id(model_key)

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Génère un plan d'exécution validé avec le modèle Gemini configuré."""
        if self._cost_tracker.is_over_budget():
            raise RuntimeError("Budget dépassé avant l'appel au planificateur.")

        user_prompt = (
            f"Objectif principal :\n{goal}\n\n"
            f"Contexte actuel du projet (JSON) :\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        )

        # À VÉRIFIER: confirmer que client.aio.models.generate_content accepte toujours
        # GenerateContentConfig avec response_mime_type="application/json" et response_schema=Plan.
        response = await self._client.aio.models.generate_content(
            model=self._model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=PLANNER_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=Plan,
                temperature=0.2,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        self._cost_tracker.record_call(
            self._model_key, input_tokens, output_tokens, role="planner"
        )

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, Plan):
            return parsed

        raw_text = getattr(response, "text", "") or ""
        try:
            payload = extract_json_block(raw_text)
            return Plan.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_validation_error(exc)) from exc
