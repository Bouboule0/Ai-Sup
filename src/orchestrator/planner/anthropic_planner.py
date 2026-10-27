from __future__ import annotations

import json
import os
from typing import Any
import httpx
from pydantic import ValidationError

from orchestrator.planner.argon_planner import PLANNER_SYSTEM_PROMPT
from orchestrator.planner.base import BasePlanner
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import Plan
from orchestrator.utils.parsing import extract_json_block, format_validation_error

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


class AnthropicPlanner(BasePlanner):
    """Planificateur utilisant un modèle Anthropic Claude (ex: Claude Opus 5.5)."""

    def __init__(self, cost_tracker: CostTracker, model_key: str = "claude-opus-5.5") -> None:
        """Initialise le planificateur Anthropic et associe le CostTracker.

        Args:
            cost_tracker: Instance de suivi budgétaire.
            model_key: Clé du modèle dans models.yaml (ex: 'claude-opus-5.5').
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("La variable d'environnement ANTHROPIC_API_KEY est absente.")
        self._api_key = api_key
        self._cost_tracker = cost_tracker
        self._model_key = model_key
        self._model_id = cost_tracker.get_model_id(model_key)

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Génère un plan d'exécution validé via l'API Anthropic Messages."""
        if self._cost_tracker.is_over_budget():
            raise RuntimeError("Budget dépassé avant l'appel au planificateur Anthropic.")

        schema_json = json.dumps(Plan.model_json_schema(), ensure_ascii=False, indent=2)
        system_prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"Réponds UNIQUEMENT avec un objet JSON valide conforme à ce schéma JSON :\n{schema_json}"
        )
        user_prompt = (
            f"Objectif principal :\n{goal}\n\n"
            f"Contexte actuel du projet (JSON) :\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        )

        # À VÉRIFIER: version d'en-tête anthropic-version et identifiant exact du modèle Claude 5.5
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self._model_id,
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self._cost_tracker.record_call(
            self._model_key, input_tokens, output_tokens, role="planner"
        )

        content_blocks = data.get("content", [])
        raw_text = "\n".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        try:
            payload = extract_json_block(raw_text)
            return Plan.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_validation_error(exc)) from exc
