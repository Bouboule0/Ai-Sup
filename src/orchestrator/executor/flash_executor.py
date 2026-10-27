from __future__ import annotations

import json
import os
from typing import Any
from google import genai
from google.genai import types
from pydantic import ValidationError

from orchestrator.executor.base import BaseExecutor
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import ExecutionResult, Step, StepStatus
from orchestrator.tools.registry import ToolRegistry
from orchestrator.utils.parsing import extract_json_block, format_validation_error

EXECUTOR_SYSTEM_PROMPT = """Tu es l'Exécuteur dans une architecture Plan-then-Execute.
Tu reçois une étape atomique (`Step`), la liste des outils autorisés avec leurs schémas d'arguments, et l'historique d'observations des outils.
Fonctionnement en boucle ReAct :
- Si tu as besoin d'appeler un ou plusieurs outils pour accomplir l'étape, remplis `tool_calls` avec les appels souhaités et mets `status` à `"in_progress"`.
- Dès que les critères de succès (`success_criteria`) sont remplis et qu'aucun appel d'outil supplémentaire n'est requis, laisse `tool_calls` vide (`[]`) et mets `status` à `"success"`.
- Si une erreur bloquante empêche la réalisation de l'étape, laisse `tool_calls` vide et mets `status` à `"failed"` en détaillant la cause dans `justification` et `error`.
Réponds toujours en respectant strictement le schéma JSON `ExecutionResult`.
"""


class FlashExecutor(BaseExecutor):
    """Exécuteur concret utilisant un modèle Google Gemini (ex: Gemini 3.8 Flash ou 3.1 Pro)."""

    def __init__(
        self,
        cost_tracker: CostTracker,
        tool_registry: ToolRegistry,
        model_key: str = "executor",
        max_react_turns: int = 4,
    ) -> None:
        """Initialise l'exécuteur Gemini.

        Args:
            cost_tracker: Suivi budgétaire partagé.
            tool_registry: Registre d'outils pour exécuter les appels demandés.
            model_key: Clé du modèle dans models.yaml (par défaut 'executor').
            max_react_turns: Nombre maximal d'itérations ReAct par étape.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("La variable d'environnement GEMINI_API_KEY est absente.")
        self._client = genai.Client(api_key=api_key)
        self._cost_tracker = cost_tracker
        self._tool_registry = tool_registry
        self._model_key = model_key
        self._model_id = cost_tracker.get_model_id(model_key)
        self._max_react_turns = max_react_turns

    async def execute(self, step: Step, state: dict[str, Any]) -> ExecutionResult:
        """Exécute une étape via une boucle ReAct courte avec le modèle Gemini configuré."""
        allowed_tools = self._tool_registry.get_tools(step.tools_allowed)
        tools_map = {t.name: t for t in allowed_tools}
        tools_spec = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema(),
            }
            for t in allowed_tools
        ]

        observations: list[dict[str, Any]] = []
        last_result: ExecutionResult | None = None

        for _ in range(self._max_react_turns):
            if self._cost_tracker.is_over_budget():
                return ExecutionResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    output="",
                    justification="Exécution interrompue : plafond budgétaire atteint.",
                    error="BudgetExceeded",
                )

            prompt = self._build_prompt(step, state, tools_spec, observations)

            # À VÉRIFIER: confirmer les paramètres exacts de GenerateContentConfig
            # avec response_schema=ExecutionResult.
            response = await self._client.aio.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXECUTOR_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=ExecutionResult,
                    temperature=0.1,
                ),
            )

            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            self._cost_tracker.record_call(
                self._model_key, input_tokens, output_tokens, role="executor"
            )

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, ExecutionResult):
                result = parsed
            else:
                raw_text = getattr(response, "text", "") or ""
                try:
                    payload = extract_json_block(raw_text)
                    result = ExecutionResult.model_validate(payload)
                except (ValueError, ValidationError) as exc:
                    err_msg = (
                        format_validation_error(exc)
                        if isinstance(exc, ValidationError)
                        else str(exc)
                    )
                    return ExecutionResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        output="",
                        justification="La réponse du modèle n'a pas pu être validée.",
                        error=err_msg,
                    )

            last_result = result

            if not result.tool_calls or result.status in (StepStatus.SUCCESS, StepStatus.FAILED):
                if result.status == StepStatus.IN_PROGRESS and not result.tool_calls:
                    result.status = StepStatus.SUCCESS
                return result

            for call in result.tool_calls:
                if call.tool_name not in tools_map:
                    observations.append(
                        {
                            "tool": call.tool_name,
                            "status": "error",
                            "output": f"Outil '{call.tool_name}' non autorisé pour cette étape.",
                        }
                    )
                    continue
                try:
                    tool_output = tools_map[call.tool_name].execute(call.arguments)
                    observations.append(
                        {
                            "tool": call.tool_name,
                            "arguments": call.arguments,
                            "status": "ok",
                            "output": tool_output,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    observations.append(
                        {
                            "tool": call.tool_name,
                            "arguments": call.arguments,
                            "status": "error",
                            "output": f"{type(exc).__name__}: {exc}",
                        }
                    )

        return ExecutionResult(
            step_id=step.id,
            status=StepStatus.FAILED,
            output=last_result.output if last_result else "",
            justification="Nombre maximal d'itérations ReAct atteint sans complétion.",
            error="MaxReactTurnsExceeded",
        )

    @staticmethod
    def _build_prompt(
        step: Step,
        state: dict[str, Any],
        tools_spec: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> str:
        """Construit le prompt complet envoyé à l'exécuteur."""
        return (
            f"Étape à exécuter :\n{step.model_dump_json(indent=2)}\n\n"
            f"Outils autorisés :\n{json.dumps(tools_spec, ensure_ascii=False, indent=2)}\n\n"
            f"État partagé du projet :\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
            f"Historique des appels d'outils sur cette étape :\n"
            f"{json.dumps(observations, ensure_ascii=False, indent=2)}"
        )
