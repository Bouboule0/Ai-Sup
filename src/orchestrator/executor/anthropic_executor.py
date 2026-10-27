from __future__ import annotations

import json
import os
from typing import Any
import httpx
from pydantic import ValidationError

from orchestrator.executor.base import BaseExecutor
from orchestrator.executor.flash_executor import EXECUTOR_SYSTEM_PROMPT, FlashExecutor
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import ExecutionResult, Step, StepStatus
from orchestrator.tools.file_system import verify_expected_outputs
from orchestrator.tools.registry import ToolRegistry
from orchestrator.utils.parsing import extract_json_block, format_validation_error

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


class AnthropicExecutor(BaseExecutor):
    """Exécuteur concret utilisant un modèle Anthropic Claude (ex: Sonnet 5.5, Claude Haiku 5.5, Opus 5.5)."""

    def __init__(
        self,
        cost_tracker: CostTracker,
        tool_registry: ToolRegistry,
        model_key: str = "claude-sonnet-5.5",
        max_react_turns: int = 4,
    ) -> None:
        """Initialise l'exécuteur Anthropic Claude."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("La variable d'environnement ANTHROPIC_API_KEY est absente.")
        self._api_key = api_key
        self._cost_tracker = cost_tracker
        self._tool_registry = tool_registry
        self._model_key = model_key
        self._model_id = cost_tracker.get_model_id(model_key)
        self._max_react_turns = max_react_turns

    async def execute(self, step: Step, state: dict[str, Any]) -> ExecutionResult:
        """Exécute une étape via une boucle ReAct courte avec un modèle Anthropic Claude."""
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
        system_prompt = (
            f"{EXECUTOR_SYSTEM_PROMPT}\n\n"
            "Tu DOIS impérativement appeler l'outil `submit_execution_result` pour retourner ta décision structurée."
        )

        for _ in range(self._max_react_turns):
            if self._cost_tracker.is_over_budget():
                return ExecutionResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    output="",
                    justification="Exécution interrompue : plafond budgétaire atteint.",
                    error="BudgetExceeded",
                )

            prompt = (
                FlashExecutor._build_prompt(step, state, tools_spec, observations)
                + "\n\nAppelle maintenant l'outil `submit_execution_result`."
            )

            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self._model_id,
                "max_tokens": 16384,
                "system": system_prompt,
                "tools": [
                    {
                        "name": "submit_execution_result",
                        "description": "Retourne le résultat structuré ExecutionResult de ce tour ReAct.",
                        "input_schema": ExecutionResult.model_json_schema(),
                    }
                ],
                "messages": [{"role": "user", "content": prompt}],
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=body)
                if resp.is_error:
                    raise RuntimeError(
                        f"Erreur API Anthropic ({resp.status_code}) sur {self._model_id} : {resp.text}"
                    )
                data = resp.json()

            usage = data.get("usage", {})
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            self._cost_tracker.record_call(
                self._model_key, input_tokens, output_tokens, role="executor"
            )

            content_blocks = data.get("content", [])
            result: ExecutionResult | None = None
            for block in content_blocks:
                if (
                    block.get("type") == "tool_use"
                    and block.get("name") == "submit_execution_result"
                ):
                    try:
                        result = ExecutionResult.model_validate(block.get("input", {}))
                    except ValidationError as exc:
                        return ExecutionResult(
                            step_id=step.id,
                            status=StepStatus.FAILED,
                            output="",
                            justification="Erreur de validation sur submit_execution_result.",
                            error=format_validation_error(exc),
                        )
                    break

            if result is None:
                raw_text = "\n".join(
                    block.get("text", "") for block in content_blocks if block.get("type") == "text"
                )
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
                        justification="La réponse du modèle Anthropic n'a pas pu être validée.",
                        error=err_msg,
                    )

            last_result = result

            if result.status == StepStatus.FAILED:
                return result

            if not result.tool_calls:
                if result.status == StepStatus.SUCCESS:
                    if step.expected_outputs and self._tool_registry.project_root is not None:
                        missing = verify_expected_outputs(
                            self._tool_registry.project_root, step.expected_outputs
                        )
                        if missing:
                            return ExecutionResult(
                                step_id=step.id,
                                status=StepStatus.FAILED,
                                output=result.output,
                                justification=f"Artefact(s) attendu(s) manquant(s) : {missing}",
                                error="MissingExpectedOutputError",
                            )
                    return result
                return ExecutionResult(
                    step_id=step.id,
                    status=StepStatus.FAILED,
                    output=result.output,
                    justification=(
                        f"Protocole ReAct invalide : statut '{result.status.value}' "
                        "reçu sans aucun appel d'outil (tool_calls vide)."
                    ),
                    error="IncompleteTurnProtocolError",
                )

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
