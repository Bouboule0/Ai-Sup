from __future__ import annotations

from orchestrator.executor.anthropic_executor import AnthropicExecutor
from orchestrator.executor.base import BaseExecutor
from orchestrator.executor.flash_executor import FlashExecutor
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.tools.registry import ToolRegistry


def create_executor(
    cost_tracker: CostTracker,
    tool_registry: ToolRegistry,
    model_key: str = "executor",
    max_react_turns: int = 4,
) -> BaseExecutor:
    """Instancie l'exécuteur approprié selon le fournisseur déclaré dans models.yaml."""
    provider = cost_tracker.get_provider(model_key)
    if provider == "google":
        return FlashExecutor(
            cost_tracker=cost_tracker,
            tool_registry=tool_registry,
            model_key=model_key,
            max_react_turns=max_react_turns,
        )
    if provider == "anthropic":
        return AnthropicExecutor(
            cost_tracker=cost_tracker,
            tool_registry=tool_registry,
            model_key=model_key,
            max_react_turns=max_react_turns,
        )
    raise ValueError(f"Fournisseur d'exécuteur non supporté : '{provider}'")


__all__ = ["AnthropicExecutor", "BaseExecutor", "FlashExecutor", "create_executor"]
