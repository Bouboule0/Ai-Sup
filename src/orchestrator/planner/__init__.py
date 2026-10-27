from __future__ import annotations

from orchestrator.planner.anthropic_planner import AnthropicPlanner
from orchestrator.planner.argon_planner import ArgonPlanner
from orchestrator.planner.base import BasePlanner
from orchestrator.runtime.cost_tracker import CostTracker


def create_planner(cost_tracker: CostTracker, model_key: str = "planner") -> BasePlanner:
    """Instancie le planificateur approprié selon le fournisseur déclaré dans models.yaml."""
    provider = cost_tracker.get_provider(model_key)
    if provider == "google":
        return ArgonPlanner(cost_tracker=cost_tracker, model_key=model_key)
    if provider == "anthropic":
        return AnthropicPlanner(cost_tracker=cost_tracker, model_key=model_key)
    raise ValueError(f"Fournisseur de planificateur non supporté : '{provider}'")


__all__ = ["AnthropicPlanner", "ArgonPlanner", "BasePlanner", "create_planner"]
