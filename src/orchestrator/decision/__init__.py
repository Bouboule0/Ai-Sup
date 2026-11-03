from __future__ import annotations

from typing import Any

from orchestrator.decision.base import BaseDecisionModel
from orchestrator.decision.jev_decision import JevDecisionError, JevDecisionModel
from orchestrator.runtime.cost_tracker import CostTracker


def create_decision_model(
    cost_tracker: CostTracker,
    model_key: str = "jev",
    **kwargs: Any,
) -> BaseDecisionModel:
    """Instancie le modèle de décision approprié selon le fournisseur déclaré dans models.yaml."""
    provider = cost_tracker.get_provider(model_key)
    if provider != "ofoxai":
        raise ValueError(f"Fournisseur de modèle de décision non supporté : '{provider}'")
    return JevDecisionModel(cost_tracker=cost_tracker, model_key=model_key, **kwargs)


__all__ = [
    "BaseDecisionModel",
    "JevDecisionError",
    "JevDecisionModel",
    "create_decision_model",
]
