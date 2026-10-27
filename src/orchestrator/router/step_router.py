from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import yaml

from orchestrator.router.base import BaseRouter
from orchestrator.schemas import Step, StepType


class StepRouter(BaseRouter):
    """Routeur basé sur des règles explicites et le budget restant."""

    def __init__(self, router_config_path: Path) -> None:
        """Charge les seuils du routeur depuis config/router.yaml.

        Args:
            router_config_path: Chemin vers le fichier config/router.yaml.
        """
        raw = yaml.safe_load(router_config_path.read_text(encoding="utf-8")) or {}
        router_cfg = raw.get("router", {})
        self._max_retries = int(router_cfg.get("max_retries_before_escalation", 2))
        self._min_budget_for_planner = float(
            router_cfg.get("min_remaining_budget_for_planner", 0.25)
        )

    async def route(self, step: Step, state: dict[str, Any]) -> Literal["planner", "executor"]:
        """Applique les règles de routage par ordre de priorité.

        Règles :
        1. Si le budget restant est strictement inférieur au seuil configuré -> 'executor'
        2. Si step.type == StepType.REPLAN -> 'planner'
        3. Si le nombre d'échecs de cette étape >= max_retries -> 'planner'
        4. Sinon -> 'executor'
        """
        remaining_budget = float(state.get("remaining_budget", float("inf")))
        if remaining_budget < self._min_budget_for_planner:
            return "executor"

        if step.type == StepType.REPLAN:
            return "planner"

        step_retries: dict[str, int] = state.get("step_retries", {})
        retries_for_step = int(step_retries.get(step.id, 0))
        if retries_for_step >= self._max_retries:
            return "planner"

        return "executor"
