from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from orchestrator.router.step_router import StepRouter
from orchestrator.schemas import Step, StepType


@pytest.fixture()
def router_config_file(tmp_path: Path) -> Path:
    """Crée un fichier router.yaml temporaire."""
    cfg = tmp_path / "router.yaml"
    cfg.write_text(
        """
router:
  max_retries_before_escalation: 2
  min_remaining_budget_for_planner: 0.25
shell:
  allowed_commands:
    - "pytest"
""",
        encoding="utf-8",
    )
    return cfg


def _make_step(step_type: StepType = StepType.CODE) -> Step:
    return Step(
        id="step_1",
        description="Étape de test",
        type=step_type,
        tools_allowed=[],
        success_criteria="OK",
        expected_outputs=[],
        depends_on=[],
    )


def test_router_default_to_executor(router_config_file: Path) -> None:
    """Vérifie qu'une étape normale avec budget suffisant part vers l'exécuteur."""
    router = StepRouter(router_config_file)
    step = _make_step(StepType.CODE)
    decision = asyncio.run(
        router.route(step, {"remaining_budget": 1.50, "step_retries": {"step_1": 0}})
    )
    assert decision == "executor"


def test_router_escalates_on_replan_type(router_config_file: Path) -> None:
    """Vérifie que StepType.REPLAN est routé vers le planificateur."""
    router = StepRouter(router_config_file)
    step = _make_step(StepType.REPLAN)
    decision = asyncio.run(
        router.route(step, {"remaining_budget": 1.50, "step_retries": {}})
    )
    assert decision == "planner"


def test_router_escalates_on_repeated_failures(router_config_file: Path) -> None:
    """Vérifie qu'une étape ayant échoué >= 2 fois est routée vers le planificateur."""
    router = StepRouter(router_config_file)
    step = _make_step(StepType.CODE)
    decision = asyncio.run(
        router.route(step, {"remaining_budget": 1.50, "step_retries": {"step_1": 2}})
    )
    assert decision == "planner"


def test_router_forces_executor_on_low_budget(router_config_file: Path) -> None:
    """Vérifie que si le budget restant est inférieur au seuil, l'exécuteur est forcé même en REPLAN."""
    router = StepRouter(router_config_file)
    step = _make_step(StepType.REPLAN)
    decision = asyncio.run(
        router.route(step, {"remaining_budget": 0.10, "step_retries": {"step_1": 5}})
    )
    assert decision == "executor"
