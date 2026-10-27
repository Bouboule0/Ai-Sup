from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from orchestrator import cli
from orchestrator.schemas import ExecutionResult, Plan, Step, StepStatus, StepType
from orchestrator.ui import SceneMode, TerminalScene


def make_step(step_id: str, step_type: StepType = StepType.CODE, **kwargs) -> Step:
    return Step(id=step_id, type=step_type, success_criteria="Done", **{"description": "Do work", **kwargs})


def test_plan_renders_model_text_literally(monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=500))
    plan = Plan(
        goal="goal [bold]literal[/bold]",
        summary="summary [/unknown] [red]literal[/red]",
        steps=[make_step("[cyan]id[/cyan]", description="description [/unknown]",
                         tools_allowed=["[green]tool[/green]"], depends_on=["[yellow]dep[/yellow]"])],
    )
    cli._render_plan(plan, "[blue]planner[/blue]")
    rendered = output.getvalue()
    for value in (plan.goal, plan.summary, plan.steps[0].id, plan.steps[0].description,
                  *plan.steps[0].tools_allowed, *plan.steps[0].depends_on, "[blue]planner[/blue]"):
        assert value in rendered


@pytest.mark.parametrize("replan", [False, True])
@pytest.mark.parametrize("decline", [False, True])
def test_interactive_completion_and_replan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replan: bool, decline: bool,
) -> None:
    first = make_step("skipped")
    second = make_step("work", StepType.REPLAN if replan else StepType.CODE)
    plans = iter([
        Plan(goal="goal", summary="initial", steps=[first, second]),
        Plan(goal="goal", summary="replanned", steps=[first, make_step("work")]),
    ])
    executed: list[str] = []

    async def plan(**kwargs):
        return next(plans)

    async def execute(step, state):
        executed.append(step.id)
        return ExecutionResult(step_id=step.id, status=StepStatus.SUCCESS, output="", justification="Done")

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(cli, "create_planner", lambda **kwargs: SimpleNamespace(plan=plan))
    monkeypatch.setattr(cli, "create_executor", lambda **kwargs: SimpleNamespace(execute=execute))
    monkeypatch.setattr(cli.asyncio, "sleep", no_sleep)
    scene = TerminalScene(console=cli.console, enabled=False)
    monkeypatch.setattr(cli, "TerminalScene", lambda **kwargs: scene)
    result = CliRunner().invoke(cli.app, [
        "run", "--goal", "goal", "--project", str(tmp_path), "--interactive", "--no-anim",
    ], input=("n\n" if decline else "y\n") + "y\ny\n")
    assert result.exit_code == (1 if decline else 0), result.output
    assert executed == (["work"] if decline else ["skipped", "work"])
    assert result.output.count("Exécuter l'étape 'skipped'") == 1
    assert scene.state.completed_steps == (1 if decline else 2)
    assert scene.state.mode == (SceneMode.PARTIAL if decline else SceneMode.SUCCESS)
    if decline:
        assert "Exécution partielle : étapes ignorées : skipped" in result.output
        assert "Objectif terminé avec succès" not in result.output
    else:
        assert "Objectif terminé avec succès" in result.output


@pytest.mark.parametrize("phase", ["plan", "initial", "replan"])
def test_invalid_plan_errors_render_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    monkeypatch.setattr(cli, "console", Console(width=240))
    invalid = Plan(goal="goal", summary="invalid", steps=[
        make_step("bad", depends_on=["[/unknown] [red]missing[/red]"]),
    ])
    plans = iter([
        Plan(goal="goal", summary="initial", steps=[make_step("replan", StepType.REPLAN)]),
        invalid,
    ] if phase == "replan" else [invalid])

    async def plan(**kwargs):
        return next(plans)

    monkeypatch.setattr(cli, "create_planner", lambda **kwargs: SimpleNamespace(plan=plan))
    monkeypatch.setattr(cli, "create_executor", lambda **kwargs: SimpleNamespace())
    args = ["plan", "--goal", "goal", "--no-anim"] if phase == "plan" else [
        "run", "--goal", "goal", "--project", str(tmp_path), "--no-anim",
    ]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1
    assert "[/unknown] [red]missing[/red]" in result.output
    assert "invalide" in result.output
