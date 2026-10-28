from __future__ import annotations

from rich.console import Console

from orchestrator.ui import SceneMode, TerminalScene


def test_terminal_scene_renders_all_modes() -> None:
    """Vérifie que chaque mode visuel génère une frame Rich Text valide."""
    console = Console(force_terminal=True, width=100)
    scene = TerminalScene(console=console, enabled=True)

    for mode in (
        SceneMode.PLANNING,
        SceneMode.EXECUTING,
        SceneMode.ESCALATION,
        SceneMode.SUCCESS,
        SceneMode.PARTIAL,
    ):
        scene.update(
            mode=mode,
            planner_name="gemini-4-argon",
            executor_name="gemini-3.8-flash",
            current_step_id="step_3",
            current_tool='write_file("parser.py")',
            completed_steps=3,
            total_steps=10,
            current_cost_usd=0.184,
            max_budget_usd=1.50,
            status_message=f"Mode test: {mode.value}",
        )
        frame = scene.render_frame()
        plain = frame.plain
        assert "Welcome to Ai-Sup" in plain
        assert "0.1840 / $1.5000" in plain
        assert f"Mode test: {mode.value}" in plain


def test_terminal_scene_preserves_markup_in_dynamic_text() -> None:
    scene = TerminalScene(console=Console(), enabled=False)
    values = {
        "planner_name": "[blue]planner[/blue]",
        "executor_name": "[red]executor[/red]",
        "current_tool": "write_file [/unknown] [green]file[/green]",
        "status_message": "Failed [/unknown]: [bold]literal[/bold]",
    }
    scene.update(mode=SceneMode.EXECUTING, **values)
    plain = scene.render_frame().plain
    for value in values.values():
        assert value in plain
    scene.update(current_tool="", current_step_id="[/unknown]")
    assert "run([/unknown])" in scene.render_frame().plain
