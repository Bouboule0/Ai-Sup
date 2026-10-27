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
