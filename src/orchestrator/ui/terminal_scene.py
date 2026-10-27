from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import sys
from typing import AsyncIterator
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.text import Text

from orchestrator import __version__


def ensure_utf8_console() -> None:
    """Configure stdout et stderr en UTF-8 sur Windows pour supporter les blocs Unicode (█, ▀, ▄, ░)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


class SceneMode(str, Enum):
    """États visuels de la scène animée du terminal."""

    PLANNING = "planning"
    EXECUTING = "executing"
    ESCALATION = "escalation"
    SUCCESS = "success"
    PARTIAL = "partial"


@dataclass
class SceneState:
    """État télémétrique affiché par la scène animée (compatible CLI, TUI ou futur serveur MCP)."""

    mode: SceneMode = SceneMode.PLANNING
    planner_name: str = "gemini-4-argon"
    executor_name: str = "gemini-3.8-flash"
    current_step_id: str = "init"
    current_tool: str = ""
    completed_steps: int = 0
    total_steps: int = 1
    current_cost_usd: float = 0.0
    max_budget_usd: float = 2.0
    status_message: str = "Initialisation de l'orchestrateur..."
    tick: int = 0


class TerminalScene:
    """Scène animée fidèle au style Claude Code (lune C tramée, nuages ░▒, mascotte à 4 pattes et oreilles)."""

    def __init__(self, console: Console, enabled: bool = True, fps: float = 4.0) -> None:
        """Initialise la scène animée.

        Args:
            console: Instance Rich Console.
            enabled: Si False, désactive l'animation Live (utile pour CI/CD ou MCP stdio).
            fps: Fréquence de rafraîchissement en images par seconde.
        """
        ensure_utf8_console()
        self._console = console
        self._enabled = enabled
        self._fps = fps
        self.state = SceneState()
        self._live: Live | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False

    def update(
        self,
        *,
        mode: SceneMode | None = None,
        planner_name: str | None = None,
        executor_name: str | None = None,
        current_step_id: str | None = None,
        current_tool: str | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        current_cost_usd: float | None = None,
        max_budget_usd: float | None = None,
        status_message: str | None = None,
    ) -> None:
        """Met à jour les informations de la scène et rafraîchit l'affichage."""
        if mode is not None:
            self.state.mode = mode
        if planner_name is not None:
            self.state.planner_name = planner_name
        if executor_name is not None:
            self.state.executor_name = executor_name
        if current_step_id is not None:
            self.state.current_step_id = current_step_id
        if current_tool is not None:
            self.state.current_tool = current_tool
        if completed_steps is not None:
            self.state.completed_steps = completed_steps
        if total_steps is not None:
            self.state.total_steps = max(1, total_steps)
        if current_cost_usd is not None:
            self.state.current_cost_usd = current_cost_usd
        if max_budget_usd is not None:
            self.state.max_budget_usd = max_budget_usd
        if status_message is not None:
            self.state.status_message = status_message

        if self._live is not None:
            self._live.update(self.render_frame())

    def render_frame(self) -> Text:
        """Construit une frame Rich Text reproduisant la mascotte et le décor Claude Code."""
        t = self.state.tick
        s = self.state

        star1 = "*" if t % 2 == 0 else "·"
        star2 = "*" if (t + 1) % 3 == 0 else " "
        star3 = "*" if t % 4 != 0 else "·"

        bubble_left = "                  "
        bubble_left_color = "bright_blue"
        bubble_right = ""
        bubble_right_color = "yellow"
        child_offset = 6
        child_color = "#d97757"  # Terracotta exact de Claude Code
        child_eye = "▪"
        # Les 4 petites pattes caractéristiques du bonhomme Claude Code
        child_legs = "█ █  █ █" if t % 2 == 0 else "▀ █  ▀ █"
        status_dot_color = "#fbbf24"

        if s.mode == SceneMode.PLANNING:
            dots = ("." * ((t % 3) + 1)).ljust(3)
            bubble_left = f"   [Plan{dots}]      "
            child_offset = 4
            child_legs = "█ █  █ █"
            status_dot_color = "bright_blue"
        elif s.mode == SceneMode.EXECUTING:
            child_offset = 5 + (t % 14)
            tool_label = escape(s.current_tool or f"run({s.current_step_id})")
            bubble_right = f"⚡ {tool_label}"
            status_dot_color = "yellow"
        elif s.mode == SceneMode.ESCALATION:
            bubble_left = "   [Replan!]      "
            bubble_left_color = "magenta"
            child_offset = 4 + (t % 2)
            child_color = "bright_red"
            child_eye = "×"
            child_legs = "█ █  █ █"
            bubble_right = "! Échec x2 -> Escalade StepRouter"
            bubble_right_color = "bright_red"
            status_dot_color = "magenta"
        elif s.mode == SceneMode.PARTIAL:
            bubble_left = "   Étapes ignorées "
            bubble_left_color = "yellow"
            bubble_right = f"{s.completed_steps}/{s.total_steps} étapes validées"
            status_dot_color = "yellow"
        else:  # SUCCESS
            bubble_left = "   ★ Bravo !      "
            bubble_left_color = "bright_green"
            child_offset = 8
            child_legs = "▀ ▀  ▀ ▀" if t % 2 == 0 else "█ █  █ █"
            bubble_right = f"✓ {s.completed_steps}/{s.total_steps} étapes validées !"
            bubble_right_color = "bright_green"
            status_dot_color = "bright_green"

        pad = " " * child_offset
        bar_width = 10
        filled = min(bar_width, int(round((s.completed_steps / max(1, s.total_steps)) * bar_width)))
        progress_bar = "█" * filled + "░" * (bar_width - filled)

        # Silhouette exacte : tête rectangulaire, oreilles sur les côtés + 2 yeux carrés noirs, bas du corps, et 4 pattes sur la ligne pointillée
        markup_lines = [
            f"[bold #d97757]Welcome to Ai-Sup[/bold #d97757] [dim]v{__version__}[/dim] · "
            f"[bright_blue]{escape(s.planner_name)}[/bright_blue] [dim](Planner)[/dim] + "
            f"[{child_color}]{escape(s.executor_name)}[/{child_color}] [dim](Executor)[/dim]",
            "[dim]" + ("·" * 76) + "[/dim]",
            f"      [white]{star1}[/white]                             [dim]{star2}[/dim]             [white]██████[/white][dim]▒▒░[/dim]",
            "              [dim]░░░░░░[/dim]                            [white]██[/white][dim]▓▓[/dim]      [dim]▓▓[/dim]",
            f"        [dim]░░░░  ░░░░░░░░░░[/dim]      [white]{star3}[/white]               [white]███[/white][dim]▒░[/dim]",
            "      [dim]░░░░░░░░░░░░░░░░░░░░[/dim]                    [white]███[/white][dim]▒░[/dim]",
            "                                    [dim]░░░░[/dim]        [white]██[/white][dim]▓▓[/dim]      [dim]▓[/dim][white]█[/white]",
            f"  [dim]*[/dim]                               [dim]░░░░░░░░[/dim]        [dim]▓▒▒[/dim][white]███[/white][dim]▒▒▓[/dim]",
            f"[{bubble_left_color}]{bubble_left}[/{bubble_left_color}]              [dim]░░░░░░░░░░░░░░[/dim]                        [dim]*[/dim]",
            f"    [bright_blue]████████[/bright_blue]    {pad} [{child_color}]████████[/{child_color}]",
            f"   [bright_blue]██[/bright_blue][black on bright_blue]▪[/black on bright_blue][bright_blue]████[/bright_blue][black on bright_blue]▪[/black on bright_blue][bright_blue]██[/bright_blue]   "
            f"{pad}[{child_color}]██[/{child_color}][black on {child_color}]{child_eye}[/black on {child_color}][{child_color}]████[/{child_color}][black on {child_color}]{child_eye}[/black on {child_color}][{child_color}]██[/{child_color}]"
            + (f"  [{bubble_right_color}]{bubble_right}[/{bubble_right_color}]" if bubble_right else ""),
            f"    [bright_blue]████████[/bright_blue]    {pad} [{child_color}]████████[/{child_color}]",
            f"[dim]····[/dim][bright_blue]█ █  █ █[/bright_blue][dim]····[/dim]{pad} [{child_color}]{child_legs}[/{child_color}][dim]"
            + ("·" * max(10, 46 - child_offset))
            + "[/dim]",
            "",
            f"[{status_dot_color}]●[/{status_dot_color}] [bold]{escape(s.status_message)}[/bold]",
            f"Progression : [bright_green]{progress_bar} {s.completed_steps}/{s.total_steps}[/bright_green]   |   "
            f"Budget : [bold yellow]${s.current_cost_usd:.4f} / ${s.max_budget_usd:.4f}[/bold yellow]",
        ]
        return Text.from_markup("\n".join(markup_lines))

    async def _animate_loop(self) -> None:
        """Boucle asynchrone de rafraîchissement des frames."""
        interval = 1.0 / max(1.0, self._fps)
        while self._running:
            await asyncio.sleep(interval)
            self.state.tick += 1
            if self._live is not None:
                self._live.update(self.render_frame())

    @asynccontextmanager
    async def session(self) -> AsyncIterator[TerminalScene]:
        """Context manager asynchrone démarrant et arrêtant proprement l'animation Rich Live."""
        if not self._enabled:
            yield self
            return

        self._running = True
        with Live(
            self.render_frame(),
            console=self._console,
            refresh_per_second=max(4, int(self._fps)),
            transient=False,
        ) as live:
            self._live = live
            self._task = asyncio.create_task(self._animate_loop())
            try:
                yield self
            finally:
                self._running = False
                if self._task is not None:
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        pass
                self._live.update(self.render_frame())
                self._live = None
