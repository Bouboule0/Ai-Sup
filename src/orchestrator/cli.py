from __future__ import annotations

import asyncio
from pathlib import Path
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from orchestrator import __version__
from orchestrator.benchmark import (
    format_benchmark_markdown_table,
    render_benchmark_rich_table,
    run_comparative_benchmark,
)
from orchestrator.executor import create_executor
from orchestrator.planner import create_planner
from orchestrator.router.step_router import StepRouter
from orchestrator.runtime.context import ExecutionContext
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import Plan, StepStatus
from orchestrator.tools.file_system import register_filesystem_tools
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.shell import register_shell_tools

app = typer.Typer(
    name="orch",
    help="Orchestrateur Plan-then-Execute Multi-Modèles (Gemini & Claude).",
    add_completion=False,
)
console = Console()


def _resolve_config_dir() -> Path:
    """Localise le dossier config/ du dépôt."""
    candidates = [
        Path.cwd() / "config",
        Path(__file__).resolve().parents[2] / "config",
    ]
    for candidate in candidates:
        if (candidate / "models.yaml").exists():
            return candidate
    raise FileNotFoundError("Impossible de localiser le dossier 'config/' (models.yaml).")


def _render_plan(plan: Plan, planner_key: str) -> None:
    """Affiche un Plan dans une table Rich."""
    table = Table(title=f"Plan ({planner_key}) : {plan.goal}")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Outils", style="green")
    table.add_column("Dépendances", style="yellow")

    for step in plan.steps:
        table.add_row(
            step.id,
            step.type.value,
            step.description,
            ", ".join(step.tools_allowed) or "-",
            ", ".join(step.depends_on) or "-",
        )
    console.print(Panel(plan.summary, title=f"Stratégie ({planner_key})", border_style="blue"))
    console.print(table)


async def _async_plan(goal: str, max_budget: float, planner_model: str) -> None:
    """Exécute uniquement la phase de planification et affiche le résultat."""
    config_dir = _resolve_config_dir()
    tracker = CostTracker(max_budget=max_budget, models_config_path=config_dir / "models.yaml")
    planner = create_planner(cost_tracker=tracker, model_key=planner_model)

    plan_obj = await planner.plan(goal=goal, context={})
    _render_plan(plan_obj, planner_model)
    console.print(
        f"[bold green]Coût planification :[/bold green] ${tracker.total_cost():.6f} "
        f"| [bold cyan]Budget restant :[/bold cyan] ${tracker.remaining_budget():.6f}"
    )


async def _async_run(
    goal: str,
    project_path: Path,
    max_budget: float,
    interactive: bool,
    pairing: str | None,
    planner_model: str,
    executor_model: str,
) -> None:
    """Exécute le cycle complet Plan-then-Execute avec le couple de modèles sélectionné."""
    config_dir = _resolve_config_dir()
    models_yaml = config_dir / "models.yaml"
    router_yaml = config_dir / "router.yaml"

    tracker = CostTracker(max_budget=max_budget, models_config_path=models_yaml)
    if pairing:
        pairing_cfg = tracker.get_pairing(pairing)
        planner_key = pairing_cfg.planner
        executor_key = pairing_cfg.executor
    else:
        planner_key = planner_model
        executor_key = executor_model

    context = ExecutionContext(project_root=project_path)

    registry = ToolRegistry()
    register_filesystem_tools(registry, context.project_root)
    register_shell_tools(registry, context.project_root, router_yaml)

    planner = create_planner(cost_tracker=tracker, model_key=planner_key)
    executor = create_executor(
        cost_tracker=tracker,
        tool_registry=registry,
        model_key=executor_key,
    )
    router = StepRouter(router_config_path=router_yaml)

    console.print(
        Panel(
            f"Couple actif : [bold blue]{planner_key}[/bold blue] (Planner) → "
            f"[bold green]{executor_key}[/bold green] (Executor)\n"
            f"Objectif : [bold]{goal}[/bold]"
        )
    )
    current_plan = await planner.plan(goal=goal, context=context.snapshot())
    _render_plan(current_plan, planner_key)

    step_index = 0
    while step_index < len(current_plan.steps):
        if tracker.is_over_budget():
            console.print("[bold red]Arrêt : budget maximum dépassé ![/bold red]")
            raise typer.Exit(code=1)

        step = current_plan.steps[step_index]
        completed: list[str] = context.get("completed_steps", [])
        unmet_deps = [dep for dep in step.depends_on if dep not in completed]
        if unmet_deps:
            console.print(
                f"[bold red]Dépendances non satisfaites pour {step.id} : {unmet_deps}[/bold red]"
            )
            raise typer.Exit(code=1)

        if interactive:
            proceed = typer.confirm(f"Exécuter l'étape '{step.id}' ({step.description}) ?")
            if not proceed:
                console.print(f"[yellow]Étape {step.id} ignorée par l'utilisateur.[/yellow]")
                step_index += 1
                continue

        state_for_router = context.snapshot()
        state_for_router["remaining_budget"] = tracker.remaining_budget()
        target = await router.route(step, state_for_router)

        if target == "planner":
            console.print(
                f"[bold magenta]Escalade vers {planner_key} (replanification) à l'étape {step.id}...[/bold magenta]"
            )
            current_plan = await planner.plan(goal=goal, context=context.snapshot())
            _render_plan(current_plan, planner_key)
            context.set("step_retries", {})
            step_index = 0
            continue

        console.print(f"[cyan]Exécution de {step.id} via {executor_key}...[/cyan]")
        result = await executor.execute(step, context.snapshot())

        if result.status == StepStatus.SUCCESS:
            completed.append(step.id)
            context.set("completed_steps", completed)
            outputs = context.get("step_outputs", {})
            outputs[step.id] = result.output
            context.set("step_outputs", outputs)
            console.print(
                f"[bold green]✓ {step.id} terminé[/bold green] — {result.justification}\n"
                f"  Coût cumulé : ${tracker.total_cost():.6f} | "
                f"Budget restant : ${tracker.remaining_budget():.6f}"
            )
            step_index += 1
        else:
            retries = context.get("step_retries", {})
            retries[step.id] = int(retries.get(step.id, 0)) + 1
            context.set("step_retries", retries)
            console.print(
                f"[bold red]✗ Échec sur {step.id} (tentative {retries[step.id]}) :[/bold red] "
                f"{result.justification}"
            )
            if tracker.remaining_budget() < 0.25 and retries[step.id] >= 2:
                console.print(
                    "[bold red]Arrêt : échecs répétés et budget insuffisant pour replanifier.[/bold red]"
                )
                raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold green]Objectif terminé avec succès ![/bold green]\n"
            f"Coût Planner ({planner_key}) : ${tracker.planner_cost():.6f} | "
            f"Coût Executor ({executor_key}) : ${tracker.executor_cost():.6f}\n"
            f"Coût total : ${tracker.total_cost():.6f} | "
            f"Budget restant : ${tracker.remaining_budget():.6f}",
            border_style="green",
        )
    )


@app.command()
def run(
    goal: str = typer.Option(..., "--goal", help="Objectif logiciel à accomplir."),
    project: Path = typer.Option(..., "--project", help="Chemin du répertoire projet cible."),
    max_budget: float = typer.Option(2.0, "--max-budget", help="Budget maximum en USD."),
    pairing: str | None = typer.Option(
        None,
        "--pairing",
        help="Identifiant d'un couple préconfiguré (ex: argon-flash, argon-pro, opus-sonnet, opus-flash).",
    ),
    planner_model: str = typer.Option(
        "planner",
        "--planner",
        help="Modèle planificateur si --pairing n'est pas utilisé (ex: gemini-4-argon, claude-opus-5.5).",
    ),
    executor_model: str = typer.Option(
        "executor",
        "--executor",
        help="Modèle exécuteur si --pairing n'est pas utilisé (ex: gemini-3.8-flash, claude-sonnet-5.5).",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Demander confirmation avant chaque étape."
    ),
) -> None:
    """Planifie et exécute un objectif sur un projet cible avec le couple d'IA choisi."""
    load_dotenv()
    asyncio.run(
        _async_run(
            goal=goal,
            project_path=project,
            max_budget=max_budget,
            interactive=interactive,
            pairing=pairing,
            planner_model=planner_model,
            executor_model=executor_model,
        )
    )


@app.command()
def plan(
    goal: str = typer.Option(..., "--goal", help="Objectif à planifier uniquement."),
    max_budget: float = typer.Option(1.0, "--max-budget", help="Budget maximum en USD."),
    planner_model: str = typer.Option(
        "planner",
        "--planner",
        help="Modèle planificateur (ex: gemini-4-argon, claude-opus-5.5).",
    ),
) -> None:
    """Génère et affiche uniquement le plan d'exécution."""
    load_dotenv()
    asyncio.run(_async_plan(goal=goal, max_budget=max_budget, planner_model=planner_model))


@app.command()
def pairings() -> None:
    """Liste tous les couples (Planner -> Executor) disponibles dans config/models.yaml."""
    config_dir = _resolve_config_dir()
    tracker = CostTracker(max_budget=10.0, models_config_path=config_dir / "models.yaml")
    table = Table(title="Couples d'IA disponibles (Planner -> Executor)")
    table.add_column("Pairing ID", style="bold cyan")
    table.add_column("Planner (Parent)", style="blue")
    table.add_column("Tarif Planner (In/Out 1M)", justify="right")
    table.add_column("Executor (Child)", style="green")
    table.add_column("Tarif Executor (In/Out 1M)", justify="right")
    table.add_column("Description", style="white")

    for p in tracker.list_pairings():
        p_price = tracker.get_model_pricing(p.planner)
        e_price = tracker.get_model_pricing(p.executor)
        table.add_row(
            p.pairing_id,
            f"{p.planner} ({p_price.provider})",
            f"${p_price.input_cost_per_million:.2f} / ${p_price.output_cost_per_million:.2f}",
            f"{p.executor} ({e_price.provider})",
            f"${e_price.input_cost_per_million:.2f} / ${e_price.output_cost_per_million:.2f}",
            p.description,
        )
    console.print(table)


@app.command()
def benchmark(
    output_md: Path | None = typer.Option(
        None,
        "--output-md",
        help="Chemin optionnel pour sauvegarder le tableau comparatif en Markdown.",
    ),
    max_budget: float = typer.Option(25.0, "--max-budget", help="Budget plafond par couple."),
) -> None:
    """Exécute le benchmark comparatif multi-modèles et affiche le tableau récapitulatif."""
    config_dir = _resolve_config_dir()
    entries = asyncio.run(
        run_comparative_benchmark(
            models_config_path=config_dir / "models.yaml",
            router_config_path=config_dir / "router.yaml",
            max_budget=max_budget,
        )
    )
    render_benchmark_rich_table(entries, console)
    if output_md is not None:
        md_content = format_benchmark_markdown_table(entries)
        output_md.write_text(md_content + "\n", encoding="utf-8")
        console.print(f"[bold green]Tableau Markdown exporté vers : {output_md}[/bold green]")


@app.command()
def version() -> None:
    """Affiche la version de l'orchestrateur."""
    console.print(f"ai-sup (gemini-orchestrator) v{__version__}")


if __name__ == "__main__":
    app()
