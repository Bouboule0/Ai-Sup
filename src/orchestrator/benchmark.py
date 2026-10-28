from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any
from rich.console import Console
from rich.table import Table

from orchestrator.executor.base import BaseExecutor
from orchestrator.planner.base import BasePlanner
from orchestrator.router.step_router import StepRouter
from orchestrator.runtime.context import ExecutionContext
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import (
    BenchmarkEntry,
    ExecutionResult,
    Plan,
    Step,
    StepStatus,
    StepType,
)


@dataclass(frozen=True)
class ModelSimulationProfile:
    """Profil empirique d'un modèle pour la simulation déterministe de benchmark."""

    plan_input_tokens: int
    plan_output_tokens: int
    plan_latency_s: float
    exec_input_tokens_per_turn: int
    exec_output_tokens_per_turn: int
    exec_turns_per_step: int
    exec_latency_per_turn_s: float
    complex_step_failures_before_replan: int


SIMULATION_PROFILES: dict[str, ModelSimulationProfile] = {
    "gemini-4-argon": ModelSimulationProfile(
        plan_input_tokens=32_000,
        plan_output_tokens=6_500,
        plan_latency_s=8.5,
        exec_input_tokens_per_turn=38_000,
        exec_output_tokens_per_turn=7_500,
        exec_turns_per_step=2,
        exec_latency_per_turn_s=7.2,
        complex_step_failures_before_replan=0,
    ),
    "gemini-3.1-pro": ModelSimulationProfile(
        plan_input_tokens=32_000,
        plan_output_tokens=6_000,
        plan_latency_s=6.0,
        exec_input_tokens_per_turn=36_000,
        exec_output_tokens_per_turn=7_000,
        exec_turns_per_step=2,
        exec_latency_per_turn_s=4.1,
        complex_step_failures_before_replan=0,
    ),
    "gemini-3.8-flash": ModelSimulationProfile(
        plan_input_tokens=30_000,
        plan_output_tokens=5_500,
        plan_latency_s=2.8,
        exec_input_tokens_per_turn=35_000,
        exec_output_tokens_per_turn=6_800,
        exec_turns_per_step=2,
        exec_latency_per_turn_s=1.9,
        complex_step_failures_before_replan=2,  # Déclenche 1 escalade vers Argon sur l'étape complexe
    ),
    "claude-opus-5.5": ModelSimulationProfile(
        plan_input_tokens=34_000,
        plan_output_tokens=7_000,
        plan_latency_s=9.8,
        exec_input_tokens_per_turn=40_000,
        exec_output_tokens_per_turn=8_000,
        exec_turns_per_step=2,
        exec_latency_per_turn_s=8.4,
        complex_step_failures_before_replan=0,
    ),
    "claude-sonnet-5.5": ModelSimulationProfile(
        plan_input_tokens=32_000,
        plan_output_tokens=6_200,
        plan_latency_s=5.2,
        exec_input_tokens_per_turn=36_000,
        exec_output_tokens_per_turn=7_200,
        exec_turns_per_step=2,
        exec_latency_per_turn_s=3.4,
        complex_step_failures_before_replan=0,
    ),
    "claude-haiku-5.5": ModelSimulationProfile(
        plan_input_tokens=28_000,
        plan_output_tokens=5_000,
        plan_latency_s=2.1,
        exec_input_tokens_per_turn=34_000,
        exec_output_tokens_per_turn=6_500,
        exec_turns_per_step=3,
        exec_latency_per_turn_s=1.5,
        complex_step_failures_before_replan=2,  # Déclenche 1 escalade vers le Planner sur l'étape complexe
    ),
}


def build_benchmark_plan() -> Plan:
    """Construit le plan de référence à 10 étapes pour évaluer tous les couples d'IA."""
    steps = [
        Step(
            id="step_1",
            description="Inspecter la structure du dépôt et lire le cahier des charges.",
            type=StepType.FILE_OP,
            tools_allowed=["list_dir", "read_file"],
            success_criteria="Structure du projet inventoriée.",
            expected_outputs=[],
            depends_on=[],
        ),
        Step(
            id="step_2",
            description="Créer les modèles de données Pydantic dans models.py.",
            type=StepType.CODE,
            tools_allowed=["write_file"],
            success_criteria="models.py valide et typé.",
            expected_outputs=["models.py"],
            depends_on=["step_1"],
        ),
        Step(
            id="step_3",
            description="Implémenter le parseur CSV/JSON tolérant aux erreurs dans parser.py.",
            type=StepType.CODE,
            tools_allowed=["write_file", "read_file"],
            success_criteria="parser.py gère les lignes malformées.",
            expected_outputs=["parser.py"],
            depends_on=["step_2"],
        ),
        Step(
            id="step_4",
            description="Implémenter l'algorithme de résolution de graphe de dépendances cycliques.",
            type=StepType.CODE,
            tools_allowed=["write_file", "read_file"],
            success_criteria="Tri topologique exact avec détection de cycles.",
            expected_outputs=["graph_solver.py"],
            depends_on=["step_2"],
        ),
        Step(
            id="step_5",
            description="Créer l'interface CLI Typer dans app.py.",
            type=StepType.CODE,
            tools_allowed=["write_file"],
            success_criteria="Commandes CLI exposées proprement.",
            expected_outputs=["app.py"],
            depends_on=["step_3", "step_4"],
        ),
        Step(
            id="step_6",
            description="Écrire les tests unitaires pour models.py et parser.py.",
            type=StepType.VERIFY,
            tools_allowed=["write_file"],
            success_criteria="Couverture des cas nominaux et erreurs.",
            expected_outputs=["test_parser.py"],
            depends_on=["step_3"],
        ),
        Step(
            id="step_7",
            description="Écrire les tests unitaires pour graph_solver.py.",
            type=StepType.VERIFY,
            tools_allowed=["write_file"],
            success_criteria="Tests de détection de cycles inclus.",
            expected_outputs=["test_graph.py"],
            depends_on=["step_4"],
        ),
        Step(
            id="step_8",
            description="Exécuter la suite pytest et analyser les résultats.",
            type=StepType.SHELL,
            tools_allowed=["run_command"],
            success_criteria="returncode=0 sur pytest.",
            expected_outputs=[],
            depends_on=["step_6", "step_7"],
        ),
        Step(
            id="step_9",
            description="Corriger les cas limites identifiés et optimiser les imports.",
            type=StepType.CODE,
            tools_allowed=["read_file", "write_file"],
            success_criteria="Code propre et conforme PEP8.",
            expected_outputs=["graph_solver.py"],
            depends_on=["step_8"],
        ),
        Step(
            id="step_10",
            description="Vérification finale complète de la suite de tests et génération du rapport.",
            type=StepType.VERIFY,
            tools_allowed=["run_command", "write_file"],
            success_criteria="100% des tests passent.",
            expected_outputs=["REPORT.md"],
            depends_on=["step_9"],
        ),
    ]
    return Plan(
        goal="Développer un pipeline CLI complet avec solveur de graphe et suite de tests",
        summary="Plan en 10 étapes atomiques combinant opérations fichiers, algorithmique et validation pytest.",
        steps=steps,
    )


def _resolve_simulation_profile(cost_tracker: CostTracker, model_key: str) -> ModelSimulationProfile:
    pricing = cost_tracker.get_model_pricing(model_key)
    profile = SIMULATION_PROFILES.get(pricing.key) or SIMULATION_PROFILES.get(pricing.model_id)
    if profile is None:
        raise ValueError(
            f"No simulation profile for configuration key '{pricing.key}' "
            f"or model ID '{pricing.model_id}'."
        )
    return profile


class SimulatedPlanner(BasePlanner):
    """Planificateur de simulation enregistrant les tokens et la latence réalistes du modèle."""

    def __init__(self, cost_tracker: CostTracker, model_key: str) -> None:
        self._cost_tracker = cost_tracker
        self._model_key = model_key
        self._profile = _resolve_simulation_profile(cost_tracker, model_key)
        self.total_latency_s = 0.0
        self.plan_calls = 0

    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Simule la génération ou la replanification d'un Plan validé."""
        self.plan_calls += 1
        self.total_latency_s += self._profile.plan_latency_s
        self._cost_tracker.record_call(
            self._model_key,
            input_tokens=self._profile.plan_input_tokens,
            output_tokens=self._profile.plan_output_tokens,
            role="planner",
        )
        return build_benchmark_plan()


class SimulatedExecutor(BaseExecutor):
    """Exécuteur de simulation reproduisant les boucles ReAct et les échecs sur étapes complexes."""

    def __init__(self, cost_tracker: CostTracker, model_key: str) -> None:
        self._cost_tracker = cost_tracker
        self._model_key = model_key
        self._profile = _resolve_simulation_profile(cost_tracker, model_key)
        self.total_latency_s = 0.0
        self.react_turns = 0
        self._replanned_steps: set[str] = set()

    def mark_replanned(self, step_id: str) -> None:
        """Indique que le Planner a fourni un sous-plan clarifié après escalade."""
        self._replanned_steps.add(step_id)

    async def execute(self, step: Step, state: dict[str, Any]) -> ExecutionResult:
        """Simule l'exécution ReAct d'une étape et consomme les tokens correspondants."""
        turns = self._profile.exec_turns_per_step
        for _ in range(turns):
            self.react_turns += 1
            self.total_latency_s += self._profile.exec_latency_per_turn_s
            self._cost_tracker.record_call(
                self._model_key,
                input_tokens=self._profile.exec_input_tokens_per_turn,
                output_tokens=self._profile.exec_output_tokens_per_turn,
                role="executor",
            )

        retries_map: dict[str, int] = state.get("step_retries", {})
        current_retries = int(retries_map.get(step.id, 0))

        # L'étape algorithmique complexe (step_4) teste la capacité de raisonnement de l'exécuteur
        if (
            step.id == "step_4"
            and step.id not in self._replanned_steps
            and current_retries < self._profile.complex_step_failures_before_replan
        ):
            return ExecutionResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                output="",
                justification="Échec d'assertion sur la détection de cycle complexe avant clarification du plan.",
                error="AssertionError: CycleDetectionFailed",
            )

        return ExecutionResult(
            step_id=step.id,
            status=StepStatus.SUCCESS,
            output=f"Artefacts générés pour {step.id}",
            justification=f"Critère '{step.success_criteria}' validé en {turns} tours ReAct.",
        )


async def run_single_pairing_benchmark(
    pairing_id: str,
    models_config_path: Path,
    router_config_path: Path,
    max_budget: float = 20.0,
) -> BenchmarkEntry:
    """Exécute le scénario de benchmark complet pour un couple (Planner -> Executor)."""
    tracker = CostTracker(max_budget=max_budget, models_config_path=models_config_path)
    pairing = tracker.get_pairing(pairing_id)
    router = StepRouter(router_config_path=router_config_path)

    planner = SimulatedPlanner(cost_tracker=tracker, model_key=pairing.planner)
    executor = SimulatedExecutor(cost_tracker=tracker, model_key=pairing.executor)

    with tempfile.TemporaryDirectory() as tmp_dir:
        context = ExecutionContext(project_root=Path(tmp_dir))
        current_plan = await planner.plan(
            goal="Benchmark Pipeline",
            context=context.snapshot(),
        )

        escalations = 0
        step_index = 0
        while step_index < len(current_plan.steps):
            if tracker.is_over_budget():
                break

            step = current_plan.steps[step_index]
            completed: list[str] = context.get("completed_steps", [])
            if step.id in completed:
                step_index += 1
                continue

            state_for_router = context.snapshot()
            state_for_router["remaining_budget"] = tracker.remaining_budget()
            target = await router.route(step, state_for_router)

            if target == "planner":
                escalations += 1
                current_plan = await planner.plan(
                    goal=current_plan.goal,
                    context=context.snapshot(),
                )
                executor.mark_replanned(step.id)
                context.set("step_retries", {})
                continue

            result = await executor.execute(step, context.snapshot())
            if result.status == StepStatus.SUCCESS:
                completed.append(step.id)
                context.set("completed_steps", completed)
                step_index += 1
            else:
                retries = context.get("step_retries", {})
                retries[step.id] = int(retries.get(step.id, 0)) + 1
                context.set("step_retries", retries)

        completed_final: list[str] = context.get("completed_steps", [])
        total_steps = len(current_plan.steps)
        steps_done = len(completed_final)
        architecture = (
            "100% Frontier" if pairing.planner == pairing.executor else "Delegated P-t-E"
        )

        return BenchmarkEntry(
            pairing_id=pairing.pairing_id,
            planner_model=pairing.planner,
            executor_model=pairing.executor,
            architecture=architecture,
            steps_completed=steps_done,
            total_steps=total_steps,
            success_rate_pct=round((steps_done / total_steps) * 100.0, 1),
            react_turns_total=executor.react_turns,
            escalations_to_planner=escalations,
            planner_cost_usd=round(tracker.planner_cost(), 4),
            executor_cost_usd=round(tracker.executor_cost(), 4),
            total_cost_usd=round(tracker.total_cost(), 4),
            savings_vs_baseline_pct=0.0,
            simulated_latency_s=round(planner.total_latency_s + executor.total_latency_s, 1),
        )


async def run_comparative_benchmark(
    models_config_path: Path,
    router_config_path: Path,
    pairing_ids: list[str] | None = None,
    max_budget: float = 20.0,
) -> list[BenchmarkEntry]:
    """Exécute le benchmark comparatif sur tous les couples demandés et calcule les économies."""
    tracker = CostTracker(max_budget=max_budget, models_config_path=models_config_path)
    all_pairings = {p.pairing_id: p for p in tracker.list_pairings()}
    target_ids = pairing_ids or list(all_pairings.keys())

    raw_entries: dict[str, BenchmarkEntry] = {}
    for pid in target_ids:
        raw_entries[pid] = await run_single_pairing_benchmark(
            pairing_id=pid,
            models_config_path=models_config_path,
            router_config_path=router_config_path,
            max_budget=max_budget,
        )

    # Calcul des économies en % par rapport au baseline (100% Frontier) de chaque couple
    final_entries: list[BenchmarkEntry] = []
    for pid in target_ids:
        entry = raw_entries[pid]
        baseline_id = all_pairings[pid].baseline
        if baseline_id not in raw_entries:
            baseline_entry = await run_single_pairing_benchmark(
                pairing_id=baseline_id,
                models_config_path=models_config_path,
                router_config_path=router_config_path,
                max_budget=max_budget,
            )
            raw_entries[baseline_id] = baseline_entry
        else:
            baseline_entry = raw_entries[baseline_id]

        if baseline_entry.total_cost_usd > 0:
            savings = (
                (baseline_entry.total_cost_usd - entry.total_cost_usd)
                / baseline_entry.total_cost_usd
            ) * 100.0
        else:
            savings = 0.0

        final_entries.append(
            entry.model_copy(update={"savings_vs_baseline_pct": round(savings, 1)})
        )

    return final_entries


def render_benchmark_rich_table(entries: list[BenchmarkEntry], console: Console) -> None:
    """Affiche le tableau comparatif complet dans la console via Rich."""
    table = Table(
        title="Benchmark simulé : 100% Frontier vs Plan-then-Execute (10 étapes)",
        caption="Coûts, économies et latences simulés à partir des profils configurés ; aucun appel API.",
        show_lines=True,
    )
    table.add_column("Pairing ID", style="bold cyan", no_wrap=True)
    table.add_column("Architecture", style="magenta")
    table.add_column("Planner (Parent)", style="blue")
    table.add_column("Executor (Child)", style="green")
    table.add_column("Succès", justify="right")
    table.add_column("Tours ReAct", justify="right")
    table.add_column("Escalades", justify="right", style="yellow")
    table.add_column("Coût Planner", justify="right")
    table.add_column("Coût Executor", justify="right")
    table.add_column("Coût Total", justify="right", style="bold white")
    table.add_column("Économie vs Frontier", justify="right", style="bold green")
    table.add_column("Latence", justify="right")

    for e in entries:
        savings_str = (
            "Baseline (0.0%)"
            if e.savings_vs_baseline_pct == 0.0
            else f"{-e.savings_vs_baseline_pct:+.1f}%"
        )
        table.add_row(
            e.pairing_id,
            e.architecture,
            e.planner_model,
            e.executor_model,
            f"{e.steps_completed}/{e.total_steps} ({e.success_rate_pct:.0f}%)",
            str(e.react_turns_total),
            str(e.escalations_to_planner),
            f"${e.planner_cost_usd:.4f}",
            f"${e.executor_cost_usd:.4f}",
            f"${e.total_cost_usd:.4f}",
            savings_str,
            f"{e.simulated_latency_s:.1f}s",
        )

    console.print(table)


def format_benchmark_markdown_table(entries: list[BenchmarkEntry]) -> str:
    """Génère un tableau Markdown à partir des résultats du benchmark."""
    headers = [
        "Simulated costs, savings, and latency from configured token profiles; no live API calls.\n",
        "| Pairing ID | Architecture | Planner (Parent) | Executor (Child) | Completion | ReAct Turns | Escalations | Planner Cost | Executor Cost | Total Cost | Savings vs Baseline | Latency |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    rows: list[str] = []
    for e in entries:
        savings_str = (
            "Baseline (0.0%)"
            if e.savings_vs_baseline_pct == 0.0
            else f"**{-e.savings_vs_baseline_pct:+.1f}%**"
        )
        rows.append(
            f"| `{e.pairing_id}` | {e.architecture} | `{e.planner_model}` | `{e.executor_model}` | "
            f"{e.steps_completed}/{e.total_steps} ({e.success_rate_pct:.0f}%) | {e.react_turns_total} | "
            f"{e.escalations_to_planner} | \\${e.planner_cost_usd:.4f} | \\${e.executor_cost_usd:.4f} | "
            f"**\\${e.total_cost_usd:.4f}** | {savings_str} | {e.simulated_latency_s:.1f}s |"
        )
    return "\n".join(headers + rows)
