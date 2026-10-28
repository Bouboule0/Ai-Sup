from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.benchmark import (
    format_benchmark_markdown_table,
    run_comparative_benchmark,
)


def test_comparative_benchmark_all_pairings() -> None:
    """Vérifie l'exécution du benchmark comparatif sur l'ensemble des couples configurés."""
    repo_root = Path(__file__).resolve().parents[2]
    models_yaml = repo_root / "config" / "models.yaml"
    router_yaml = repo_root / "config" / "router.yaml"

    entries = asyncio.run(
        run_comparative_benchmark(
            models_config_path=models_yaml,
            router_config_path=router_yaml,
            max_budget=25.0,
        )
    )

    assert len(entries) >= 6
    by_id = {e.pairing_id: e for e in entries}

    # Vérifie que les deux baselines 100% Frontier ont 0% d'économie par rapport à eux-mêmes
    assert by_id["argon-only"].architecture == "100% Frontier"
    assert by_id["argon-only"].savings_vs_baseline_pct == 0.0
    assert by_id["opus-only"].architecture == "100% Frontier"
    assert by_id["opus-only"].savings_vs_baseline_pct == 0.0

    # Vérifie que la délégation P-t-E réduit significativement le coût par rapport au Frontier
    assert by_id["argon-flash"].total_cost_usd < by_id["argon-only"].total_cost_usd
    assert by_id["argon-flash"].savings_vs_baseline_pct > 45.0

    # 20 turns of 36k input at $2/M plus 7k output at $12/M.
    assert by_id["argon-pro"].executor_cost_usd == 3.12
    assert by_id["opus-sonnet"].total_cost_usd < by_id["opus-only"].total_cost_usd
    assert by_id["opus-sonnet"].savings_vs_baseline_pct > 55.0

    # Vérifie que les modèles légers (Flash / Haiku) déclenchent bien l'escalade StepRouter sur l'étape complexe
    assert by_id["argon-flash"].escalations_to_planner == 1
    assert by_id["argon-pro"].escalations_to_planner == 0
    assert by_id["opus-sonnet"].escalations_to_planner == 0

    md_table = format_benchmark_markdown_table(entries)
    assert "| `argon-flash` |" in md_table
    assert "| `opus-sonnet` |" in md_table


def test_benchmark_renders_signed_cost_changes() -> None:
    from io import StringIO
    from rich.console import Console
    from orchestrator.benchmark import render_benchmark_rich_table
    from orchestrator.schemas import BenchmarkEntry

    entry = BenchmarkEntry(
        pairing_id="test", planner_model="planner", executor_model="executor",
        architecture="Delegated P-t-E", steps_completed=1, total_steps=1, success_rate_pct=100,
        react_turns_total=1, escalations_to_planner=0, planner_cost_usd=1,
        executor_cost_usd=1, total_cost_usd=2, simulated_latency_s=1,
    )
    entries = [entry.model_copy(update={"savings_vs_baseline_pct": pct}) for pct in (25, -10, 0)]
    output = StringIO()
    render_benchmark_rich_table(entries, Console(file=output, width=400))
    for rendered in (output.getvalue(), format_benchmark_markdown_table(entries)):
        assert "-25.0%" in rendered
        assert "+10.0%" in rendered
        assert "Baseline (0.0%)" in rendered
        assert "--10.0%" not in rendered
        assert "simul" in rendered.lower()


def test_simulation_profiles_resolve_config_keys_and_provider_ids(tmp_path: Path) -> None:
    from orchestrator.benchmark import SIMULATION_PROFILES, SimulatedExecutor, SimulatedPlanner
    from orchestrator.runtime.cost_tracker import CostTracker
    import pytest
    import yaml

    cfg = tmp_path / "models.yaml"
    cfg.write_text(yaml.safe_dump({"models": {
        "gemini-3.1-pro": {"id": "gemini-3.1-pro-preview", "input_cost_per_million": 1, "output_cost_per_million": 2},
        "planner": {"id": "gemini-4-argon", "input_cost_per_million": 2, "output_cost_per_million": 10},
        "unsupported": {"id": "unknown-provider-id", "input_cost_per_million": 1, "output_cost_per_million": 2},
    }}))
    tracker = CostTracker(10, cfg)
    for actor in (SimulatedPlanner, SimulatedExecutor):
        for key in ("gemini-3.1-pro", "gemini-3.1-pro-preview"):
            assert actor(tracker, key)._profile == SIMULATION_PROFILES["gemini-3.1-pro"]
        assert actor(tracker, "planner")._profile == SIMULATION_PROFILES["gemini-4-argon"]
        with pytest.raises(ValueError, match="unsupported.*unknown-provider-id"):
            actor(tracker, "unsupported")
        with pytest.raises(KeyError, match="not-configured"):
            actor(tracker, "not-configured")
