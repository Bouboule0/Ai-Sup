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

    assert by_id["argon-pro"].total_cost_usd < by_id["argon-only"].total_cost_usd
    assert by_id["opus-sonnet"].total_cost_usd < by_id["opus-only"].total_cost_usd
    assert by_id["opus-sonnet"].savings_vs_baseline_pct > 55.0

    # Vérifie que les modèles légers (Flash / Haiku) déclenchent bien l'escalade StepRouter sur l'étape complexe
    assert by_id["argon-flash"].escalations_to_planner == 1
    assert by_id["argon-pro"].escalations_to_planner == 0
    assert by_id["opus-sonnet"].escalations_to_planner == 0

    md_table = format_benchmark_markdown_table(entries)
    assert "| `argon-flash` |" in md_table
    assert "| `opus-sonnet` |" in md_table
