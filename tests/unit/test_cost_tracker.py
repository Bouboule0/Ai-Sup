from __future__ import annotations

from pathlib import Path
import pytest

from orchestrator.runtime.cost_tracker import CostTracker


@pytest.fixture()
def models_config_file(tmp_path: Path) -> Path:
    """Crée un fichier models.yaml temporaire aux tarifs d'Argon et Flash."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
models:
  planner:
    id: "gemini-4-argon"
    input_cost_per_million: 2.00
    output_cost_per_million: 10.00
    capabilities:
      structured_output: true
      function_calling: true
  executor:
    id: "gemini-3.8-flash"
    input_cost_per_million: 0.75
    output_cost_per_million: 3.75
    capabilities:
      structured_output: true
      function_calling: true
""",
        encoding="utf-8",
    )
    return cfg


def test_cost_calculation_and_budget(models_config_file: Path) -> None:
    """Vérifie le calcul exact des coûts input/output et la détection de dépassement."""
    tracker = CostTracker(max_budget=1.00, models_config_path=models_config_file)

    # 100k input Argon ($0.20) + 50k output Argon ($0.50) = $0.70
    cost_argon = tracker.record_call("planner", input_tokens=100_000, output_tokens=50_000)
    assert pytest.approx(cost_argon, rel=1e-6) == 0.70
    assert pytest.approx(tracker.total_cost(), rel=1e-6) == 0.70
    assert pytest.approx(tracker.remaining_budget(), rel=1e-6) == 0.30
    assert not tracker.is_over_budget()

    # 200k input Flash ($0.15) + 100k output Flash ($0.375) = $0.525 -> Total $1.225
    cost_flash = tracker.record_call("gemini-3.8-flash", input_tokens=200_000, output_tokens=100_000)
    assert pytest.approx(cost_flash, rel=1e-6) == 0.525
    assert tracker.is_over_budget()
    assert tracker.remaining_budget() < 0


def test_negative_tokens_rejected(models_config_file: Path) -> None:
    """Vérifie qu'un nombre négatif de tokens lève une ValueError."""
    tracker = CostTracker(max_budget=2.0, models_config_path=models_config_file)
    with pytest.raises(ValueError):
        tracker.record_call("executor", input_tokens=-10, output_tokens=100)


@pytest.mark.parametrize("config_path", ["config/models.yaml", "src/orchestrator/config/models.yaml"])
@pytest.mark.parametrize("pricing_day, expected_prices", [
    ("2026-11-30", (2.0, 10.0)),
    ("2026-12-01", (4.0, 20.0)),
])
def test_configured_prices_at_argon_transition(
    config_path: str, pricing_day: str, expected_prices: tuple[float, float],
) -> None:
    from datetime import date

    root = Path(__file__).resolve().parents[2]
    tracker = CostTracker(10, root / config_path, pricing_date=date.fromisoformat(pricing_day))
    for key in ("planner", "gemini-4-argon"):
        price = tracker.get_model_pricing(key)
        assert (price.input_cost_per_million, price.output_cost_per_million) == expected_prices
        assert tracker.record_call(key, 1_000, 1_000) == pytest.approx(sum(expected_prices) / 1_000)
    opus = tracker.get_model_pricing("claude-opus-5.5")
    assert (opus.input_cost_per_million, opus.output_cost_per_million) == (4.0, 20.0)
    assert tracker.get_model_id("gemini-3.1-pro") == "gemini-3.1-pro-preview"


def test_packaged_model_config_matches_source() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    assert yaml.safe_load((root / "config/models.yaml").read_text()) == yaml.safe_load(
        (root / "src/orchestrator/config/models.yaml").read_text()
    )
