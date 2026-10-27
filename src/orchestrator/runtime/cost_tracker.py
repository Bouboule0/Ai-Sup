from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class ModelPricing:
    """Tarification et métadonnées fournisseur d'un modèle par million de tokens."""

    key: str
    model_id: str
    provider: str
    input_cost_per_million: float
    output_cost_per_million: float


@dataclass(frozen=True)
class ModelPairing:
    """Définition d'un couple (Planner -> Executor) pour l'orchestration et le benchmark."""

    pairing_id: str
    description: str
    planner: str
    executor: str
    baseline: str


class CostTracker:
    """Suit la consommation de tokens et le coût cumulé par rapport à un budget."""

    def __init__(self, max_budget: float, models_config_path: Path) -> None:
        """Initialise le suivi des coûts depuis le fichier models.yaml.

        Args:
            max_budget: Budget maximal autorisé en dollars USD.
            models_config_path: Chemin vers config/models.yaml.
        """
        if max_budget < 0:
            raise ValueError("max_budget doit être positif ou nul.")
        self._max_budget = float(max_budget)
        self._total_cost = 0.0
        self._planner_cost = 0.0
        self._executor_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._pricing_by_key: dict[str, ModelPricing] = {}
        self._pricing_by_id: dict[str, ModelPricing] = {}
        self._pairings: dict[str, ModelPairing] = {}
        self._load_config(models_config_path)

    def _load_config(self, config_path: Path) -> None:
        """Charge les tarifs et les couples de modèles depuis le fichier YAML."""
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        models = raw.get("models", {})
        for role_key, info in models.items():
            pricing = ModelPricing(
                key=str(role_key),
                model_id=str(info["id"]),
                provider=str(info.get("provider", "google")),
                input_cost_per_million=float(info["input_cost_per_million"]),
                output_cost_per_million=float(info["output_cost_per_million"]),
            )
            self._pricing_by_key[role_key] = pricing
            self._pricing_by_id[pricing.model_id] = pricing

        pairings = raw.get("pairings", {})
        for pairing_id, p_info in pairings.items():
            self._pairings[str(pairing_id)] = ModelPairing(
                pairing_id=str(pairing_id),
                description=str(p_info.get("description", "")),
                planner=str(p_info["planner"]),
                executor=str(p_info["executor"]),
                baseline=str(p_info.get("baseline", pairing_id)),
            )

    def get_model_pricing(self, role_or_id: str) -> ModelPricing:
        """Retourne l'objet ModelPricing pour une clé ou un identifiant de modèle."""
        pricing = self._pricing_by_key.get(role_or_id) or self._pricing_by_id.get(role_or_id)
        if pricing is None:
            raise KeyError(f"Modèle inconnu dans la configuration : {role_or_id}")
        return pricing

    def get_model_id(self, role_or_id: str) -> str:
        """Résout l'identifiant API d'un modèle à partir de son rôle ('planner'/'executor') ou ID."""
        return self.get_model_pricing(role_or_id).model_id

    def get_provider(self, role_or_id: str) -> str:
        """Retourne le fournisseur ('google', 'anthropic', etc.) d'un modèle."""
        return self.get_model_pricing(role_or_id).provider

    def list_pairings(self) -> list[ModelPairing]:
        """Retourne la liste des couples (Planner -> Executor) configurés."""
        return list(self._pairings.values())

    def get_pairing(self, pairing_id: str) -> ModelPairing:
        """Récupère un couple configuré par son identifiant."""
        if pairing_id not in self._pairings:
            raise KeyError(
                f"Couple '{pairing_id}' inconnu. Couples disponibles : {list(self._pairings.keys())}"
            )
        return self._pairings[pairing_id]

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        role: str | None = None,
    ) -> float:
        """Enregistre un appel LLM et retourne le coût de cet appel en USD.

        Args:
            model: Clé de rôle ('planner', 'executor') ou identifiant du modèle.
            input_tokens: Nombre de tokens en entrée.
            output_tokens: Nombre de tokens en sortie.
            role: Rôle explicite optionnel ('planner' ou 'executor') pour la ventilation des coûts.

        Returns:
            Coût de l'appel en USD.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Le nombre de tokens ne peut pas être négatif.")

        pricing = self.get_model_pricing(model)
        call_cost = (
            (input_tokens / 1_000_000.0) * pricing.input_cost_per_million
            + (output_tokens / 1_000_000.0) * pricing.output_cost_per_million
        )
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += call_cost

        effective_role = role or (model if model in ("planner", "executor") else None)
        if effective_role == "planner":
            self._planner_cost += call_cost
        elif effective_role == "executor":
            self._executor_cost += call_cost

        return call_cost

    def total_cost(self) -> float:
        """Retourne le coût total cumulé en USD."""
        return round(self._total_cost, 8)

    def planner_cost(self) -> float:
        """Retourne le coût cumulé attribué au Planner en USD."""
        return round(self._planner_cost, 8)

    def executor_cost(self) -> float:
        """Retourne le coût cumulé attribué à l'Executor en USD."""
        return round(self._executor_cost, 8)

    def remaining_budget(self) -> float:
        """Retourne le budget restant en USD (peut être négatif en cas de dépassement)."""
        return round(self._max_budget - self._total_cost, 8)

    def is_over_budget(self) -> bool:
        """Indique si le coût cumulé a atteint ou dépassé le budget maximum."""
        return self._total_cost >= self._max_budget
