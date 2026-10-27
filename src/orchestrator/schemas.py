from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types d'étapes supportés par le planificateur et le routeur."""

    CODE = "code"
    FILE_OP = "file_op"
    SHELL = "shell"
    VERIFY = "verify"
    REPLAN = "replan"


class StepStatus(str, Enum):
    """Statuts d'exécution d'une étape."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    """Représente une étape atomique du plan d'exécution."""

    id: str = Field(..., description="Identifiant unique de l'étape (ex: step_1).")
    description: str = Field(..., description="Instruction précise de ce qui doit être fait.")
    type: StepType = Field(..., description="Catégorie de l'étape.")
    tools_allowed: list[str] = Field(
        default_factory=list,
        description="Noms des outils autorisés pour cette étape.",
    )
    success_criteria: str = Field(
        ...,
        description="Critère vérifiable déterminant si l'étape est réussie.",
    )
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="Fichiers ou artefacts attendus en sortie.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Liste des identifiants d'étapes prérequises.",
    )


class Plan(BaseModel):
    """Contrat de planification produit par le Planner et consommé par l'Executor."""

    goal: str = Field(..., description="Objectif initial de l'utilisateur.")
    summary: str = Field(..., description="Résumé de la stratégie adoptée.")
    steps: list[Step] = Field(..., min_length=1, description="Liste ordonnée des étapes.")


class ToolCallRequest(BaseModel):
    """Demande d'appel d'outil émise par l'exécuteur."""

    tool_name: str = Field(..., description="Nom de l'outil à exécuter.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passés à l'outil.",
    )


class ExecutionResult(BaseModel):
    """Résultat structuré retourné après l'exécution d'une étape."""

    step_id: str = Field(..., description="Identifiant de l'étape exécutée.")
    status: StepStatus = Field(..., description="Statut final de l'étape.")
    output: str = Field(..., description="Synthèse ou contenu produit par l'étape.")
    justification: str = Field(
        ...,
        description="Explication de la réussite ou de la cause d'échec.",
    )
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="Appels d'outils demandés ou effectués durant le tour.",
    )
    error: str | None = Field(
        default=None,
        description="Message d'erreur détaillé en cas d'échec.",
    )


class BenchmarkEntry(BaseModel):
    """Résultat agrégé d'un couple (Planner -> Executor) sur un scénario de benchmark."""

    pairing_id: str = Field(..., description="Identifiant du couple (ex: argon-flash).")
    planner_model: str = Field(..., description="Nom du modèle Planificateur.")
    executor_model: str = Field(..., description="Nom du modèle Exécuteur.")
    architecture: str = Field(
        ...,
        description="'100% Frontier' si planner == executor, sinon 'Delegated P-t-E'.",
    )
    steps_completed: int = Field(..., description="Nombre d'étapes réussies.")
    total_steps: int = Field(..., description="Nombre total d'étapes du plan.")
    success_rate_pct: float = Field(..., description="Taux de complétion des étapes en %.")
    react_turns_total: int = Field(..., description="Nombre total d'appels à l'Exécuteur.")
    escalations_to_planner: int = Field(
        ...,
        description="Nombre de replanifications déclenchées par le StepRouter.",
    )
    planner_cost_usd: float = Field(..., description="Coût cumulé du Planner en USD.")
    executor_cost_usd: float = Field(..., description="Coût cumulé de l'Executor en USD.")
    total_cost_usd: float = Field(..., description="Coût total (Planner + Executor) en USD.")
    savings_vs_baseline_pct: float = Field(
        default=0.0,
        description="Économie en % par rapport au baseline 100% Frontier correspondant.",
    )
    simulated_latency_s: float = Field(
        ...,
        description="Durée totale estimée ou mesurée en secondes.",
    )
