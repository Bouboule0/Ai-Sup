from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _validate_structure_and_cycles(self) -> Plan:
        """Valide l'unicité des identifiants d'étapes, l'absence d'auto-dépendance et de cycles."""
        seen_ids: set[str] = set()
        steps_by_id: dict[str, Step] = {}
        for step in self.steps:
            if step.id in seen_ids:
                raise ValueError(f"Identifiant d'étape dupliqué dans le plan : '{step.id}'")
            if step.id in step.depends_on:
                raise ValueError(f"L'étape '{step.id}' ne peut pas dépendre d'elle-même.")
            seen_ids.add(step.id)
            steps_by_id[step.id] = step

        # Détection de cycle (DFS 3 couleurs : 0=non visité, 1=en cours, 2=terminé)
        state: dict[str, int] = {sid: 0 for sid in seen_ids}

        def _visit(node_id: str) -> None:
            state[node_id] = 1
            for dep_id in steps_by_id[node_id].depends_on:
                if dep_id not in state:
                    continue
                if state[dep_id] == 1:
                    raise ValueError(
                        f"Cycle de dépendances détecté impliquant '{node_id}' et '{dep_id}'."
                    )
                if state[dep_id] == 0:
                    _visit(dep_id)
            state[node_id] = 2

        for sid in seen_ids:
            if state[sid] == 0:
                _visit(sid)

        return self

    def validate_dependencies(self, completed_steps: Iterable[str] = ()) -> None:
        """Vérifie que chaque prérequis existe (dans le plan ou l'état complété) et respecte l'ordre d'exécution."""
        completed_set = set(completed_steps)
        index_by_id = {step.id: idx for idx, step in enumerate(self.steps)}

        for idx, step in enumerate(self.steps):
            for dep in step.depends_on:
                if dep in completed_set:
                    continue
                if dep not in index_by_id:
                    raise ValueError(
                        f"Dépendance introuvable '{dep}' requise par l'étape '{step.id}'."
                    )
                if index_by_id[dep] >= idx:
                    raise ValueError(
                        f"Ordre de dépendance invalide : '{step.id}' dépend de '{dep}' "
                        "qui est planifiée après elle."
                    )


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


class DecisionQuestionType(str, Enum):
    """Types de questions supportées par un modèle de décision."""

    PREDICATE = "predicate"
    CHOICE = "choice"
    SCORE = "score"


class DecisionOption(BaseModel):
    """Option d'une question de type `choice`."""

    value: str = Field(..., description="Valeur renvoyée par le modèle pour cette option.")
    description: str | None = Field(default=None, description="Description de l'option.")


class DecisionLevel(BaseModel):
    """Niveau d'une question de type `score`."""

    label: str = Field(..., description="Libellé du niveau.")
    description: str | None = Field(default=None, description="Description du niveau.")


class DecisionQuestion(BaseModel):
    """Question fermée envoyée à un modèle de décision (Jev)."""

    type: DecisionQuestionType = Field(..., description="Type de réponse attendue.")
    name: str = Field(..., min_length=1, description="Identifiant unique de la question.")
    instructions: str = Field(..., min_length=1, description="Question posée au modèle.")
    criteria: dict[str, str] | None = Field(
        default=None,
        description="Pour `predicate`, sens des réponses 'true' et 'false'.",
    )
    choices: list[DecisionOption] | None = Field(
        default=None,
        description="Pour `choice`, options possibles (2 à 255).",
    )
    levels: list[DecisionLevel] | None = Field(
        default=None,
        description="Pour `score`, niveaux ordonnés (2 à 10).",
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> DecisionQuestion:
        """Vérifie que les options ou niveaux requis sont présents et cohérents."""
        if self.type == DecisionQuestionType.CHOICE:
            if self.choices is None or not 2 <= len(self.choices) <= 255:
                raise ValueError("Une question 'choice' requiert entre 2 et 255 options.")
            values = [option.value for option in self.choices]
            if len(values) != len(set(values)):
                raise ValueError("Les valeurs des options 'choice' doivent être uniques.")
        if self.type == DecisionQuestionType.SCORE:
            if self.levels is None or not 2 <= len(self.levels) <= 10:
                raise ValueError("Une question 'score' requiert entre 2 et 10 niveaux.")
            labels = [level.label for level in self.levels]
            if len(labels) != len(set(labels)):
                raise ValueError("Les libellés des niveaux 'score' doivent être uniques.")
        return self


class DecisionProbability(BaseModel):
    """Probabilité associée à une option ou à un niveau."""

    value: str | int = Field(..., description="Valeur de l'option ou index du niveau.")
    label: str | None = Field(default=None, description="Libellé du niveau, si applicable.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité de cette valeur.")


class DecisionAnswer(BaseModel):
    """Réponse structurée d'un modèle de décision à une question."""

    type: DecisionQuestionType = Field(..., description="Type de la question répondue.")
    name: str = Field(..., description="Nom de la question répondue.")
    probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probabilité du 'oui' pour une question `predicate`.",
    )
    choice: str | None = Field(default=None, description="Option retenue pour `choice`.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confiance du modèle pour `choice` et `score`.",
    )
    score: float | None = Field(default=None, ge=0.0, description="Note pour `score`.")
    legend: dict[str, str] | None = Field(
        default=None,
        description="Correspondance index de niveau -> description.",
    )
    probabilities: list[DecisionProbability] | None = Field(
        default=None,
        description="Probabilité de chaque option ou niveau.",
    )

    @model_validator(mode="after")
    def _validate_expected_field(self) -> DecisionAnswer:
        """Vérifie que le champ de réponse attendu pour le type est présent."""
        if self.type == DecisionQuestionType.PREDICATE and self.probability is None:
            raise ValueError("Une réponse 'predicate' doit contenir 'probability'.")
        if self.type == DecisionQuestionType.CHOICE and self.choice is None:
            raise ValueError("Une réponse 'choice' doit contenir 'choice'.")
        if self.type == DecisionQuestionType.SCORE and self.score is None:
            raise ValueError("Une réponse 'score' doit contenir 'score'.")
        return self


DecisionInput = str | dict[str, Any] | list[dict[str, Any]]


class DecisionRequest(BaseModel):
    """Corps de requête envoyé à l'API Decisions d'OfoxAI."""

    model: str = Field(..., min_length=1, description="Identifiant du modèle de décision.")
    input: DecisionInput = Field(..., description="Contenu texte ou JSON à évaluer.")
    questions: list[DecisionQuestion] = Field(
        ...,
        min_length=1,
        description="Questions à poser dans la même requête.",
    )

    @model_validator(mode="after")
    def _validate_unique_names(self) -> DecisionRequest:
        """Vérifie que chaque question possède un nom unique."""
        names = [question.name for question in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("Les noms des questions doivent être uniques.")
        return self


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
