from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from orchestrator.schemas import DecisionAnswer, DecisionInput, DecisionQuestion


class BaseDecisionModel(ABC):
    """Interface abstraite d'un modèle de décision typé (ex: Jev / TypeSafe)."""

    @abstractmethod
    async def decide(
        self,
        content: DecisionInput,
        questions: Sequence[DecisionQuestion],
    ) -> list[DecisionAnswer]:
        """Évalue un contenu et retourne une réponse structurée par question.

        Args:
            content: Contenu texte ou JSON à évaluer.
            questions: Questions fermées à poser dans la même requête.

        Returns:
            Liste de réponses dans le même ordre que les questions.
        """
