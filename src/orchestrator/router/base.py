from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from orchestrator.schemas import Step


class BaseRouter(ABC):
    """Interface abstraite du routeur décidant quel modèle traite l'étape."""

    @abstractmethod
    async def route(self, step: Step, state: dict[str, Any]) -> Literal["planner", "executor"]:
        """Détermine si l'étape doit être confiée au 'planner' (Argon) ou à l''executor' (Flash).

        Args:
            step: Étape courante à évaluer.
            state: Dictionnaire contenant l'état d'exécution (retries, remaining_budget, etc.).

        Returns:
            'planner' ou 'executor'.
        """
