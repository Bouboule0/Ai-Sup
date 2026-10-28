from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from orchestrator.schemas import ExecutionResult, Step


class BaseExecutor(ABC):
    """Interface abstraite pour l'exécution d'une étape du plan."""

    @abstractmethod
    async def execute(self, step: Step, state: dict[str, Any]) -> ExecutionResult:
        """Exécute une étape unique et retourne son résultat structuré.

        Args:
            step: Étape atomique issue du Plan.
            state: État courant partagé du projet.

        Returns:
            Instance de ExecutionResult.
        """
