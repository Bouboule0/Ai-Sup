from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from orchestrator.schemas import Plan


class BasePlanner(ABC):
    """Interface abstraite pour tout planificateur produisant un Plan validé."""

    @abstractmethod
    async def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Génère un plan structuré à partir d'un objectif et d'un contexte d'exécution.

        Args:
            goal: Objectif global demandé par l'utilisateur.
            context: État courant du projet (fichiers existants, erreurs précédentes, etc.).

        Returns:
            Instance validée de Plan.
        """
