from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    """Représente un outil exécutable mis à disposition de l'Exécuteur."""

    name: str
    description: str
    args_schema: type[BaseModel]
    func: Callable[..., str]

    def execute(self, raw_args: dict[str, Any]) -> str:
        """Valide les arguments via Pydantic et exécute la fonction associée."""
        validated = self.args_schema.model_validate(raw_args)
        kwargs = {field_name: getattr(validated, field_name) for field_name in type(validated).model_fields}
        return self.func(**kwargs)


class ToolRegistry:
    """Registre central des outils disponibles."""

    def __init__(self) -> None:
        """Initialise un registre vide."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Enregistre un outil dans le registre."""
        self._tools[tool.name] = tool

    def get_tools(self, names: list[str]) -> list[Tool]:
        """Retourne la liste des outils correspondant aux noms demandés.

        Args:
            names: Liste des noms d'outils autorisés pour l'étape.

        Returns:
            Liste d'instances Tool.

        Raises:
            KeyError: Si un outil demandé n'est pas enregistré.
        """
        result: list[Tool] = []
        for name in names:
            if name not in self._tools:
                raise KeyError(f"Outil non enregistré : '{name}'")
            result.append(self._tools[name])
        return result

    def list_all(self) -> list[Tool]:
        """Retourne tous les outils enregistrés."""
        return list(self._tools.values())


GLOBAL_REGISTRY = ToolRegistry()


def register_tool(tool: Tool) -> None:
    """Enregistre un outil dans le registre global."""
    GLOBAL_REGISTRY.register(tool)


def get_tools(names: list[str]) -> list[Tool]:
    """Récupère une sélection d'outils depuis le registre global."""
    return GLOBAL_REGISTRY.get_tools(names)
