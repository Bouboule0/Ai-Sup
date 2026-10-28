from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ExecutionContext:
    """Gère l'état partagé entre les étapes avec persistance JSON dans le projet cible."""

    def __init__(self, project_root: Path, state_filename: str = "state.json") -> None:
        """Initialise ou recharge l'état depuis le fichier JSON du projet.

        Args:
            project_root: Répertoire racine du projet cible.
            state_filename: Nom du fichier d'état (par défaut 'state.json').
        """
        self._project_root = project_root.resolve()
        self._project_root.mkdir(parents=True, exist_ok=True)
        self._state_path = self._project_root / state_filename
        self._data: dict[str, Any] = {
            "completed_steps": [],
            "step_retries": {},
            "step_outputs": {},
            "artifacts": {},
        }
        if self._state_path.exists():
            try:
                loaded = json.loads(self._state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except json.JSONDecodeError:
                pass

    @property
    def project_root(self) -> Path:
        """Retourne le chemin racine résolu du projet cible."""
        return self._project_root

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de l'état partagé."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Définit une valeur dans l'état partagé et sauvegarde immédiatement sur disque."""
        self._data[key] = value
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        """Retourne une copie profonde de l'état courant."""
        return copy.deepcopy(self._data)

    def _persist(self) -> None:
        """Écrit l'état courant dans state.json."""
        self._state_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
