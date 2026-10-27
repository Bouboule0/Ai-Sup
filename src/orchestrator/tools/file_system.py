from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from orchestrator.tools.registry import Tool, ToolRegistry


class ReadFileArgs(BaseModel):
    """Arguments pour l'outil read_file."""

    path: str = Field(..., description="Chemin relatif du fichier à lire dans le projet.")


class WriteFileArgs(BaseModel):
    """Arguments pour l'outil write_file."""

    path: str = Field(..., description="Chemin relatif du fichier à écrire dans le projet.")
    content: str = Field(..., description="Contenu texte complet à écrire.")


class ListDirArgs(BaseModel):
    """Arguments pour l'outil list_dir."""

    path: str = Field(default=".", description="Chemin relatif du dossier à lister.")


def validate_project_path(project_root: Path, relative_path: str) -> Path:
    """Valide qu'un chemin reste strictement confiné dans la racine du projet.

    Args:
        project_root: Racine absolue du projet.
        relative_path: Chemin fourni par l'appelant.

    Returns:
        Chemin absolu résolu et validé.

    Raises:
        PermissionError: Si le chemin contient '..' ou sort de project_root.
    """
    raw_path = Path(relative_path)
    if ".." in raw_path.parts:
        raise PermissionError(f"Chemin interdit contenant '..' : {relative_path}")
    if raw_path.is_absolute():
        raise PermissionError(f"Chemin absolu interdit : {relative_path}")

    root_resolved = project_root.resolve()
    target_resolved = (root_resolved / raw_path).resolve()

    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise PermissionError(f"Accès hors du répertoire projet refusé : {relative_path}")

    return target_resolved


def verify_expected_outputs(project_root: Path, expected_outputs: list[str]) -> list[str]:
    """Retourne la liste des fichiers attendus (expected_outputs) qui n'existent pas sur le disque."""
    missing: list[str] = []
    for rel_path in expected_outputs:
        try:
            target = validate_project_path(project_root, rel_path)
            if not target.exists():
                missing.append(rel_path)
        except PermissionError:
            missing.append(rel_path)
    return missing


def register_filesystem_tools(registry: ToolRegistry, project_root: Path) -> None:
    """Enregistre les outils read_file, write_file et list_dir liés à project_root."""
    registry.project_root = project_root.resolve()

    def _read_file(path: str) -> str:
        target = validate_project_path(project_root, path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        return target.read_text(encoding="utf-8")

    def _write_file(path: str, content: str) -> str:
        target = validate_project_path(project_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Fichier écrit avec succès : {path} ({len(content)} caractères)"

    def _list_dir(path: str = ".") -> str:
        target = validate_project_path(project_root, path)
        if not target.exists() or not target.is_dir():
            raise NotADirectoryError(f"Répertoire introuvable : {path}")
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        return "\n".join(entries) if entries else "(dossier vide)"

    registry.register(
        Tool(
            name="read_file",
            description="Lit le contenu texte d'un fichier du projet.",
            args_schema=ReadFileArgs,
            func=_read_file,
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description="Crée ou écrase un fichier texte dans le projet.",
            args_schema=WriteFileArgs,
            func=_write_file,
        )
    )
    registry.register(
        Tool(
            name="list_dir",
            description="Liste les fichiers et sous-dossiers d'un répertoire du projet.",
            args_schema=ListDirArgs,
            func=_list_dir,
        )
    )
