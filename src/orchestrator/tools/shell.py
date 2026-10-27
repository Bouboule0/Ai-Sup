from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess
from pydantic import BaseModel, Field
import yaml

from orchestrator.tools.file_system import validate_project_path
from orchestrator.tools.registry import Tool, ToolRegistry

FORBIDDEN_SHELL_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "\r")
FORBIDDEN_INLINE_EXEC_FLAGS = ("-c", "--command", "-e", "--eval")
_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:[\\/]")


class RunCommandArgs(BaseModel):
    """Arguments pour l'outil run_command."""

    command: str = Field(..., description="Commande shell à exécuter (soumise à liste blanche).")


def load_allowed_commands(router_config_path: Path) -> list[str]:
    """Charge la liste blanche des commandes autorisées depuis config/router.yaml."""
    raw = yaml.safe_load(router_config_path.read_text(encoding="utf-8")) or {}
    shell_cfg = raw.get("shell", {})
    allowed = shell_cfg.get("allowed_commands", [])
    return [str(cmd).strip() for cmd in allowed if str(cmd).strip()]


def _extract_candidate_path(token: str) -> str | None:
    """Extrait un chemin potentiel depuis un argument CLI (ex: '--cov=../outside' ou '../a.py')."""
    if token.startswith("-"):
        if "=" in token:
            _, value = token.split("=", 1)
            return value.strip() or None
        return None
    return token


def _is_argument_safe(token: str, project_root: Path | None) -> bool:
    """Vérifie qu'un argument CLI n'exécute pas de code inline et ne sort pas de project_root."""
    if token in FORBIDDEN_INLINE_EXEC_FLAGS or token.startswith("-c"):
        return False
    for flag in FORBIDDEN_INLINE_EXEC_FLAGS:
        if token.startswith(f"{flag}="):
            return False
    if token.startswith("-") and not token.startswith("--"):
        if any(flag in token[1:] for flag in ("c", "e")):
            return False

    candidate = _extract_candidate_path(token)
    if candidate is None:
        return True

    normalized = candidate.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if ".." in parts or candidate.startswith("~"):
        return False

    is_abs = candidate.startswith(("/", "\\")) or bool(_WINDOWS_DRIVE_RE.match(candidate))
    if is_abs:
        if project_root is None:
            return False
        try:
            validate_project_path(project_root, candidate)
        except PermissionError:
            return False

    if project_root is not None and ("/" in candidate or "\\" in candidate or "." in candidate):
        try:
            validate_project_path(project_root, candidate)
        except PermissionError:
            return False

    return True


def is_command_allowed(
    command: str,
    allowed_prefixes: list[str],
    project_root: Path | None = None,
) -> bool:
    """Vérifie qu'une commande correspond à la liste blanche, sans injection ni traversée de chemin."""
    cleaned = command.strip()
    if not cleaned:
        return False

    for token in FORBIDDEN_SHELL_TOKENS:
        if token in cleaned:
            return False

    try:
        cmd_parts = shlex.split(cleaned)
    except ValueError:
        return False

    if not cmd_parts:
        return False

    matched_prefix_len: int | None = None
    for allowed in allowed_prefixes:
        allowed_parts = shlex.split(allowed)
        if cmd_parts[: len(allowed_parts)] == allowed_parts:
            matched_prefix_len = len(allowed_parts)
            break

    if matched_prefix_len is None:
        return False

    for arg in cmd_parts[matched_prefix_len:]:
        if not _is_argument_safe(arg, project_root):
            return False

    return True


def register_shell_tools(
    registry: ToolRegistry,
    project_root: Path,
    router_config_path: Path,
) -> None:
    """Enregistre l'outil run_command sécurisé par la liste blanche de router.yaml."""
    allowed_prefixes = load_allowed_commands(router_config_path)
    resolved_root = project_root.resolve()

    def _run_command(command: str) -> str:
        if not is_command_allowed(command, allowed_prefixes, project_root=resolved_root):
            raise PermissionError(
                f"Commande refusée par la politique de sécurité shell : '{command}'. "
                f"Commandes autorisées (sans '-c' ni traversée '..') : {allowed_prefixes}"
            )
        args = shlex.split(command)
        completed = subprocess.run(
            args,
            cwd=str(resolved_root),
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        return (
            f"returncode={completed.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    registry.register(
        Tool(
            name="run_command",
            description="Exécute une commande shell présente dans la liste blanche.",
            args_schema=RunCommandArgs,
            func=_run_command,
        )
    )
