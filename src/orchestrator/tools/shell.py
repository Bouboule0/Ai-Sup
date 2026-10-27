from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from pydantic import BaseModel, Field
import yaml

from orchestrator.tools.registry import Tool, ToolRegistry

FORBIDDEN_SHELL_TOKENS = (";", "&&", "||", "|", "`", "$(", ">", "<")


class RunCommandArgs(BaseModel):
    """Arguments pour l'outil run_command."""

    command: str = Field(..., description="Commande shell à exécuter (soumise à liste blanche).")


def load_allowed_commands(router_config_path: Path) -> list[str]:
    """Charge la liste blanche des commandes autorisées depuis config/router.yaml."""
    raw = yaml.safe_load(router_config_path.read_text(encoding="utf-8")) or {}
    shell_cfg = raw.get("shell", {})
    allowed = shell_cfg.get("allowed_commands", [])
    return [str(cmd).strip() for cmd in allowed if str(cmd).strip()]


def is_command_allowed(command: str, allowed_prefixes: list[str]) -> bool:
    """Vérifie qu'une commande correspond à la liste blanche et ne contient pas d'injection."""
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

    for allowed in allowed_prefixes:
        allowed_parts = shlex.split(allowed)
        if cmd_parts[: len(allowed_parts)] == allowed_parts:
            return True
    return False


def register_shell_tools(
    registry: ToolRegistry,
    project_root: Path,
    router_config_path: Path,
) -> None:
    """Enregistre l'outil run_command sécurisé par la liste blanche de router.yaml."""
    allowed_prefixes = load_allowed_commands(router_config_path)

    def _run_command(command: str) -> str:
        if not is_command_allowed(command, allowed_prefixes):
            raise PermissionError(
                f"Commande refusée par la liste blanche : '{command}'. "
                f"Commandes autorisées : {allowed_prefixes}"
            )
        args = shlex.split(command)
        completed = subprocess.run(
            args,
            cwd=str(project_root.resolve()),
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
