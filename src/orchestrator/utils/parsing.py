from __future__ import annotations

import json
import re
from typing import Any
from pydantic import ValidationError


def extract_json_block(text: str) -> dict[str, Any]:
    """Extrait et parse un objet JSON depuis une réponse texte ou bloc Markdown.

    Args:
        text: Texte brut potentiellement entouré de balises markdown ```json ... ```.

    Returns:
        Dictionnaire Python issu du JSON.

    Raises:
        ValueError: Si aucun JSON valide ne peut être extrait.
    """
    cleaned = text.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = fence_pattern.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"Bloc JSON détecté mais invalide: {exc}") from exc

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"Impossible de parser le JSON extrait: {exc}") from exc

    raise ValueError("Aucun objet JSON valide trouvé dans la réponse.")


def format_validation_error(error: ValidationError) -> str:
    """Formate proprement une erreur de validation Pydantic pour les journaux.

    Args:
        error: Instance de ValidationError levée par Pydantic v2.

    Returns:
        Chaîne lisible listant chaque champ en erreur et la raison.
    """
    lines: list[str] = ["Erreur de validation Pydantic :"]
    for err in error.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "Erreur inconnue")
        err_type = err.get("type", "unknown")
        lines.append(f"  - [{loc}] ({err_type}): {msg}")
    return "\n".join(lines)
