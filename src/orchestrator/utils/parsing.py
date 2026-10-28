from __future__ import annotations

import json
import re
from typing import Any
from pydantic import ValidationError


def extract_json_block(text: str) -> dict[str, Any]:
    """Extrait et parse le premier ou plus grand objet JSON valide depuis une réponse LLM.

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
    for match in fence_pattern.finditer(cleaned):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    best_obj: dict[str, Any] | None = None
    best_len = -1
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            try:
                obj, end_idx = decoder.raw_decode(cleaned, idx)
                if isinstance(obj, dict) and (end_idx - idx) > best_len:
                    best_obj = obj
                    best_len = end_idx - idx
            except json.JSONDecodeError:
                continue

    if best_obj is not None:
        return best_obj

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
