from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import ValidationError

from orchestrator.decision.base import BaseDecisionModel
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import (
    DecisionAnswer,
    DecisionInput,
    DecisionQuestion,
    DecisionRequest,
)

DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_S = 0.5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class JevDecisionError(RuntimeError):
    """Erreur d'appel ou de réponse du modèle de décision Jev."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Initialise une erreur Jev avec les métadonnées utiles au diagnostic."""
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.request_id = request_id


class JevDecisionModel(BaseDecisionModel):
    """Client du modèle de décision Jev (TypeSafe) exposé par OfoxAI."""

    def __init__(
        self,
        cost_tracker: CostTracker,
        model_key: str = "jev",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        max_input_chars: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialise le client Jev et associe le CostTracker.

        Args:
            cost_tracker: Suivi budgétaire partagé.
            model_key: Clé du modèle de décision dans models.yaml (par défaut 'jev').
            api_key: Clé OfoxAI ; par défaut lue dans OFOX_API_KEY.
            base_url: Base URL compatible OpenAI (endpoint `/decisions`).
            timeout_s: Délai maximal d'une requête HTTP.
            max_attempts: Nombre total de tentatives sur erreur réseau, 429 ou 5xx.
            retry_backoff_s: Attente de base entre deux tentatives.
            max_input_chars: Troncature des entrées texte au-delà de cette taille.
            transport: Transport httpx optionnel (injection pour les tests).
        """
        resolved_key = api_key or os.environ.get("OFOX_API_KEY")
        if not resolved_key:
            raise EnvironmentError("La variable d'environnement OFOX_API_KEY est absente.")
        if max_attempts < 1:
            raise ValueError("max_attempts doit être supérieur ou égal à 1.")
        if max_input_chars is not None and max_input_chars < 1:
            raise ValueError("max_input_chars doit être positif ou nul.")
        self._cost_tracker = cost_tracker
        self._model_key = model_key
        self._model_id = cost_tracker.get_model_id(model_key)
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._max_attempts = int(max_attempts)
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._max_input_chars = max_input_chars
        self._transport = transport

    @property
    def model_id(self) -> str:
        """Retourne l'identifiant API du modèle de décision configuré."""
        return self._model_id

    async def decide(
        self,
        content: DecisionInput,
        questions: Sequence[DecisionQuestion],
    ) -> list[DecisionAnswer]:
        """Envoie les questions à Jev et retourne les réponses validées.

        Args:
            content: Contenu texte ou JSON à évaluer.
            questions: Questions fermées à poser dans la même requête.

        Returns:
            Liste de réponses dans le même ordre que les questions.

        Raises:
            ValueError: Si les questions sont vides ou leurs noms ne sont pas uniques.
            JevDecisionError: Si l'appel HTTP ou la réponse du modèle est invalide.
        """
        question_list = list(questions)
        if not question_list:
            raise ValueError("Au moins une question est requise.")
        names = [question.name for question in question_list]
        if len(names) != len(set(names)):
            raise ValueError("Les noms des questions doivent être uniques.")

        request = DecisionRequest(
            model=self._model_id,
            input=self._truncate_content(content),
            questions=question_list,
        )
        response = await self._post_with_retries(
            request.model_dump(mode="json", exclude_none=True),
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        payload = self._parse_json(response)
        if not isinstance(payload, dict):
            raise JevDecisionError("La réponse Jev doit être un objet JSON.")
        self._record_usage(payload.get("usage"))

        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, list) or len(raw_answers) != len(question_list):
            raise JevDecisionError(
                "La réponse Jev ne contient pas exactement une réponse par question."
            )

        answers: list[DecisionAnswer] = []
        for raw_answer, question in zip(raw_answers, question_list):
            if not isinstance(raw_answer, dict):
                raise JevDecisionError("Une réponse Jev n'est pas un objet JSON.")
            if raw_answer.get("name") != question.name:
                raise JevDecisionError(
                    f"La réponse Jev ne correspond pas à la question '{question.name}'."
                )
            try:
                answers.append(DecisionAnswer.model_validate(raw_answer))
            except ValidationError as exc:
                raise JevDecisionError(
                    f"Réponse Jev invalide pour la question '{question.name}'."
                ) from exc
        return answers

    async def _post_with_retries(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        """Envoie la requête avec un nombre borné de tentatives sur erreurs transitoires."""
        url = f"{self._base_url}/decisions"
        async with httpx.AsyncClient(
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    if attempt < self._max_attempts:
                        await asyncio.sleep(self._retry_backoff_s * attempt)
                        continue
                    raise JevDecisionError(f"Appel Jev impossible : {exc}") from exc

                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._max_attempts:
                    await asyncio.sleep(self._retry_backoff_s * attempt)
                    continue
                if response.status_code >= 400:
                    raise self._build_error(response)
                return response
        raise JevDecisionError("Appel Jev impossible : aucune réponse reçue.")

    @staticmethod
    def _parse_json(response: httpx.Response) -> Any:
        """Décode le corps JSON de la réponse ou lève une JevDecisionError."""
        try:
            return response.json()
        except ValueError as exc:
            raise JevDecisionError("La réponse Jev n'est pas un JSON valide.") from exc

    @staticmethod
    def _build_error(response: httpx.Response) -> JevDecisionError:
        """Construit une erreur détaillée à partir d'une réponse HTTP en échec."""
        message = response.text.strip()[:500] or f"HTTP {response.status_code}"
        error_type: str | None = None
        request_id: str | None = response.headers.get("x-request-id")
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                error_type = error.get("type") or error_type
                request_id = error.get("request_id") or request_id
            request_id = payload.get("request_id") or request_id
        return JevDecisionError(
            f"Jev a renvoyé HTTP {response.status_code} : {message}",
            status_code=response.status_code,
            error_type=error_type,
            request_id=request_id,
        )

    def _record_usage(self, usage: Any) -> None:
        """Enregistre les tokens d'entrée facturés (la sortie Jev est gratuite)."""
        if not isinstance(usage, dict):
            return
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        if input_tokens > 0:
            self._cost_tracker.record_call(
                self._model_key,
                input_tokens,
                output_tokens=0,
                role="decision",
            )

    def _truncate_content(self, content: DecisionInput) -> DecisionInput:
        """Tronque une entrée texte trop longue pour borner le coût et le contexte."""
        if self._max_input_chars is None or not isinstance(content, str):
            return content
        if len(content) <= self._max_input_chars:
            return content
        suffix = "\n...[tronqué]"
        if self._max_input_chars <= len(suffix):
            return suffix[: self._max_input_chars]
        return content[: self._max_input_chars - len(suffix)] + suffix
