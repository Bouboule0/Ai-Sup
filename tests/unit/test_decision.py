from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from orchestrator.decision import JevDecisionError, JevDecisionModel, create_decision_model
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import (
    DecisionLevel,
    DecisionOption,
    DecisionQuestion,
    DecisionQuestionType,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = REPO_ROOT / "config" / "models.yaml"


@pytest.fixture()
def tracker() -> CostTracker:
    """Retourne un CostTracker configuré avec les tarifs réels du dépôt."""
    return CostTracker(max_budget=1.0, models_config_path=MODELS_CONFIG)


def _predicate_question() -> DecisionQuestion:
    """Construit une question predicate de test."""
    return DecisionQuestion(
        type=DecisionQuestionType.PREDICATE,
        name="escalate_to_planner",
        instructions="Faut-il replanifier cette étape ?",
    )


def _predicate_response(probability: float = 0.82) -> dict[str, object]:
    """Construit une réponse Jev predicate de test."""
    return {
        "model": "typesafe/jev-latest",
        "answers": [
            {
                "type": "predicate",
                "name": "escalate_to_planner",
                "probability": probability,
            }
        ],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 50,
            "total_tokens": 1_000_050,
        },
    }


def _mock_model(
    tracker: CostTracker,
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: object,
) -> JevDecisionModel:
    """Construit un client Jev branché sur un transport httpx de test."""
    return JevDecisionModel(
        cost_tracker=tracker,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        retry_backoff_s=0.0,
        **kwargs,
    )


def test_predicate_request_and_input_only_cost(tracker: CostTracker) -> None:
    """Vérifie le payload, l'authentification, le parsing et la facturation entrée seule."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_predicate_response())

    model = _mock_model(tracker, handler)
    answers = asyncio.run(model.decide("Un échec structurel.", [_predicate_question()]))

    assert captured["url"] == "https://api.ofox.ai/v1/decisions"
    assert captured["method"] == "POST"
    assert captured["authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "typesafe/jev-latest"
    assert payload["input"] == "Un échec structurel."
    assert payload["questions"] == [
        {
            "type": "predicate",
            "name": "escalate_to_planner",
            "instructions": "Faut-il replanifier cette étape ?",
        }
    ]
    assert answers[0].probability == pytest.approx(0.82)
    assert tracker.decision_cost() == pytest.approx(0.042)
    assert tracker.total_cost() == pytest.approx(0.042)
    assert tracker.planner_cost() == 0.0
    assert tracker.executor_cost() == 0.0


def test_choice_and_score_answers_are_parsed(tracker: CostTracker) -> None:
    """Vérifie le parsing des réponses choice et score, y compris probabilités et légende."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-latest",
                "answers": [
                    {
                        "type": "choice",
                        "name": "action",
                        "choice": "replan",
                        "confidence": 0.91,
                        "probabilities": [
                            {"value": "retry", "probability": 0.09},
                            {"value": "replan", "probability": 0.91},
                        ],
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "score": 1.4,
                        "confidence": 0.88,
                        "legend": {"0": "faible", "1": "élevée"},
                        "probabilities": [
                            {"value": 0, "label": "faible", "probability": 0.6},
                            {"value": 1, "label": "élevée", "probability": 0.4},
                        ],
                    },
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
            },
        )

    questions = [
        DecisionQuestion(
            type=DecisionQuestionType.CHOICE,
            name="action",
            instructions="Quelle action choisir ?",
            choices=[
                DecisionOption(value="retry", description="Réessayer"),
                DecisionOption(value="replan", description="Replanifier"),
            ],
        ),
        DecisionQuestion(
            type=DecisionQuestionType.SCORE,
            name="severity",
            instructions="Quelle est la gravité ?",
            levels=[
                DecisionLevel(label="faible", description="Peu grave"),
                DecisionLevel(label="élevée", description="Très grave"),
            ],
        ),
    ]

    model = _mock_model(tracker, handler)
    answers = asyncio.run(model.decide({"error": "boom"}, questions))

    assert answers[0].choice == "replan"
    assert answers[0].confidence == pytest.approx(0.91)
    assert answers[0].probabilities is not None
    assert answers[0].probabilities[0].value == "retry"
    assert answers[1].score == pytest.approx(1.4)
    assert answers[1].legend == {"0": "faible", "1": "élevée"}
    assert tracker.decision_cost() == pytest.approx(0.000042)


def test_http_error_includes_status_type_and_request_id(tracker: CostTracker) -> None:
    """Vérifie qu'une erreur 400 est remontée avec ses métadonnées et sans retry."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "questions are required for decision",
                    "type": "invalid_request_error",
                    "request_id": "req_400",
                }
            },
        )

    model = _mock_model(tracker, handler, max_attempts=3)
    with pytest.raises(JevDecisionError) as excinfo:
        asyncio.run(model.decide("contenu", [_predicate_question()]))

    assert excinfo.value.status_code == 400
    assert excinfo.value.error_type == "invalid_request_error"
    assert excinfo.value.request_id == "req_400"
    assert "questions are required for decision" in str(excinfo.value)
    assert len(calls) == 1
    assert tracker.decision_cost() == 0.0


def test_transient_http_errors_are_retried(tracker: CostTracker) -> None:
    """Vérifie qu'une erreur 503 transitoire est retentée avant de réussir."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "temporaire"}})
        return httpx.Response(200, json=_predicate_response(0.7))

    model = _mock_model(tracker, handler, max_attempts=2)
    answers = asyncio.run(model.decide("contenu", [_predicate_question()]))

    assert len(calls) == 2
    assert answers[0].probability == pytest.approx(0.7)


def test_transport_error_is_wrapped_and_bounded(tracker: CostTracker) -> None:
    """Vérifie qu'une erreur réseau est transformée en JevDecisionError après la dernière tentative."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectTimeout("timeout simulé")

    model = _mock_model(tracker, handler, max_attempts=1)
    with pytest.raises(JevDecisionError, match="Appel Jev impossible"):
        asyncio.run(model.decide("contenu", [_predicate_question()]))
    assert len(calls) == 1


def test_missing_api_key_is_rejected(
    tracker: CostTracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vérifie que la clé OfoxAI est obligatoire pour instancier le client."""
    monkeypatch.delenv("OFOX_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OFOX_API_KEY"):
        JevDecisionModel(cost_tracker=tracker)


def test_duplicate_question_names_are_rejected_before_http(tracker: CostTracker) -> None:
    """Vérifie que deux questions de même nom sont refusées avant tout appel réseau."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_predicate_response())

    question = _predicate_question()
    model = _mock_model(tracker, handler)
    with pytest.raises(ValueError, match="uniques"):
        asyncio.run(model.decide("contenu", [question, question]))
    assert calls == []


def test_answer_count_mismatch_is_rejected(tracker: CostTracker) -> None:
    """Vérifie qu'une réponse sans une entrée par question est refusée."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "typesafe/jev-latest", "answers": [], "usage": {"prompt_tokens": 10}},
        )

    model = _mock_model(tracker, handler)
    with pytest.raises(JevDecisionError, match="une réponse par question"):
        asyncio.run(model.decide("contenu", [_predicate_question()]))


def test_long_text_input_is_truncated(tracker: CostTracker) -> None:
    """Vérifie que les entrées texte dépassant la limite sont tronquées avec un marqueur."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_predicate_response())

    model = _mock_model(tracker, handler, max_input_chars=20)
    asyncio.run(model.decide("x" * 100, [_predicate_question()]))

    payload = captured["payload"]
    assert isinstance(payload, dict)
    truncated = payload["input"]
    assert isinstance(truncated, str)
    assert len(truncated) == 20
    assert truncated.endswith("[tronqué]")


def test_choice_question_requires_valid_options() -> None:
    """Vérifie la validation Pydantic des options d'une question choice."""
    with pytest.raises(ValidationError):
        DecisionQuestion(
            type=DecisionQuestionType.CHOICE,
            name="action",
            instructions="Quelle action ?",
            choices=[DecisionOption(value="seule")],
        )


def test_factory_and_provider_configuration(tracker: CostTracker) -> None:
    """Vérifie que la factory accepte le fournisseur ofoxai et retourne le client Jev."""
    assert tracker.get_provider("jev") == "ofoxai"
    model = create_decision_model(
        tracker,
        api_key="test-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_predicate_response())),
    )
    assert isinstance(model, JevDecisionModel)
    assert model.model_id == "typesafe/jev-latest"


def test_packaged_router_config_matches_source() -> None:
    """Vérifie que les deux copies de router.yaml restent identiques."""
    import yaml

    source = yaml.safe_load((REPO_ROOT / "config" / "router.yaml").read_text(encoding="utf-8"))
    packaged = yaml.safe_load(
        (REPO_ROOT / "src" / "orchestrator" / "config" / "router.yaml").read_text(encoding="utf-8")
    )
    assert source == packaged
