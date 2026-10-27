from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.schemas import ExecutionResult, Plan, Step, StepStatus, StepType
from orchestrator.utils.parsing import extract_json_block, format_validation_error


def test_valid_step_and_plan() -> None:
    """Vérifie la création valide d'un Step et d'un Plan."""
    step = Step(
        id="step_1",
        description="Créer le fichier main.py",
        type=StepType.FILE_OP,
        tools_allowed=["write_file"],
        success_criteria="Le fichier main.py existe.",
        expected_outputs=["main.py"],
        depends_on=[],
    )
    plan = Plan(
        goal="Créer une application minimale",
        summary="Une seule étape d'écriture.",
        steps=[step],
    )
    assert plan.steps[0].id == "step_1"
    assert plan.steps[0].type == StepType.FILE_OP


def test_plan_rejects_empty_steps() -> None:
    """Vérifie qu'un Plan sans étape lève une ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Plan(goal="Vide", summary="Aucune étape", steps=[])
    formatted = format_validation_error(exc_info.value)
    assert "steps" in formatted


def test_execution_result_schema() -> None:
    """Vérifie la sérialisation et validation de ExecutionResult."""
    res = ExecutionResult(
        step_id="step_1",
        status=StepStatus.SUCCESS,
        output="Fichier créé.",
        justification="L'outil write_file a réussi.",
    )
    assert res.status == StepStatus.SUCCESS
    assert res.tool_calls == []
    assert res.error is None


def test_extract_json_block_from_markdown() -> None:
    """Vérifie l'extraction d'un bloc JSON entouré de balises markdown."""
    raw = 'Voici la réponse :\n```json\n{"goal": "test", "summary": "ok", "steps": []}\n```'
    extracted = extract_json_block(raw)
    assert extracted["goal"] == "test"
