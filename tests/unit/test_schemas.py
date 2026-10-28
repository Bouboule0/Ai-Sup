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


def test_plan_rejects_duplicate_step_ids_and_cycles() -> None:
    """Vérifie que Plan rejette les IDs dupliqués, auto-dépendances et cycles."""
    step_a = Step(
        id="step_1",
        description="A",
        type=StepType.FILE_OP,
        success_criteria="ok",
    )
    with pytest.raises(ValidationError, match="dupliqué"):
        Plan(goal="G", summary="S", steps=[step_a, step_a])

    step_c1 = Step(
        id="step_1",
        description="A",
        type=StepType.FILE_OP,
        success_criteria="ok",
        depends_on=["step_2"],
    )
    step_c2 = Step(
        id="step_2",
        description="B",
        type=StepType.FILE_OP,
        success_criteria="ok",
        depends_on=["step_1"],
    )
    with pytest.raises(ValidationError, match="Cycle de dépendances"):
        Plan(goal="G", summary="S", steps=[step_c1, step_c2])


def test_plan_validate_dependencies_rejects_missing_and_out_of_order() -> None:
    """Vérifie que validate_dependencies rejette les prérequis inexistants ou inversés."""
    step_missing = Step(
        id="step_1",
        description="A",
        type=StepType.FILE_OP,
        success_criteria="ok",
        depends_on=["missing_step"],
    )
    plan_missing = Plan(goal="G", summary="S", steps=[step_missing])
    with pytest.raises(ValueError, match="introuvable"):
        plan_missing.validate_dependencies()

    # Si 'missing_step' fait partie des étapes déjà complétées, le plan est valide
    plan_missing.validate_dependencies(completed_steps=["missing_step"])

    step_later = Step(
        id="step_1",
        description="A",
        type=StepType.FILE_OP,
        success_criteria="ok",
        depends_on=["step_2"],
    )
    step_first = Step(
        id="step_2",
        description="B",
        type=StepType.FILE_OP,
        success_criteria="ok",
        depends_on=[],
    )
    plan_order = Plan(goal="G", summary="S", steps=[step_later, step_first])
    with pytest.raises(ValueError, match="Ordre de dépendance invalide"):
        plan_order.validate_dependencies()
