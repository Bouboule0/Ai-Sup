from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
import httpx
import pytest

from orchestrator.cli import _resolve_config_dir
from orchestrator.executor.anthropic_executor import AnthropicExecutor
from orchestrator.executor.flash_executor import FlashExecutor
from orchestrator.runtime.cost_tracker import CostTracker
from orchestrator.schemas import Step, StepStatus, StepType
from orchestrator.tools.file_system import register_filesystem_tools
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.shell import is_command_allowed, load_allowed_commands


class _FakeGeminiUsage:
    prompt_token_count = 10
    candidates_token_count = 10


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = _FakeGeminiUsage()


class _FakeGeminiModels:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def generate_content(self, **kwargs: Any) -> _FakeGeminiResponse:
        return _FakeGeminiResponse(self._payload)


class _FakeGeminiAio:
    def __init__(self, payload: str) -> None:
        self.models = _FakeGeminiModels(payload)


class _FakeGeminiClient:
    def __init__(self, payload: str) -> None:
        self.aio = _FakeGeminiAio(payload)


def test_finding_1_flash_executor_rejects_in_progress_without_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: FlashExecutor ne doit jamais promouvoir status=in_progress sans tool_calls en SUCCESS."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    config_dir = _resolve_config_dir()
    tracker = CostTracker(max_budget=5.0, models_config_path=config_dir / "models.yaml")
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    payload = (
        '{"step_id": "step_1", "status": "in_progress", '
        '"output": "Thinking...", "justification": "Not done yet", "tool_calls": []}'
    )
    executor = FlashExecutor(
        cost_tracker=tracker,
        tool_registry=registry,
    )
    executor._client = _FakeGeminiClient(payload)  # type: ignore[assignment]
    step = Step(
        id="step_1",
        description="Write app.py",
        type=StepType.FILE_OP,
        tools_allowed=["write_file"],
        success_criteria="app.py created",
        expected_outputs=["app.py"],
    )
    result = asyncio.run(executor.execute(step, {}))
    assert result.status == StepStatus.FAILED
    assert result.error == "IncompleteTurnProtocolError"


def test_finding_1_flash_executor_verifies_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: FlashExecutor doit échouer si status=success mais expected_outputs est absent du disque."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    config_dir = _resolve_config_dir()
    tracker = CostTracker(max_budget=5.0, models_config_path=config_dir / "models.yaml")
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    payload = (
        '{"step_id": "step_1", "status": "success", '
        '"output": "Done", "justification": "Claimed created", "tool_calls": []}'
    )
    executor = FlashExecutor(
        cost_tracker=tracker,
        tool_registry=registry,
    )
    executor._client = _FakeGeminiClient(payload)  # type: ignore[assignment]
    step = Step(
        id="step_1",
        description="Write missing.py",
        type=StepType.FILE_OP,
        tools_allowed=["write_file"],
        success_criteria="missing.py exists",
        expected_outputs=["missing.py"],
    )
    result = asyncio.run(executor.execute(step, {}))
    assert result.status == StepStatus.FAILED
    assert result.error == "MissingExpectedOutputError"

    # Une fois le fichier réellement créé, l'étape réussit
    (tmp_path / "missing.py").write_text("x = 1\n", encoding="utf-8")
    result_ok = asyncio.run(executor.execute(step, {}))
    assert result_ok.status == StepStatus.SUCCESS


def test_finding_1_anthropic_executor_rejects_in_progress_and_missing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: AnthropicExecutor rejette in_progress sans outils et vérifie expected_outputs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    config_dir = _resolve_config_dir()
    tracker = CostTracker(max_budget=5.0, models_config_path=config_dir / "models.yaml")
    registry = ToolRegistry()
    register_filesystem_tools(registry, tmp_path)

    executor = AnthropicExecutor(
        cost_tracker=tracker,
        tool_registry=registry,
        model_key="claude-haiku-5.5",
    )

    current_input: dict[str, Any] = {
        "step_id": "step_1",
        "status": "in_progress",
        "output": "Wait",
        "justification": "No tools called",
        "tool_calls": [],
    }

    async def _fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "usage": {"input_tokens": 10, "output_tokens": 10},
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_execution_result",
                        "input": current_input,
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)
    step = Step(
        id="step_1",
        description="Create out.py",
        type=StepType.FILE_OP,
        tools_allowed=["write_file"],
        success_criteria="out.py exists",
        expected_outputs=["out.py"],
    )
    res1 = asyncio.run(executor.execute(step, {}))
    assert res1.status == StepStatus.FAILED
    assert res1.error == "IncompleteTurnProtocolError"

    current_input = {
        "step_id": "step_1",
        "status": "success",
        "output": "Done",
        "justification": "Finished",
        "tool_calls": [],
    }
    res2 = asyncio.run(executor.execute(step, {}))
    assert res2.status == StepStatus.FAILED
    assert res2.error == "MissingExpectedOutputError"


def test_finding_2_resolve_config_dir_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 2: _resolve_config_dir fonctionne même hors du checkout git via orchestrator/config."""
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_config_dir()
    assert (resolved / "models.yaml").exists()
    assert (resolved / "router.yaml").exists()


def test_finding_4_shell_blocks_inline_code_and_path_traversal(tmp_path: Path) -> None:
    """Finding 4: is_command_allowed bloque python -c et toute traversée hors de project_root."""
    config_dir = _resolve_config_dir()
    allowed = load_allowed_commands(config_dir / "router.yaml")

    # Cas exact reproduit par CodeRabbit
    exploit_cmd = 'python -c "print(open(\'../outside.txt\').read())"'
    assert is_command_allowed(exploit_cmd, allowed, project_root=tmp_path) is False

    # Traversée de chemin dans un argument
    assert is_command_allowed("pytest ../outside_tests", allowed, project_root=tmp_path) is False
    assert is_command_allowed("python /etc/passwd", allowed, project_root=tmp_path) is False

    # Commandes légitimes confinées au projet
    assert is_command_allowed("pytest -q", allowed, project_root=tmp_path) is True
    assert is_command_allowed("python test_app.py", allowed, project_root=tmp_path) is True


@pytest.mark.parametrize("command", [
    'python -Bc "print(1)"', 'python -Be "print(1)"',
    'python -Bc="print(1)"', 'python -Be="print(1)"',
])
def test_shell_rejects_clustered_inline_execution(command: str, tmp_path: Path) -> None:
    assert not is_command_allowed(command, ["python"], project_root=tmp_path)


@pytest.mark.parametrize("command, expected", [
    ("pytest -k=case", True),
    ("pytest -k=case=example", True),
    ("pytest -k=../outside", False),
])
def test_shell_checks_short_option_values_as_paths(
    command: str, expected: bool, tmp_path: Path,
) -> None:
    assert is_command_allowed(command, ["pytest"], project_root=tmp_path) is expected


@pytest.mark.parametrize("executor_class", [FlashExecutor, AnthropicExecutor])
def test_executor_fails_for_unregistered_tools(
    executor_class, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    tracker = CostTracker(5, _resolve_config_dir() / "models.yaml")
    executor = executor_class(cost_tracker=tracker, tool_registry=ToolRegistry())
    step = Step(id="step_1", type=StepType.CODE, description="work", success_criteria="done",
                tools_allowed=["missing_tool"])
    result = asyncio.run(executor.execute(step, {}))
    assert result.status == StepStatus.FAILED
    assert result.step_id == step.id
    assert "missing_tool" in (result.error or "")
    assert tracker.total_cost() == 0


@pytest.mark.parametrize("role", ["planner", "executor"])
@pytest.mark.parametrize("counts, expected", [
    ({"candidates_token_count": 10, "thoughts_token_count": 7}, 17),
    ({"candidates_token_count": None, "thoughts_token_count": 7}, 7),
    ({"candidates_token_count": 10, "thoughts_token_count": None}, 10),
    ({}, 0),
    (None, 0),
])
def test_gemini_accounts_for_thinking_tokens_and_uses_json_schema(
    role: str, counts, expected: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from orchestrator.planner.argon_planner import ArgonPlanner
    from orchestrator.schemas import ExecutionResult, Plan

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    tracker = CostTracker(5, _resolve_config_dir() / "models.yaml")
    step = Step(id="step_1", type=StepType.CODE, description="work", success_criteria="done")
    payload = (Plan(goal="goal", summary="plan", steps=[step]) if role == "planner" else
               ExecutionResult(step_id=step.id, status=StepStatus.SUCCESS, output="done", justification="done"))

    async def generate_content(**kwargs):
        if role == "executor":
            config = kwargs["config"]
            assert config.response_schema is None
            assert config.response_json_schema == ExecutionResult.model_json_schema()
        return SimpleNamespace(text=payload.model_dump_json(), usage_metadata=(
            SimpleNamespace(prompt_token_count=0, **counts) if counts is not None else None
        ))

    actor = (ArgonPlanner(tracker) if role == "planner" else FlashExecutor(tracker, ToolRegistry()))
    actor._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    result = asyncio.run(actor.plan("goal", {}) if role == "planner" else actor.execute(step, {}))
    assert result == payload
    price = tracker.get_model_pricing(role).output_cost_per_million
    assert tracker.total_cost() == pytest.approx(expected * price / 1_000_000)
