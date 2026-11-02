"""配置与 schema 单元测试。"""

from app.core.config import settings
from app.core.schemas import AgentState, ControlKind, RequirementParse


def test_settings_defaults():
    assert settings.app_mode == "mock"
    assert settings.max_steps == 10
    assert settings.llm_timeout_s > 0
    assert settings.rag_top_k > 0


def test_agent_state_defaults():
    state = AgentState(requirement="测试")
    assert state.requirement == "测试"
    assert state.step_count == 0
    assert state.retrieved == []
    assert state.retry_count == 0


def test_control_kind_enum():
    assert ControlKind.DISPLAY.value == "display"
    assert ControlKind.BUTTON.value == "button"
    assert ControlKind.INPUT.value == "input"


def test_requirement_parse_validation():
    p = RequirementParse(intent="x", items=[])
    assert p.ambiguity == []
