"""LangGraph 状态机端到端测试 + LLM 客户端降级测试。"""

import pytest

from app.core.schemas import AgentState
from app.graph.agent_graph import route_after_parse, route_after_validate, run_agent
from app.llm.client import LLMClient
from app.rag.vector_store import load_knowledge_from_dir


@pytest.fixture(autouse=True)
def _load_knowledge():
    load_knowledge_from_dir("data/knowledge")
    yield


@pytest.mark.asyncio
async def test_run_agent_full_pipeline():
    """完整链路：自然语言 -> XML -> 校验 -> 部署。"""
    state = await run_agent("顶部显示主轴转速，左侧放启动按钮")
    assert state.final_answer != ""
    assert "<screen>" in state.xml
    assert state.validation is not None
    assert state.validation.valid is True
    assert state.step_count >= 4  # parse -> retrieve -> generate -> validate -> ...


@pytest.mark.asyncio
async def test_run_agent_retrieves_knowledge():
    state = await run_agent("做一个主轴转速显示界面")
    # RAG 应召回相关知识（mock 模式下至少有一条）
    assert len(state.retrieved) > 0


def test_route_after_parse_clarify():
    # 需要澄清时路由到 clarify
    state = AgentState(requirement="", needs_clarify=True)
    assert route_after_parse(state) == "clarify"

    state2 = AgentState(requirement="", needs_clarify=False)
    assert route_after_parse(state2) == "retrieve"


def test_route_after_validate_retry():
    # 校验失败且需重试 -> 回炉 generate
    state = AgentState(requirement="", needs_retry=True)
    assert route_after_validate(state) == "generate"

    state2 = AgentState(requirement="", needs_retry=False)
    assert route_after_validate(state2) == "confirm"


@pytest.mark.asyncio
async def test_llm_client_mock_structured_output():
    client = LLMClient()
    from app.core.schemas import RequirementParse

    result = await client.generate_structured("顶部显示主轴转速，左侧放启动按钮", RequirementParse)
    assert isinstance(result, RequirementParse)
    assert len(result.items) >= 2


@pytest.mark.asyncio
async def test_llm_client_mock_text():
    client = LLMClient()
    result = await client.generate_text("你好")
    assert isinstance(result, str)
    assert result.startswith("[mock]")
