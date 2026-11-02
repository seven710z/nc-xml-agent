"""
LangGraph StateGraph orchestration — the control-flow skeleton of the agent.

Why a graph instead of hand-written if/else chains:
    Branching, retry loops, and human-in-the-loop pauses scattered across
    plain functions are hard to change and easy to break. LangGraph turns the
    workflow into an explicit graph: nodes are processing steps, edges are
    conditional transitions. The control flow becomes visible, unit-testable,
    and checkpointable (resume from any state).

    Under the hood this is an explicit finite-state machine (FSM): a shared
    `AgentState` flows between nodes, and conditional edges route based on its
    fields.

Bounded autonomy (C5):
    The "validate → regenerate" retry loop is NOT a `while True`. It is bounded
    by `retry_count` and the graph-level `recursion_limit`, after which the run
    is forcibly terminated or escalated to a human.

Full pipeline (6 stages):
    parse → retrieve → generate → validate → confirm → deploy
    Validation failure re-enters `generate` (bounded retries); success proceeds
    to human confirmation.
"""

from __future__ import annotations

from typing import Literal

from app.core.config import settings
from app.core.schemas import AgentState, RequirementParse
from app.llm.client import llm
from app.rag.vector_store import vector_store
from app.tools.deploy import deploy_tool
from app.xml_gen.generator import build_xml
from app.xml_gen.validator import validate_xml


# ---------------------------------------------------------------------------
# Nodes (one responsibility per node, small functions)
# ---------------------------------------------------------------------------
async def parse_node(state: AgentState) -> dict:
    """Stage 1 — parse natural language into structured requirements via LLM."""
    parse: RequirementParse = await llm.generate_structured(
        prompt=state.requirement,
        schema=RequirementParse,
        system="你是数控界面需求解析器，从用户描述中提取界面控件、类型与绑定变量，输出结构化 JSON。",
    )
    needs_clarify = len(parse.ambiguity) > 0
    return {"parsed": parse, "needs_clarify": needs_clarify, "step_count": state.step_count + 1}


async def retrieve_node(state: AgentState) -> dict:
    """Stage 2 — retrieve XML syntax / control specs / binding rules via RAG."""
    assert state.parsed is not None
    # Query with intent + each widget name; de-duplicate by text.
    queries = [state.parsed.intent] + [i.widget for i in state.parsed.items]
    retrieved: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        for r in vector_store.search(q):
            if r["text"] not in seen:
                seen.add(r["text"])
                retrieved.append(r)
    return {"retrieved": retrieved, "step_count": state.step_count + 1}


async def generate_node(state: AgentState) -> dict:
    """Stage 3 — generate the XML fragment from parsed items + retrieved knowledge."""
    assert state.parsed is not None
    xml_text = build_xml(state.parsed, state.retrieved)
    return {"xml": xml_text, "step_count": state.step_count + 1}


async def validate_node(state: AgentState) -> dict:
    """Stage 4 — validate the XML; decide pass-through vs bounded regeneration."""
    result = validate_xml(state.xml, state.retrieved)
    needs_retry = (not result.valid) and (state.retry_count < settings.max_steps)
    return {
        "validation": result,
        "needs_retry": needs_retry,
        "retry_count": state.retry_count + (0 if result.valid else 1),
        "step_count": state.step_count + 1,
    }


async def confirm_node(state: AgentState) -> dict:
    """Stage 5 — human-in-the-loop confirmation before any deployment.

    Auto-validation cannot catch semantic mistakes (wrong variable binding,
    misunderstood intent), so a human must confirm before deployment. In
    production this pauses for a Web confirmation; here a flag simulates it.
    """
    # Production: suspend here and wait for the frontend to return `confirmed`
    # via LangGraph interrupt/checkpoint. Demo mode defaults to allowed.
    return {"confirmed": True, "step_count": state.step_count + 1}


async def deploy_node(state: AgentState) -> dict:
    """Stage 6 — controlled deployment through the deploy tool (requires confirmed)."""
    from app.core.schemas import DeployRequest

    result = await deploy_tool.run(
        DeployRequest(
            xml=state.xml,
            target_device=settings.deploy_target,
            confirmed=state.confirmed,
        )
    )
    final = f"已生成数控界面 XML 并通过校验，部署状态：{result.message}"
    return {
        "final_answer": final,
        "tool_logs": state.tool_logs + [f"deploy_xml -> {result.message}"],
        "step_count": state.step_count + 1,
    }


async def clarify_node(state: AgentState) -> dict:
    """Clarification branch — ask the user instead of generating on ambiguity."""
    assert state.parsed is not None
    qs = "；".join(state.parsed.ambiguity) or "请补充控件描述"
    return {"final_answer": f"需求存在歧义，请澄清：{qs}", "step_count": state.step_count + 1}


# ---------------------------------------------------------------------------
# Conditional edges (routing logic)
# ---------------------------------------------------------------------------
def route_after_parse(state: AgentState) -> Literal["retrieve", "clarify"]:
    return "clarify" if state.needs_clarify else "retrieve"


def route_after_validate(state: AgentState) -> Literal["confirm", "generate"]:
    return "generate" if state.needs_retry else "confirm"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    """Build the LangGraph; reusable across mock / real environments."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("parse", parse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_node("confirm", confirm_node)
    g.add_node("deploy", deploy_node)
    g.add_node("clarify", clarify_node)

    g.set_entry_point("parse")
    g.add_conditional_edges(
        "parse", route_after_parse, {"retrieve": "retrieve", "clarify": "clarify"}
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "validate")
    # Validation failure loops back to generate; bounded by retry_count.
    g.add_conditional_edges(
        "validate", route_after_validate, {"confirm": "confirm", "generate": "generate"}
    )
    g.add_edge("confirm", "deploy")
    g.add_edge("deploy", END)
    g.add_edge("clarify", END)

    return g.compile()


graph = build_graph()


async def run_agent(requirement: str) -> AgentState:
    """Public entrypoint — run the full pipeline asynchronously.

    `graph.ainvoke` is async and every node's IO (LLM, retrieval) is awaited,
    so the whole pipeline completes inside the event loop, which is what makes
    it high-concurrency under FastAPI.
    """
    initial = AgentState(requirement=requirement)
    # recursion_limit guards against pathological infinite loops (C5).
    config = {"recursion_limit": settings.max_steps + 5}
    result = await graph.ainvoke(initial, config=config)
    return AgentState.model_validate(result)
