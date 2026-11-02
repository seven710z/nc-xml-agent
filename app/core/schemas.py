"""
Structured data models — Pydantic schemas for agent state and business entities.

Why structured output matters:
    A raw LLM emits free text that downstream code cannot parse reliably. Strong
    Pydantic schemas, combined with the LLM's JSON / function-call constraints,
    freeze the contract between "model generation" and "code consumption". The
    LangGraph `State` must also be a serializable, validatable structure —
    otherwise values passed between nodes degrade into loose dicts and bugs
    become hard to localize.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured output for the requirement-parsing stage
# ---------------------------------------------------------------------------
class ControlKind(str, Enum):
    """CNC UI control types (also referenced by the RAG knowledge base)."""

    BUTTON = "button"
    DISPLAY = "display"  # numeric / status display
    INPUT = "input"  # parameter input box
    GRAPH = "graph"  # curve / chart
    TABLE = "table"  # data table
    LABEL = "label"  # text label


class RequirementItem(BaseModel):
    """A single UI element parsed from the requirement."""

    widget: str = Field(description="控件中文名，如：主轴转速显示")
    control_kind: ControlKind = Field(description="控件类型")
    binding: str | None = Field(default=None, description="绑定的数控变量/寄存器，如 #5001")
    data_source: str | None = Field(default=None, description="数据来源说明")
    layout_hint: str | None = Field(default=None, description="布局位置提示，如 顶部/左侧")


class RequirementParse(BaseModel):
    """Full parse result — the target schema for LLM structured output."""

    intent: str = Field(description="用户意图一句话概括")
    items: list[RequirementItem] = Field(description="解析出的界面元素列表")
    ambiguity: list[str] = Field(default_factory=list, description="需要向用户澄清的歧义点")


# ---------------------------------------------------------------------------
# XML validation results
# ---------------------------------------------------------------------------
class ValidationIssue(BaseModel):
    severity: str = Field(description="error | warning")
    line: int | None = Field(default=None)
    message: str = Field(description="问题描述")
    rule: str = Field(description="触发的校验规则名，如 wellformed / schema / binding")


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------
class AgentState(BaseModel):
    """Shared state for the LangGraph StateGraph.

    Pydantic instead of a plain dict: values passed between nodes stay typed and
    validatable, and it enables checkpoint/resume (LangGraph checkpoints depend
    on a serializable state).
    """

    # Input
    requirement: str = ""  # user's natural-language requirement
    # Intermediate products
    parsed: RequirementParse | None = None  # parse result
    retrieved: list[dict] = Field(default_factory=list)  # RAG-retrieved chunks
    xml: str = ""  # generated XML fragment
    validation: ValidationResult | None = None
    # Flow control
    needs_clarify: bool = False
    needs_retry: bool = False
    retry_count: int = 0  # bounded together with max_steps
    step_count: int = 0
    # Human-in-the-loop
    confirmed: bool = False
    # Results
    final_answer: str = ""
    error: str | None = None
    tool_logs: list[str] = Field(default_factory=list)  # sanitized tool-call logs


# ---------------------------------------------------------------------------
# Tool-calling argument schemas
# ---------------------------------------------------------------------------
class DeployRequest(BaseModel):
    """Controlled-deploy tool input — a dangerous op that requires human confirmation."""

    xml: str = Field(description="要部署的 XML 片段内容")
    target_device: str = Field(description="目标数控设备标识")
    confirmed: bool = Field(default=False, description="是否已通过人工确认")


class DeployResult(BaseModel):
    ok: bool
    message: str
    device: str | None = None
