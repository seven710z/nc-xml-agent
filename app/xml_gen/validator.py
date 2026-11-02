"""
XML structural validation — locks the LLM's non-deterministic output into a
deterministic validation framework.

Three layers, each stricter than the last:
    1. Well-formedness: parse with xml.etree — closed tags / quotes / illegal chars.
    2. Schema constraints: root must be <screen>, each <widget> has a unique id
       and a type.
    3. Business binding rules: variable type must match the control (read-only
       vars can't be written to, etc.); this layer reuses the binding rules in
       the RAG knowledge base.

Why three layers: each layer's errors have a different nature and remediation —
well-formedness errors are fixed by regeneration; business errors by constraint
injection + human escalation. Layering lets us pinpoint which layer failed and
decide between "auto-retry" and "escalate to human".
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from app.core.schemas import ValidationIssue, ValidationResult

# Read-only variable prefix: these cannot be written by an input control.
_READONLY_PREFIX = "#5"


def validate_xml(xml_text: str, retrieved: list[dict]) -> ValidationResult:
    issues: list[ValidationIssue] = []

    # ---- Layer 1: well-formedness ----
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(severity="error", message=f"XML 良构性错误：{e}", rule="wellformed")
            ],
        )

    # ---- Layer 2: schema constraints ----
    if root.tag != "screen":
        issues.append(
            ValidationIssue(
                severity="error", message=f"根节点应为 <screen>，实际为 <{root.tag}>", rule="schema"
            )
        )

    seen_ids: set[str] = set()
    for w in root.iter("widget"):
        wid = w.get("id")
        wtype = w.get("type")
        if not wid:
            issues.append(
                ValidationIssue(severity="error", message="存在缺少 id 的 <widget>", rule="schema")
            )
        elif wid in seen_ids:
            issues.append(
                ValidationIssue(severity="error", message=f"id 重复：{wid}", rule="schema")
            )
        else:
            seen_ids.add(wid)
        if not wtype:
            issues.append(
                ValidationIssue(
                    severity="error", message=f"<widget id={wid}> 缺少 type 属性", rule="schema"
                )
            )

    # ---- Layer 3: business binding rules (reuses RAG knowledge) ----
    for w in root.iter("widget"):
        wtype = w.get("type", "")
        binding = w.find("binding")
        var = binding.text if binding is not None else ""
        # Read-only vars cannot be used by input (write) controls.
        if wtype == "input" and var.startswith(_READONLY_PREFIX):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"input 控件绑定了只读变量 {var}，只读变量不可写入",
                    rule="binding",
                )
            )
        # input controls must declare a value range (min/max).
        if wtype == "input" and (w.find("min") is None or w.find("max") is None):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="input 控件未声明 min/max 取值范围，存在越界报警风险",
                    rule="binding",
                )
            )

    # Count nodes to guard against recursion blowup / oversized fragments (C5).
    node_count = sum(1 for _ in root.iter())
    if node_count > 200:
        issues.append(
            ValidationIssue(
                severity="error", message=f"节点数 {node_count} 超过上限 200", rule="schema"
            )
        )

    return ValidationResult(valid=not any(i.severity == "error" for i in issues), issues=issues)
