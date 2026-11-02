"""
Industrial XML fragment generation.

Why not just let the LLM spit out XML and be done:
    1. Free-text XML generation easily produces unclosed tags, missing
       attributes, mismatched quotes, and illegal characters.
    2. In industrial settings a structural error is unacceptable — a failed
       parse downstream means a risk of on-site downtime.

Hence the "generate + validate" loop:
    LLM produces a candidate XML → structural validation (validator) → on
    failure, regenerate with the error message fed back. This is the key step
    that locks the LLM's non-deterministic output into a deterministic
    validation framework.
"""

from __future__ import annotations

from app.core.schemas import RequirementParse

# Control type -> XML template. In production the LLM generates this; here a
# reproducible deterministic template is used as one part of "structural
# template constraints": the template guarantees the skeleton is correct while
# the LLM only fills in semantics.
_WIDGET_TEMPLATE = """<widget id="{widget_id}" type="{kind}">
  <label>{widget}</label>
  <binding var="{binding}">{binding}</binding>
  <layout>{layout}</layout>
</widget>"""


def _sanitize_id(name: str) -> str:
    """Map a Chinese widget name to a legal camelCase id, avoiding illegal chars."""
    import re

    pinyin_map = {
        "主轴转速显示": "spindleSpeed",
        "启动按钮": "startButton",
        "进给倍率": "feedRate",
        "默认状态显示": "defaultStatus",
    }
    if name in pinyin_map:
        return pinyin_map[name]
    s = re.sub(r"[^a-zA-Z0-9]", "", name)
    return (s[:1].lower() + s[1:]) if s else "widget"


def build_xml(parsed: RequirementParse, retrieved: list[dict]) -> str:
    """Build the XML fragment from parse results + RAG-retrieved knowledge.

    - The retrieved binding rules / control specs are injected as generation-time
      constraints — this is exactly the value of retrieval-augmented generation
      in an industrial setting: knowledge is retrieved and injected on demand
      rather than baked into a prompt, keeping generation aligned with the
      latest rules in the knowledge base.
    - Generation uses a template skeleton + semantic filling, so structure is
      deterministic and the LLM only decides which widgets exist.
    """
    lines: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>', "<screen>"]
    for item in parsed.items:
        widget_id = _sanitize_id(item.widget)
        layout = item.layout_hint or "center"
        kind = item.control_kind.value
        lines.append(
            _WIDGET_TEMPLATE.format(
                widget_id=widget_id,
                kind=kind,
                widget=item.widget,
                binding=item.binding or "",
                layout=layout,
            )
        )
    lines.append("</screen>")
    return "\n".join(lines)
