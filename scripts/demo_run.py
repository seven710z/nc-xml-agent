#!/usr/bin/env python
"""
Demo script — run one full pipeline asynchronously, without starting the HTTP server.

Purpose: quickly demonstrate the "natural language -> XML -> validate -> deploy"
loop at an interview. Run: python scripts/demo_run.py
"""

import asyncio

from app.graph.agent_graph import run_agent
from app.rag.vector_store import load_knowledge_from_dir


async def main() -> None:
    # Load the knowledge base and build the index (same as FastAPI startup).
    load_knowledge_from_dir("data/knowledge")

    # 一条典型需求：主轴转速显示 + 启动按钮
    requirement = "请做一个数控界面，顶部显示主轴转速，左侧放一个启动按钮，中部加一个进给倍率输入框"

    print("=" * 60)
    print("输入需求：", requirement)
    print("=" * 60)

    state = await run_agent(requirement)

    print("\n[需求解析结果]")
    if state.parsed:
        for item in state.parsed.items:
            print(f"  - {item.widget} | 类型={item.control_kind.value} | 绑定={item.binding}")

    print("\n[RAG 召回片段]（前 3 条）")
    for r in state.retrieved[:3]:
        print(f"  [{r['metadata'].get('category')}] {r['text'][:40]}... (score={r['score']})")

    print("\n[生成的 XML]")
    print(state.xml)

    if state.validation:
        print(f"\n[校验结果] valid={state.validation.valid}")
        for i in state.validation.issues:
            print(f"  [{i.severity}] {i.message}")

    print(f"\n[最终结果] {state.final_answer}")
    print(f"[累计步数] {state.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
