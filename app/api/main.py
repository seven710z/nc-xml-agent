"""
FastAPI async backend — the agent's HTTP entrypoint.

Why async for an agent backend:
    1. LLM calls and vector retrieval are IO-bound and high-latency (hundreds of
       ms to seconds). A synchronous thread/process model would burn through the
       thread pool fast; asyncio is single-threaded with an event loop that
       yields the CPU while waiting on IO — a natural fit for concurrent IO.
    2. FastAPI sits on ASGI + Starlette; declaring a route as `async def` runs
       it in the event loop, seamlessly integrated with the asyncio ecosystem.
    3. trace_id spans the whole pipeline (observability): every request gets a
       trace_id that propagates through the agent, enabling log correlation and
       fast localization — one dividing line between production and demo.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.graph.agent_graph import run_agent
from app.rag.vector_store import load_knowledge_from_dir

app = FastAPI(title="数控 XML 界面生成与自动部署 AI Agent", version="0.1.0")

# CORS (frontend is out of scope; placeholder to allow cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    requirement: str = Field(description="用户自然语言描述的数控界面需求")
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="链路追踪 ID")


class GenerateResponse(BaseModel):
    trace_id: str
    final_answer: str
    xml: str
    validation_valid: bool
    latency_ms: int


@app.on_event("startup")
async def startup() -> None:
    """Load the knowledge base and build the index on startup."""
    load_knowledge_from_dir("data/knowledge")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mode": settings.app_mode}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Core endpoint: natural-language requirement -> full agent pipeline -> XML + deploy state."""
    start = time.perf_counter()
    # run_agent is async, so requests run concurrently inside the event loop.
    state = await run_agent(req.requirement)
    latency = int((time.perf_counter() - start) * 1000)

    return GenerateResponse(
        trace_id=req.trace_id,
        final_answer=state.final_answer,
        xml=state.xml,
        validation_valid=state.validation.valid if state.validation else False,
        latency_ms=latency,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
