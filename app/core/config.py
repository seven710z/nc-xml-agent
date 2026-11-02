"""
Centralized configuration — everything injected via env vars; no hardcoded secrets (C6).

Rationale:
    One of the dividing lines between a production-grade agent and a demo is
    whether timeouts / retries / budgets / caps are configurable. All of these
    (LLM timeout, retry backoff, token budget, max autonomous steps, RAG top-k,
    MCP server address) converge here and are read from environment variables
    via pydantic-settings, so tuning or switching environments (mock / real
    model / real MCP) never requires a code change.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Runtime mode: mock = run the vertical slice with no API key ----
    app_mode: str = "mock"  # mock | real

    # ---- LLM (Qwen2.5 base, OpenAI-compatible protocol) ----
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""  # MUST come from env / .env, never hardcoded
    llm_model: str = "qwen2.5-7b-instruct"  # LoRA-adapted model name
    llm_timeout_s: float = 30.0  # per-call timeout (seconds)
    llm_max_retries: int = 2  # max retries for retryable errors
    llm_temperature: float = 0.1  # low temp for structured generation

    # ---- Autonomous-loop budget (C5 bounded autonomy) ----
    max_steps: int = 10  # LangGraph max steps
    token_budget: int = 8192  # per-task token budget
    wall_clock_budget_s: float = 120.0  # wall-clock cap

    # ---- RAG retrieval ----
    chroma_persist_dir: str = "./data/chroma"  # Chroma persistence dir
    embedding_model: str = "BAAI/bge-small-zh-v1.5"  # Chinese embedding model
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35  # below this = no hit

    # ---- MCP tool service ----
    mcp_server_url: str = "http://127.0.0.1:8100"  # NC XML tool MCP endpoint

    # ---- XML generation & validation ----
    xml_max_depth: int = 12  # max XML tree depth, guards recursion
    xml_max_nodes: int = 200  # max nodes per fragment

    # ---- Deployment (hardware layer is out of scope; config placeholder only) ----
    deploy_target: str = "cnc-dev-01"  # target CNC device id


@lru_cache
def get_settings() -> Settings:
    """Process-level singleton, avoids re-parsing .env on every access."""
    return Settings()


settings = get_settings()
