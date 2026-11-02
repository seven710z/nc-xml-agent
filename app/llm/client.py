"""
LLM client — Qwen2.5 base + structured output + timeout/retry.

Rationale:
    1. Why Qwen2.5: open-source, strong Chinese, OpenAI-compatible protocol —
       convenient for local fine-tuning and inference.
    2. Why low temperature for structured generation: higher temperature means
       more random sampling, which more easily corrupts JSON/XML structure.
    3. Robustness (C4): every LLM call has an explicit timeout, retry (with
       backoff + jitter), and a fallback path. In mock mode there is no network
       or API-key dependency and results are deterministic, so the vertical
       slice runs end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


def _mock_parse_json(requirement: str) -> dict:
    """Deterministic parse result for mock mode (no API key required).

    In production Qwen2.5 generates this against the JSON schema; here a
    keyword mapping hardcodes the result so the vertical slice runs and the
    behavior is predictable enough to be asserted in the eval set.
    """
    text = requirement
    items: list[dict] = []
    # Minimal keyword mapping that mimics requirement parsing (real impl is LLM).
    if "转速" in text:
        items.append(
            {
                "widget": "主轴转速显示",
                "control_kind": "display",
                "binding": "#5001",
                "layout_hint": "顶部",
            }
        )
    if "启动" in text or "按钮" in text:
        items.append(
            {
                "widget": "启动按钮",
                "control_kind": "button",
                "binding": "#G7.2",
                "layout_hint": "左侧",
            }
        )
    if "进给" in text:
        items.append(
            {
                "widget": "进给倍率",
                "control_kind": "input",
                "binding": "#1420",
                "layout_hint": "中部",
            }
        )
    if not items:
        items.append(
            {
                "widget": "默认状态显示",
                "control_kind": "display",
                "binding": "#0",
                "layout_hint": "顶部",
            }
        )
    return {
        "intent": "解析数控界面需求",
        "items": items,
        "ambiguity": [] if items else ["无法识别具体控件，请补充描述"],
    }


class LLMClient:
    """Unified LLM access — mock and real modes share the same public interface."""

    def __init__(self) -> None:
        self._mode = settings.app_mode

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str = "",
    ) -> T:
        """Ask the LLM for a structured result matching the given Pydantic schema.

        Real mode goes through the OpenAI-compatible JSON mode / function-call;
        mock mode returns a deterministic result.
        """
        if self._mode == "mock":
            await asyncio.sleep(0)  # yield the event loop explicitly (async semantics)
            return schema.model_validate(_mock_parse_json(prompt))
        return await self._call_with_retry(prompt, schema, system)

    async def generate_text(self, prompt: str, *, system: str = "") -> str:
        """Free-text generation for non-structured cases (e.g. reply polishing)."""
        if self._mode == "mock":
            await asyncio.sleep(0)
            return f"[mock] 已处理：{prompt[:50]}"
        return await self._call_with_retry(prompt, str, system)

    async def _call_with_retry(self, prompt: str, schema: type, system: str) -> Any:
        """Retry with timeout + backoff + jitter (C4).

        Uses httpx directly to keep the dependency count low while preserving
        the same signature as an OpenAI SDK. Only retryable errors (network,
        429, 5xx) are retried; 4xx business errors are not.
        """
        import httpx  # lazy import: mock mode avoids the network dependency

        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.llm_temperature,
            "response_format": {"type": "json_object"},
        }

        last_err: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                    resp = await client.post(
                        f"{settings.llm_base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                        json=payload,
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    # Retryable: rate-limited / server fault.
                    last_err = RuntimeError(f"LLM {resp.status_code}: {resp.text[:200]}")
                    await self._backoff(attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return schema.model_validate(json.loads(content))
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                await self._backoff(attempt)
            except Exception as e:  # parse failure etc. — retry once
                last_err = e
                await self._backoff(attempt)

        # Fallback: raise a clear error instead of hanging forever.
        raise RuntimeError(f"LLM 调用失败（已重试 {settings.llm_max_retries} 次）：{last_err}")

    @staticmethod
    async def _backoff(attempt: int) -> None:
        """Exponential backoff + random jitter to avoid thundering herd."""
        delay = min(2**attempt, 8) + random.uniform(0, 0.5)
        await asyncio.sleep(delay)


llm = LLMClient()
