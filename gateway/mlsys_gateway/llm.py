"""Streaming LLM clients: OpenAI-compatible (vLLM / llama.cpp) and Anthropic (demo profile)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anthropic.types import MessageParam
from mlsys_common.models import Usage


@dataclass
class StreamEvent:
    kind: str  # token | usage | done
    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None  # stop | length | ... (set on the done event)
    raw: dict = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def stream(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> AsyncIterator[StreamEvent]: ...


class OpenAICompatLLM:
    """Streams /v1/chat/completions from vLLM or llama.cpp. Thinking disabled via chat_template_kwargs (Qwen3.x)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        disable_thinking: bool = True,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.disable_thinking = disable_thinking
        self._client = client or httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        finish_reason: str | None = None
        async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=body) as r:
            if r.status_code >= 400:
                detail = (await r.aread()).decode(errors="replace")[:500]
                raise RuntimeError(f"LLM backend {r.status_code}: {detail}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    tok = delta.get("content")
                    if tok:
                        yield StreamEvent("token", tok)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                if obj.get("usage"):
                    u = obj["usage"]
                    yield StreamEvent(
                        "usage",
                        usage=Usage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        ),
                    )
        yield StreamEvent("done", finish_reason=finish_reason)

    async def models(self) -> list[str]:
        r = await self._client.get(f"{self.base_url}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


class AnthropicLLM:
    """Claude (demo generation, spec §5.11). Same prompt templates and citation contract."""

    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._c = AsyncAnthropic(api_key=api_key)

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        async with self._c.messages.stream(
            model=self.model,
            system=system,
            messages=cast("Iterable[MessageParam]", msgs),
            max_tokens=max_tokens,
        ) as s:
            async for tok in s.text_stream:
                yield StreamEvent("token", tok)
            final = await s.get_final_message()
        u = final.usage
        yield StreamEvent(
            "usage",
            usage=Usage(
                prompt_tokens=u.input_tokens,
                completion_tokens=u.output_tokens,
                total_tokens=u.input_tokens + u.output_tokens,
            ),
        )
        stop = final.stop_reason or ""
        yield StreamEvent(
            "done", finish_reason={"end_turn": "stop", "max_tokens": "length"}.get(stop, stop)
        )


class FakeLLM:
    """Deterministic stand-in for tests and the `make dev` no-GPU path."""

    model = "fake-llm"

    def __init__(
        self,
        answer: str = "Quantization reduces numeric precision to shrink models [1]. It trades accuracy for memory [2].",
        delay_s: float = 0.0,
    ) -> None:
        self.answer = answer
        self.delay_s = delay_s
        self.calls: list[list[dict[str, str]]] = []

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        import asyncio

        self.calls.append(messages)
        words = self.answer.split(" ")
        for i, w in enumerate(words):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield StreamEvent("token", w if i == len(words) - 1 else w + " ")
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        yield StreamEvent(
            "usage",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(words),
                total_tokens=prompt_tokens + len(words),
            ),
        )
        yield StreamEvent("done", finish_reason="length" if len(words) >= max_tokens else "stop")


def now_ms() -> float:
    return time.perf_counter() * 1000.0
