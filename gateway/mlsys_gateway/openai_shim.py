"""OpenAI-compatible shim (spec §5.5): POST /v1/chat/completions, GET /v1/models.

The last user message is the question. Citations cannot ride custom SSE events here, so they
degrade gracefully into a trailing markdown "Sources" block in the assistant content.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from mlsys_common.models import AskRequest
from pydantic import BaseModel, Field

from mlsys_gateway.pipeline import Deps, PipelineEvent, ask


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = Field(default=None, description="non-standard: number of cited chunks")


def extract_question(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            if isinstance(m.content, str):
                return m.content
            return " ".join(p.get("text", "") for p in m.content if p.get("type") == "text")
    raise ValueError("no user message")


def sources_block(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = [f"[{c['n']}] {c['heading_path']} (chunk {c['chunk_id']})" for c in citations]
    return "\n\n**Sources**\n" + "\n".join(lines)


def _chunk(
    cid: str,
    model: str,
    created: int,
    delta: dict,
    finish: str | None = None,
    usage: dict | None = None,
) -> dict:
    d: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        d["usage"] = usage
    return d


async def stream_chat(deps: Deps, req: ChatCompletionRequest) -> AsyncIterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = req.model or deps.llm.model
    citations: list[dict] = []
    usage: dict | None = None
    yield (
        "data: "
        + json.dumps(_chunk(cid, model, created, {"role": "assistant", "content": ""}))
        + "\n\n"
    )
    async for ev in ask(deps, AskRequest(question=extract_question(req.messages), top_k=req.top_k)):
        if ev.event == "citations":
            citations = ev.data
        elif ev.event == "token":
            yield "data: " + json.dumps(_chunk(cid, model, created, {"content": ev.data})) + "\n\n"
        elif ev.event == "done":
            u = ev.data["usage"]
            usage = {
                "prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["completion_tokens"],
                "total_tokens": u["total_tokens"],
            }
        elif ev.event == "error":
            yield (
                "data: "
                + json.dumps(
                    _chunk(cid, model, created, {"content": f"\n\n[error: {ev.data['message']}]"})
                )
                + "\n\n"
            )
    tail = sources_block(citations)
    if tail:
        yield "data: " + json.dumps(_chunk(cid, model, created, {"content": tail})) + "\n\n"
    yield (
        "data: " + json.dumps(_chunk(cid, model, created, {}, finish="stop", usage=usage)) + "\n\n"
    )
    yield "data: [DONE]\n\n"


async def complete_chat(deps: Deps, req: ChatCompletionRequest) -> dict:
    parts: list[str] = []
    citations: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    events: list[PipelineEvent] = []
    async for ev in ask(deps, AskRequest(question=extract_question(req.messages), top_k=req.top_k)):
        events.append(ev)
        if ev.event == "citations":
            citations = ev.data
        elif ev.event == "token":
            parts.append(ev.data)
        elif ev.event == "done":
            u = ev.data["usage"]
            usage = {
                "prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["completion_tokens"],
                "total_tokens": u["total_tokens"],
            }
    content = "".join(parts) + sources_block(citations)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or deps.llm.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
