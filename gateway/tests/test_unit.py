"""Pure unit tests (no DB)."""

from mlsys_common.models import RetrievedChunk
from mlsys_gateway.openai_shim import (
    ChatCompletionRequest,
    ChatMessage,
    extract_question,
    sources_block,
)
from mlsys_gateway.prompts import ABSTAIN_PHRASE, build_messages, is_abstention, load_prompt


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        volume=1,
        chapter_num=1,
        chapter_title="T",
        heading_path=f"Vol 1 > Ch 1: T > S{i}",
        text=f"text {i}",
        content_hash=f"h{i}",
        fusion_score=0.1,
    )


def test_prompt_v1_loads_and_numbers_blocks():
    msgs = build_messages("What is X?", [_chunk(1), _chunk(2)], "v1")
    assert msgs[0]["role"] == "system" and ABSTAIN_PHRASE in msgs[0]["content"]
    assert "[1] (Vol 1 > Ch 1: T > S1)\ntext 1" in msgs[1]["content"]
    assert "[2] (Vol 1 > Ch 1: T > S2)\ntext 2" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: What is X?")
    assert set(load_prompt("v1")) == {"system", "user", "block"}


def test_abstention_detection():
    assert is_abstention("The book does not appear to cover this. The closest topic is [2].")
    assert not is_abstention("Quantization is [1].")


def test_extract_question_and_sources():
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="x"),
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="a"),
            ChatMessage(role="user", content=[{"type": "text", "text": "second"}]),
        ]
    )
    assert extract_question(req.messages) == "second"
    assert "[1] Vol 1 > Ch 1 (chunk 5)" in sources_block(
        [{"n": 1, "heading_path": "Vol 1 > Ch 1", "chunk_id": 5}]
    )
    assert sources_block([]) == ""
