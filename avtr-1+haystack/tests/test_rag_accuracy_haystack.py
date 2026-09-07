from __future__ import annotations

import asyncio
import json

import httpx
from scripts.rag_accuracy_test import (
    EvalCase,
    extract_haystack,
    request_once,
)


def test_extract_haystack_preserves_native_rank_and_score() -> None:
    ranked, scores = extract_haystack(
        {
            "documents": [
                {
                    "id": "chunk-2",
                    "content": "second",
                    "score": 0.94,
                    "document_id": "BC-002",
                },
                {
                    "id": "chunk-1",
                    "content": "first",
                    "score": 0.81,
                    "document_id": "BC-001",
                },
            ]
        },
        5,
    )
    assert ranked == ["BC-002", "BC-001"]
    assert scores == [0.94, 0.81]


def test_extract_haystack_deduplicates_document_ids() -> None:
    ranked, scores = extract_haystack(
        {
            "documents": [
                {"score": 0.9, "document_id": "BC-001"},
                {"score": 0.8, "document_id": "BC-001"},
            ]
        },
        5,
    )
    assert ranked == ["BC-001"]
    assert scores == [0.9]


def test_haystack_evaluator_uses_hayhooks_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"documents": []})

    case = EvalCase(
        case_id="case-1",
        base_id="case-1",
        category="test",
        difficulty="easy",
        question="What is the evidence?",
        expected_doc_ids=("BC-001",),
    )
    client = httpx.AsyncClient(
        base_url="http://hayhooks.test",
        transport=httpx.MockTransport(handler),
    )

    async def run() -> None:
        try:
            await request_once(client, "haystack", case, 5)
        finally:
            await client.aclose()

    asyncio.run(run())
    request = seen[0]
    assert request.url.path == "/avtr_haystack_query/run"
    assert json.loads(request.content) == {
        "query": "What is the evidence?",
        "initial_top_k": 10,
        "final_top_k": 5,
    }
