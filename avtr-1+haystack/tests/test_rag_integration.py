from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from avaturn_live_streamer.conversation_engines.rag_client import (
    RagClient,
    RagSettings,
    rag_prompt,
)


def query_with_payload(backend: str, payload: object) -> tuple[dict[str, object], httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    settings = RagSettings(
        backend=backend,
        base_url="http://rag.test",
        dataset_ids=["dataset-1"],
        top_k=3,
    )
    http = httpx.AsyncClient(
        base_url=settings.resolved_base_url,
        transport=httpx.MockTransport(handler),
    )
    client = RagClient(settings, http)

    async def run() -> dict[str, object]:
        try:
            return await client.query("What is the policy?")
        finally:
            await client.aclose()

    return asyncio.run(run()), seen[0]


def test_txtai_normalization_and_request() -> None:
    result, request = query_with_payload(
        "txtai",
        [{"id": "DOC-1", "text": "Approved context", "score": 0.9}],
    )

    assert result["status"] == "ok"
    assert result["backend"] == "txtai"
    assert "Approved context" in str(result["context"])
    assert request.url.path == "/search"
    assert request.url.params["limit"] == "3"


def test_lightrag_context_only_request() -> None:
    result, request = query_with_payload(
        "lightrag",
        {"response": "Retrieved LightRAG context", "references": [{"file": "one.txt"}]},
    )

    body = json.loads(request.content)
    assert result["status"] == "ok"
    assert body["only_need_context"] is True
    assert body["mode"] == "naive"
    assert body["chunk_top_k"] == 3
    assert body["enable_rerank"] is False


def test_ragflow_vector_only_request() -> None:
    result, request = query_with_payload(
        "ragflow",
        {
            "code": 0,
            "data": {
                "chunks": [
                    {
                        "document_keyword": "DOC-2",
                        "content_with_weight": "Retrieved RAGFlow context",
                        "similarity": 0.8,
                    }
                ]
            },
        },
    )

    body = json.loads(request.content)
    assert result["status"] == "ok"
    assert body["dataset_ids"] == ["dataset-1"]
    assert body["keyword"] is False
    assert body["use_kg"] is False
    assert body["vector_similarity_weight"] == 1.0


def test_haystack_request_shape_and_native_document_normalization() -> None:
    result, request = query_with_payload(
        "haystack",
        {
            "documents": [
                {
                    "id": "chunk-1",
                    "content": "Ranked original evidence",
                    "score": 0.93,
                    "document_id": "BC-001",
                    "file_name": "BC-001.txt",
                    "split_id": 0,
                }
            ]
        },
    )

    body = json.loads(request.content)
    assert request.url.path == "/avtr_haystack_query/run"
    assert body == {
        "query": "What is the policy?",
        "initial_top_k": 10,
        "final_top_k": 3,
    }
    assert result["status"] == "ok"
    assert result["backend"] == "haystack"
    assert "[Source BC-001]" in str(result["context"])
    assert "Ranked original evidence" in str(result["context"])
    references = result["references"]
    assert isinstance(references, list)
    assert references[0] == {
        "rank": 1,
        "document_id": "BC-001",
        "source": "BC-001",
        "score": 0.93,
        "chunk_id": "chunk-1",
    }


def test_haystack_empty_documents() -> None:
    result, _ = query_with_payload("haystack", {"documents": []})
    assert result["status"] == "empty"
    assert result["context"] == ""
    assert result["references"] == []


def test_haystack_malformed_documents() -> None:
    result, _ = query_with_payload("haystack", {"documents": "invalid"})
    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid_response"


def test_haystack_http_failure_degrades_safely() -> None:
    result, _ = query_with_payload("haystack", {"detail": "failure"})

    # A structurally valid object without documents is an empty retrieval.
    assert result["status"] == "empty"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    settings = RagSettings(backend="haystack", base_url="http://rag.test")
    http = httpx.AsyncClient(
        base_url=settings.resolved_base_url,
        transport=httpx.MockTransport(handler),
    )
    client = RagClient(settings, http)

    async def run() -> dict[str, object]:
        try:
            return await client.query("service failure")
        finally:
            await client.aclose()

    unavailable = asyncio.run(run())
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "http_503"


def test_haystack_timeout_degrades_safely() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    settings = RagSettings(backend="haystack", base_url="http://rag.test")
    http = httpx.AsyncClient(
        base_url=settings.resolved_base_url,
        transport=httpx.MockTransport(handler),
    )
    client = RagClient(settings, http)

    async def run() -> dict[str, object]:
        try:
            return await client.query("timeout")
        finally:
            await client.aclose()

    result = asyncio.run(run())
    assert result["status"] == "unavailable"
    assert result["reason"] == "timeout"


def test_haystack_token_budget_keeps_complete_evidence_blocks() -> None:
    payload = {
        "documents": [
            {
                "id": "chunk-1",
                "content": "short evidence",
                "score": 0.9,
                "document_id": "BC-001",
            },
            {
                "id": "chunk-2",
                "content": "word " * 500,
                "score": 0.8,
                "document_id": "BC-002",
            },
        ]
    }
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    settings = RagSettings(
        backend="haystack",
        base_url="http://rag.test",
        top_k=3,
        max_context_tokens=100,
    )
    http = httpx.AsyncClient(
        base_url=settings.resolved_base_url,
        transport=httpx.MockTransport(handler),
    )
    client = RagClient(settings, http)

    async def run() -> dict[str, object]:
        try:
            return await client.query("budget test")
        finally:
            await client.aclose()

    result = asyncio.run(run())
    assert "short evidence" in str(result["context"])
    assert "BC-002" not in str(result["context"])
    assert len(result["references"]) == 1


def test_prompt_rules_are_idempotent() -> None:
    prompt = rag_prompt("Be concise.")
    assert "search_knowledge_base" in prompt
    assert rag_prompt(prompt) == prompt


class FakeRagClient:
    async def query(self, query: str) -> dict[str, object]:
        return {
            "status": "ok",
            "backend": "txtai",
            "context": f"context for {query}",
            "references": [],
            "latency_ms": 1.0,
        }

    async def aclose(self) -> None:
        return None


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, message: str) -> None:
        self.messages.append(json.loads(message))


def test_tool_output_triggers_followup_response() -> None:
    from avaturn_live_streamer.conversation_engines.configs import (
        OpenAIRealtimeAPIConversationEngineConfig,
    )
    from avaturn_live_streamer.conversation_engines.realtime_api_client import (
        KnowledgeToolCall,
        RealtimeApiClient,
    )

    async def run() -> list[dict[str, object]]:
        websocket = FakeWebSocket()
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="test"),
            FakeRagClient(),  # type: ignore[arg-type]
        )
        await client._handle_tool_call(
            websocket,  # type: ignore[arg-type]
            KnowledgeToolCall("call-1", '{"query":"refund policy"}', 0),
        )
        return websocket.messages

    messages = asyncio.run(run())
    assert messages[0]["type"] == "conversation.item.create"
    assert messages[0]["item"]["call_id"] == "call-1"  # type: ignore[index]
    assert messages[1] == {"type": "response.create"}


def test_stale_tool_output_does_not_create_response() -> None:
    from avaturn_live_streamer.conversation_engines.configs import (
        OpenAIRealtimeAPIConversationEngineConfig,
    )
    from avaturn_live_streamer.conversation_engines.realtime_api_client import (
        KnowledgeToolCall,
        RealtimeApiClient,
    )

    async def run() -> list[dict[str, object]]:
        websocket = FakeWebSocket()
        client = RealtimeApiClient(
            OpenAIRealtimeAPIConversationEngineConfig(client_secret="test"),
            FakeRagClient(),  # type: ignore[arg-type]
        )
        client._turn_generation = 1
        await client._handle_tool_call(
            websocket,  # type: ignore[arg-type]
            KnowledgeToolCall("call-2", '{"query":"hours"}', 0),
        )
        return websocket.messages

    messages = asyncio.run(run())
    assert len(messages) == 1
    assert messages[0]["type"] == "conversation.item.create"


def test_audit_logger_writes_full_content_and_redacts_secrets(tmp_path: Path) -> None:
    from avaturn_live_streamer.conversation_engines.audit_log import (
        AuditLogSettings,
        RagAuditLogger,
    )

    async def run() -> Path:
        logger = RagAuditLogger(
            AuditLogSettings(directory=str(tmp_path), full_content=True),
            stream_id="stream/test",
            backend="lightrag",
        )
        await logger.start()
        logger.record(
            "turn_completed",
            turn_id="turn-000001",
            input_transcript="What does HER2-positive mean?",
            retrieval={"context": "Full retrieved evidence"},
            final_output="HER2 is a biomarker.",
            api_key="must-not-appear",
        )
        await logger.close()
        return logger.path

    path = asyncio.run(run())
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    turn = next(row for row in rows if row["event"] == "turn_completed")
    assert turn["input_transcript"] == "What does HER2-positive mean?"
    assert turn["retrieval"]["context"] == "Full retrieved evidence"
    assert turn["final_output"] == "HER2 is a biomarker."
    assert turn["api_key"] == "[REDACTED]"
