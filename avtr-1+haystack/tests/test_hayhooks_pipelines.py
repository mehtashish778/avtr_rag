from __future__ import annotations

from inspect import signature
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from pipelines.avtr_haystack_index.pipeline_wrapper import (
    PipelineWrapper as IndexPipelineWrapper,
)
from pipelines.avtr_haystack_query.pipeline_wrapper import (
    PipelineWrapper as QueryPipelineWrapper,
)


class FakePipeline:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.inputs: list[dict[str, Any]] = []

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(data)
        return self.result


def test_wrappers_skip_mcp_registration() -> None:
    assert IndexPipelineWrapper.skip_mcp is True
    assert QueryPipelineWrapper.skip_mcp is True


def test_run_api_annotations_are_concrete_for_hayhooks_openapi() -> None:
    for wrapper in (IndexPipelineWrapper, QueryPipelineWrapper):
        run_signature = signature(wrapper.run_api)
        assert not isinstance(run_signature.return_annotation, str)
        assert all(
            not isinstance(parameter.annotation, str)
            for parameter in run_signature.parameters.values()
        )


def test_index_wrapper_uses_native_haystack_components(monkeypatch: Any) -> None:
    monkeypatch.setenv("HAYSTACK__QDRANT_URL", ":memory:")
    wrapper = IndexPipelineWrapper()
    wrapper.setup()

    expected = {
        "file_type_router": "FileTypeRouter",
        "text_file_converter": "TextFileToDocument",
        "markdown_converter": "MarkdownToDocument",
        "document_joiner": "DocumentJoiner",
        "document_cleaner": "DocumentCleaner",
        "document_splitter": "DocumentSplitter",
        "document_embedder": "SentenceTransformersDocumentEmbedder",
        "document_writer": "DocumentWriter",
    }
    for name, class_name in expected.items():
        assert type(wrapper.pipeline.get_component(name)).__name__ == class_name
    writer = wrapper.pipeline.get_component("document_writer")
    assert writer.policy is DuplicatePolicy.OVERWRITE
    assert wrapper.pipeline.to_dict()["connections"] == [
        {
            "sender": "file_type_router.text/plain",
            "receiver": "text_file_converter.sources",
        },
        {
            "sender": "file_type_router.text/markdown",
            "receiver": "markdown_converter.sources",
        },
        {
            "sender": "text_file_converter.documents",
            "receiver": "document_joiner.documents",
        },
        {
            "sender": "markdown_converter.documents",
            "receiver": "document_joiner.documents",
        },
        {
            "sender": "document_joiner.documents",
            "receiver": "document_cleaner.documents",
        },
        {
            "sender": "document_cleaner.documents",
            "receiver": "document_splitter.documents",
        },
        {
            "sender": "document_splitter.documents",
            "receiver": "document_embedder.documents",
        },
        {
            "sender": "document_embedder.documents",
            "receiver": "document_writer.documents",
        },
    ]


def test_query_wrapper_uses_native_haystack_components(monkeypatch: Any) -> None:
    monkeypatch.setenv("HAYSTACK__QDRANT_URL", ":memory:")
    wrapper = QueryPipelineWrapper()
    wrapper.setup()

    expected = {
        "text_embedder": "SentenceTransformersTextEmbedder",
        "embedding_retriever": "QdrantEmbeddingRetriever",
        "similarity_ranker": "SentenceTransformersSimilarityRanker",
        "document_joiner": "DocumentJoiner",
    }
    for name, class_name in expected.items():
        assert type(wrapper.pipeline.get_component(name)).__name__ == class_name
    assert wrapper.pipeline.to_dict()["connections"] == [
        {
            "sender": "text_embedder.embedding",
            "receiver": "embedding_retriever.query_embedding",
        },
        {
            "sender": "embedding_retriever.documents",
            "receiver": "similarity_ranker.documents",
        },
        {
            "sender": "similarity_ranker.documents",
            "receiver": "document_joiner.documents",
        },
    ]

    joiner = wrapper.pipeline.get_component("document_joiner")
    duplicate = Document(id="same", content="evidence", score=0.8)
    assert joiner.run(documents=[[duplicate], [duplicate]])["documents"] == [duplicate]


def test_index_wrapper_preserves_upload_metadata() -> None:
    wrapper = IndexPipelineWrapper()
    fake = FakePipeline({"document_writer": {"documents_written": 2}})
    wrapper.pipeline = cast(Any, fake)
    upload = SimpleNamespace(
        filename="BC-001.txt",
        content_type="text/plain",
        file=BytesIO(b"approved evidence"),
    )

    result = wrapper.run_api(cast(Any, [upload]))

    assert result == {"files": ["BC-001.txt"], "documents_written": 2}
    stream = fake.inputs[0]["file_type_router"]["sources"][0]
    assert stream.meta == {"file_name": "BC-001.txt", "document_id": "BC-001"}
    assert stream.data == b"approved evidence"


def test_query_wrapper_uses_runtime_top_k_and_native_document_dicts() -> None:
    document = Document(
        id="chunk-1",
        content="ranked evidence",
        meta={"document_id": "BC-001"},
        score=0.91,
    )
    wrapper = QueryPipelineWrapper()
    fake = FakePipeline({"document_joiner": {"documents": [document]}})
    wrapper.pipeline = cast(Any, fake)

    result = wrapper.run_api("treatment factors", initial_top_k=12, final_top_k=2)

    assert fake.inputs == [
        {
            "text_embedder": {"text": "treatment factors"},
            "embedding_retriever": {"top_k": 12},
            "similarity_ranker": {"query": "treatment factors", "top_k": 2},
            "document_joiner": {"top_k": 2},
        }
    ]
    rows = result["documents"]
    assert isinstance(rows, list)
    assert rows[0]["id"] == "chunk-1"
    assert rows[0]["document_id"] == "BC-001"
    assert rows[0]["score"] == 0.91
