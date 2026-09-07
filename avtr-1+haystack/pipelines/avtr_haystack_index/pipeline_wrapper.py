"""Hayhooks wrapper for the native Haystack indexing pipeline."""

import os
from pathlib import Path

from fastapi import UploadFile
from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    """Index uploaded text and Markdown files into the Haystack Qdrant store."""

    skip_mcp = True

    def setup(self) -> None:
        from haystack import Pipeline
        from haystack.components.converters import (
            MarkdownToDocument,
            TextFileToDocument,
        )
        from haystack.components.joiners import DocumentJoiner
        from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
        from haystack.components.routers import FileTypeRouter
        from haystack.components.writers import DocumentWriter
        from haystack.document_stores.types import DuplicatePolicy
        from haystack_integrations.components.embedders.sentence_transformers import (
            SentenceTransformersDocumentEmbedder,
        )
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

        embedding_model = os.environ.get(
            "HAYSTACK__EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        embedding_revision = os.environ.get(
            "HAYSTACK__EMBEDDING_REVISION",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        )
        model_kwargs = {"revision": embedding_revision} if embedding_revision else None

        document_store = QdrantDocumentStore(
            url=os.environ.get("HAYSTACK__QDRANT_URL", "http://127.0.0.1:6333"),
            index=os.environ.get(
                "HAYSTACK__QDRANT_COLLECTION",
                "avtr_breast_cancer_haystack",
            ),
            embedding_dim=int(os.environ.get("HAYSTACK__EMBEDDING_DIM", "384")),
            similarity=os.environ.get("HAYSTACK__QDRANT_SIMILARITY", "cosine"),
            recreate_index=False,
            return_embedding=False,
            wait_result_from_api=True,
        )

        pipeline = Pipeline()
        pipeline.add_component(
            "file_type_router",
            FileTypeRouter(mime_types=["text/plain", "text/markdown"]),
        )
        pipeline.add_component("text_file_converter", TextFileToDocument())
        pipeline.add_component("markdown_converter", MarkdownToDocument())
        pipeline.add_component(
            "document_joiner",
            DocumentJoiner(join_mode="concatenate", sort_by_score=False),
        )
        pipeline.add_component(
            "document_cleaner",
            DocumentCleaner(
                remove_empty_lines=True,
                remove_extra_whitespaces=True,
            ),
        )
        pipeline.add_component(
            "document_splitter",
            DocumentSplitter(
                split_by=os.environ.get("HAYSTACK__SPLIT_BY", "word"),
                split_length=int(os.environ.get("HAYSTACK__CHUNK_SIZE", "350")),
                split_overlap=int(os.environ.get("HAYSTACK__CHUNK_OVERLAP", "50")),
            ),
        )
        pipeline.add_component(
            "document_embedder",
            SentenceTransformersDocumentEmbedder(
                model=embedding_model,
                model_kwargs=model_kwargs,
            ),
        )
        pipeline.add_component(
            "document_writer",
            DocumentWriter(
                document_store=document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )

        pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
        pipeline.connect("text_file_converter.documents", "document_joiner.documents")
        pipeline.connect("markdown_converter.documents", "document_joiner.documents")
        pipeline.connect("document_joiner.documents", "document_cleaner.documents")
        pipeline.connect("document_cleaner.documents", "document_splitter.documents")
        pipeline.connect("document_splitter.documents", "document_embedder.documents")
        pipeline.connect("document_embedder.documents", "document_writer.documents")
        self.pipeline = pipeline

    def run_api(self, files: list[UploadFile] | None = None) -> dict[str, object]:
        """Index uploaded approved reference files with the native pipeline."""
        from haystack.dataclasses import ByteStream

        if not files:
            return {"files": [], "documents_written": 0}

        streams: list[ByteStream] = []
        filenames: list[str] = []
        for upload in files:
            if not upload.filename:
                continue
            filename = Path(upload.filename).name
            mime_type = upload.content_type
            if mime_type not in {"text/plain", "text/markdown"}:
                mime_type = "text/markdown" if filename.endswith(".md") else "text/plain"
            streams.append(
                ByteStream(
                    data=upload.file.read(),
                    mime_type=mime_type,
                    meta={"file_name": filename, "document_id": Path(filename).stem},
                )
            )
            filenames.append(filename)

        if not streams:
            return {"files": [], "documents_written": 0}

        result = self.pipeline.run({"file_type_router": {"sources": streams}})
        writer_result = result.get("document_writer", {})
        return {
            "files": filenames,
            "documents_written": writer_result.get("documents_written", 0),
        }
