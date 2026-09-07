"""Hayhooks wrapper for the native Haystack retrieval and ranking pipeline."""

import os

from hayhooks import BasePipelineWrapper


class PipelineWrapper(BasePipelineWrapper):
    """Retrieve and rerank original evidence from the Haystack Qdrant store."""

    skip_mcp = True

    def setup(self) -> None:
        from haystack import Pipeline
        from haystack.components.joiners import DocumentJoiner
        from haystack_integrations.components.embedders.sentence_transformers import (
            SentenceTransformersTextEmbedder,
        )
        from haystack_integrations.components.rankers.sentence_transformers import (
            SentenceTransformersSimilarityRanker,
        )
        from haystack_integrations.components.retrievers.qdrant import (
            QdrantEmbeddingRetriever,
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
        embedding_kwargs = {"revision": embedding_revision} if embedding_revision else None
        ranker_revision = os.environ.get(
            "HAYSTACK__RANKER_REVISION",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        )
        ranker_kwargs = {"revision": ranker_revision} if ranker_revision else None

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
            "text_embedder",
            SentenceTransformersTextEmbedder(
                model=embedding_model,
                model_kwargs=embedding_kwargs,
            ),
        )
        pipeline.add_component(
            "embedding_retriever",
            QdrantEmbeddingRetriever(document_store=document_store, top_k=10),
        )
        pipeline.add_component(
            "similarity_ranker",
            SentenceTransformersSimilarityRanker(
                model=os.environ.get(
                    "HAYSTACK__RANKER_MODEL",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
                top_k=3,
                model_kwargs=ranker_kwargs,
                tokenizer_kwargs=ranker_kwargs,
            ),
        )
        pipeline.add_component(
            "document_joiner",
            DocumentJoiner(join_mode="concatenate", top_k=3, sort_by_score=True),
        )
        pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever.documents", "similarity_ranker.documents")
        pipeline.connect("similarity_ranker.documents", "document_joiner.documents")
        self.pipeline = pipeline

    def run_api(
        self,
        query: str,
        initial_top_k: int = 10,
        final_top_k: int = 3,
    ) -> dict[str, object]:
        """Retrieve and rerank evidence without generating an answer."""
        result = self.pipeline.run(
            {
                "text_embedder": {"text": query},
                "embedding_retriever": {"top_k": initial_top_k},
                "similarity_ranker": {"query": query, "top_k": final_top_k},
                "document_joiner": {"top_k": final_top_k},
            }
        )
        documents = result.get("document_joiner", {}).get("documents", [])
        return {"documents": [document.to_dict() for document in documents]}
