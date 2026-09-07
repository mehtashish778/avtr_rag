# AVTR-1 + Haystack architecture

This variant keeps OpenAI Realtime as the only answer and speech generator.
Hayhooks serves two native Haystack pipelines; Qdrant stores the embedded
evidence.

```mermaid
flowchart TB
    U([User voice]) --> RTC[RealtimeApiClient]
    RTC --> O[OpenAI Realtime]
    O -->|search_knowledge_base| RC[RagClient.query]
    RC -->|POST /avtr_haystack_query/run| HH[Hayhooks :1416]
    HH --> TE[SentenceTransformersTextEmbedder]
    TE --> QR[QdrantEmbeddingRetriever]
    QR --> RR[SentenceTransformersSimilarityRanker]
    RR --> DJ[DocumentJoiner]
    DJ -->|native Documents| RC
    RC -->|grounding context| O
    O --> AVTR[AVTR renderer]
    AVTR --> OUT([Avatar audio and video])

    DOCS[(Approved .txt/.md corpus)] --> ADMIN[rag_admin ingest]
    ADMIN -->|multipart files| IDX[Hayhooks index endpoint]
    IDX --> FTR[FileTypeRouter]
    FTR --> CONV[Native text/Markdown converters]
    CONV --> CLEAN[DocumentCleaner]
    CLEAN --> SPLIT[DocumentSplitter]
    SPLIT --> EMB[SentenceTransformersDocumentEmbedder]
    EMB --> WRITE[DocumentWriter]
    WRITE --> QD[(Qdrant)]
    QD --> QR
```

Hayhooks owns HTTP routing, validation, status, deployment discovery, and
OpenAPI generation. AVTR does not define a FastAPI service. Custom code is
limited to the two required `BasePipelineWrapper` classes and AVTR's existing
backend normalization boundary.

Generated endpoints:

- `GET /status`
- `POST /avtr_haystack_index/run`
- `POST /avtr_haystack_query/run`
- `GET /openapi.json`

Hayhooks 1.24 includes the typed request/response definitions in OpenAPI; it
does not create separate per-pipeline schema routes.
