# AVTR-1 + txtai architecture

This is the runtime and ingestion architecture without audit/logging steps.

```mermaid
flowchart TB
    U([User voice]) --> A1["LocalRTCWorklet<br/>_read_user_audio_loop()"]
    A1 --> A2["RealtimeApiClient<br/>_send_speech() / _listener()"]
    A2 --> O["OpenAI Realtime<br/>VAD, transcription and tool selection"]

    G1["rag_prompt() + KNOWLEDGE_TOOL<br/>registration"] -. session configuration .-> O

    O -->|search_knowledge_base call| G2["_enqueue_tool_calls()<br/>→ _tool_worker()"]
    G2 --> G3["_handle_tool_call()<br/>→ RagClient.query()"]
    G3 --> B1["txtai API<br/>GET /search"]
    B1 --> B2["txtai Embeddings.search()<br/>semantic similarity + Top-K"]
    B2 --> B3[("txtai vector index<br/>breast-cancer corpus")]
    B3 -->|ranked text and scores| G4["_normalize_txtai()<br/>context + references"]
    G4 --> G5["function_call_output<br/>+ response.create"]
    G5 --> O

    O -->|grounded response audio| A3["_decode_chunk()<br/>SegmentChunkGenerated"]
    O -->|general conversation without RAG| A3
    A3 --> A4["AVTR renderer pipeline<br/>speech-to-motion + video"]
    A4 --> OUT([Avatar audio and video])

    D[("Breast-cancer documents")] --> G6["rag_admin.ingest_txtai()"]
    G6 --> B4["txtai POST /add<br/>GET /index"]
    B4 --> B3

    classDef avtr fill:#f3e8ff,stroke:#7e22ce,color:#2e1065,stroke-width:3px;
    classDef rag fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:3px;
    classDef integration fill:#dcfce7,stroke:#16a34a,color:#052e16,stroke-width:3px;
    classDef external fill:#f3f4f6,stroke:#6b7280,color:#111827,stroke-width:2px;

    class A1,A2,A3,A4 avtr;
    class B1,B2,B3,B4 rag;
    class G1,G2,G3,G4,G5,G6 integration;
    class U,O,OUT,D external;
```

## Boundary legend

- Purple: original AVTR-1 functions.
- Blue: original txtai/RAG functions.
- Green: functions created to integrate AVTR-1 with txtai.
- Grey: external actors, hosted services, source data, and final output.

The integration boundary is implemented mainly in
`src/avaturn_live_streamer/conversation_engines/rag_client.py` and the RAG tool
handling added to `realtime_api_client.py`. The controlled retrieval endpoint is
`http://127.0.0.1:8001/search`.
