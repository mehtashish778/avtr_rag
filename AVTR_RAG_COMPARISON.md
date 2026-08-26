# AVTR-1 RAG integration status

The three independent AVTR-1 checkouts now use the same OpenAI Realtime tool
contract and differ only in their configured retrieval backend.

| Variant | Default endpoint | Retrieval profile |
|---|---|---|
| `avtr-1+txtai` | `http://127.0.0.1:8001` | dense Top-3, OpenAI `text-embedding-3-small` |
| `avtr-1+lightrag` | `http://127.0.0.1:9621` | `naive`, context-only Top-3, reranker off |
| `avtr-1+ragflow` | `http://127.0.0.1:9380` | vector Top-3, keyword/KG/reranker off |

## What is complete

- Realtime `search_knowledge_base` registration for minted and compatibility sessions.
- Non-blocking retrieval worker and stale-response suppression on interruption.
- Backend response normalization, context bounds, timeouts, graceful failures,
  references, and latency logging.
- Backend-neutral corpus ingestion and smoke utility.
- Secret-free environment templates, pinned lightweight service requirements,
  and backend-specific CIAI runbooks.
- Unit contract checks for request shapes and normalization.
- Live adapter smoke checks against the existing benchmark indexes:
  txtai, LightRAG, and RAGFlow all returned Top-3 context successfully.

The live smoke numbers are connectivity checks, not a performance result: the
services were in different warm/cold states and RAGFlow ran in Docker while the
other services were started separately.

## Data blocker

No breast-cancer corpus exists in the current Avatar workspace. The older
`Voice_RAG_Package_Eval/case_study` corpus contains fictional clinic
operations and must not be relabeled as breast-cancer data.

Place the same reviewed/de-identified `.txt` or `.md` files under each
variant's `knowledge_base/documents/`, or pass one shared directory to:

```bash
pixi run -e streamer python scripts/rag_admin.py ingest \
  --documents-dir /home/ashish.mehta/avtr_rag/avtr-1+lightrag/knowledge_base/documents
```

## CIAI sequence

1. Use the canonical workspace under `/home/ashish.mehta/avtr_rag/`.
2. Request one A100 with Slurm; never start AVTR GPU work on the login node.
3. Follow each variant's `RAG_CIAI.md`.
4. Ingest the identical corpus and wait for all asynchronous indexing to finish.
5. Run a warm smoke query before the first voice session.
6. Start `pixi run -e streamer interactive-demo` and tunnel port 7860.
7. Run the same question order at concurrency 1, 4, and 8; compare p95 retrieval
   latency, error rate, Hit@3, Recall@3, MRR, and end-to-end response latency.
8. Keep controlled vector retrieval separate from each backend's best-native
   configuration experiment.

RAGFlow on CIAI still needs an approved container/service deployment pattern;
the current site facts do not confirm Docker, Apptainer, or Enroot support.
