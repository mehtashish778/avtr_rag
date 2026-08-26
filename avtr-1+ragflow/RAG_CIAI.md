# AVTR-1 + RAGFlow on CIAI

This variant uses OpenAI Realtime for voice and RAGFlow's retrieval API. The
controlled profile uses Top-3, vector weight 1.0, similarity threshold 0.0,
keyword search off, knowledge graph off, and no reranker.

## Deployment constraint

RAGFlow is a multi-service Docker Compose stack (RAGFlow, database, Redis, and
Elasticsearch). The CIAI site facts do not yet confirm Docker, Apptainer, or
Enroot support. Do not run Docker on the login node or invent a container
workflow. Confirm the supported runtime with the MBZUAI wiki/admins first.

Until that is confirmed, run the already validated official RAGFlow v0.27.0
CPU stack on a host reachable from the CIAI compute node, or ask HPC admins for
the approved service deployment pattern. Then set `RAG__BASE_URL` to that
reachable URL. `127.0.0.1:9380` is correct only when RAGFlow runs in the same
Slurm allocation.

## AVTR allocation

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 -p long -t 03:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+ragflow
pixi install
cp -n ciai.env.example .env
chmod 600 .env
```

Set these real values only in the untracked `.env`:

```dotenv
RAG__BASE_URL=http://<RAGFLOW_HOST>:9380
RAG__API_KEY=<RAGFLOW_API_KEY>
RAG__DATASET_IDS='["<RAGFLOW_DATASET_ID>"]'
```

The API key must belong to the account that owns the dataset.

## Ingest, verify, and run AVTR

After RAGFlow is healthy and the dataset exists:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+ragflow
pixi run -e streamer python scripts/rag_admin.py ingest
# RAGFlow parsing is asynchronous. Wait until every document is DONE.
pixi run -e streamer python scripts/rag_admin.py smoke \
  --query "Enter a question answered by the approved corpus"
pixi run -e streamer interactive-demo
```

Tunnel AVTR port 7860 from the laptop as described in `CIAI_RUNBOOK.md`.

## Evaluation controls

Use the identical corpus, Top-3, query set, embedding model, and concurrency in
all three variants. Keep RAGFlow service placement identical to the other
backends when comparing latency; otherwise label network latency separately.
Save both service and AVTR logs. The structured audit trail below records full
turn content and timing for evaluation.

## Detailed evaluation logs

With `AUDIT__ENABLED=true`, each browser session creates a private JSONL file
under `logs/rag-audit/`. Every completed turn records the voice transcript,
standalone knowledge-base query, full normalized retrieved context and
references, final spoken transcript, OpenAI usage, and stage-by-stage latency.
The directory and files are created with modes `700` and `600` on Linux.

Generate aggregate speed and deterministic retrieval/answer proxies with:

```bash
pixi run -e streamer python scripts/audit_report.py \
  --output logs/rag-audit/report.json
```

These logs contain potentially sensitive medical conversation text. Do not
commit, email, or place them on shared public storage.
