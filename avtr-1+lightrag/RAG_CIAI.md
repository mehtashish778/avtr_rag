# AVTR-1 + LightRAG on CIAI

This variant uses OpenAI Realtime for the voice conversation and LightRAG in
`naive`, context-only mode for the controlled comparison. Top-3 chunks are
requested and reranking is disabled.

## One-time setup

Run AVTR GPU setup inside a Slurm GPU allocation and keep environments/data on
shared Lustre storage:

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 -p long -t 03:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+lightrag
pixi install

python3 -m venv .rag-venv
source .rag-venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements-rag.txt
```

Create an untracked `.env` from `ciai.env.example`, protect it with
`chmod 600 .env`, and configure both LightRAG bindings:

```dotenv
HOST=127.0.0.1
PORT=9621
WORKING_DIR=/l/users/ashish.mehta/avtr_rag-runtime/lightrag/data/lightrag
INPUT_DIR=/home/ashish.mehta/avtr_rag/avtr-1+lightrag/knowledge_base/documents
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_API_KEY=<OPENAI_API_KEY>
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BINDING_API_KEY=<OPENAI_API_KEY>
RERANK_BINDING=null
```

These service credentials are separate from the per-session key entered in the
AVTR browser UI.

## Start and ingest

Inside the same GPU allocation:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+lightrag
mkdir -p logs data/lightrag
source .rag-venv/bin/activate
lightrag-server > logs/lightrag-service.log 2>&1 &
LIGHTRAG_PID=$!

pixi run -e streamer python scripts/rag_admin.py ingest
# LightRAG ingestion is asynchronous. Wait until all documents show processed.
pixi run -e streamer python scripts/rag_admin.py smoke \
  --query "Enter a question answered by the approved corpus"

pixi run -e streamer interactive-demo
kill "$LIGHTRAG_PID"
```

Tunnel AVTR port 7860 from the laptop as described in `CIAI_RUNBOOK.md`.

## Evaluation controls

Do not evaluate while LightRAG is indexing. Use the identical corpus, Top-3,
query set, embedding model, and concurrency in all three variants. Save both
service and AVTR logs. The structured audit trail below records full turn
content and timing for evaluation.

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
