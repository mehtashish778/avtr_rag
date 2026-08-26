# AVTR-1 + txtai on CIAI

This variant uses OpenAI Realtime for the voice conversation and txtai only for
retrieval. The retrieval profile matches the earlier controlled benchmark:
`text-embedding-3-small`, Top-3, dense vector search, and no reranker.

## One-time setup

Run GPU-dependent AVTR setup only inside a Slurm GPU allocation. Use shared
Lustre storage:

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 -p long -t 03:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
pixi install

python3 -m venv .rag-venv
source .rag-venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements-rag.txt
```

Create the untracked `.env` from `ciai.env.example`, add the RAG settings,
and protect it with `chmod 600 .env`. The txtai process also needs
`OPENAI_API_KEY`; this is separate from the per-session key entered in the
AVTR browser UI.

## Start and ingest

Inside the same GPU allocation:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
mkdir -p logs
source .rag-venv/bin/activate
export CONFIG="$PWD/service-configs/txtai-openai.yml"
python -m uvicorn "txtai.api:app" --host 127.0.0.1 --port 8001 --no-access-log \
  > logs/txtai-service.log 2>&1 &
TXTAI_PID=$!

pixi run -e streamer python scripts/rag_admin.py ingest
pixi run -e streamer python scripts/rag_admin.py smoke \
  --query "Enter a question answered by the approved corpus"

pixi run -e streamer interactive-demo
kill "$TXTAI_PID"
```

Keep txtai on loopback; AVTR talks to it on the same compute node. Tunnel AVTR
port 7860 from the laptop as described in `CIAI_RUNBOOK.md`.

## Evaluation controls

Use the identical corpus, Top-3, query set, embedding model, and concurrency in
all three variants. Save `logs/txtai-service.log` and the AVTR log. AVTR emits
one `RAG query completed` event per tool call with backend, status, and
retrieval latency. The structured audit trail below records full turn content.

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
