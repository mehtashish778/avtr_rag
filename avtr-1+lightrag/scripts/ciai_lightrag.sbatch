#!/bin/bash
#SBATCH -J avtr-lightrag
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJECT=/home/ashish.mehta/avtr_rag/avtr-1+lightrag
cd "$PROJECT"
mkdir -p logs data/lightrag .cache/pip
chmod 600 .env

export PIP_CACHE_DIR="$PROJECT/.cache/pip"
set -a
source .env
set +a

echo "host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Materialize the AVTR environments in this checkout. Model artifacts and the
# Hugging Face cache are shared with the already-working AVTR-1 installation.
pixi install

if [[ ! -x .rag-venv/bin/python ]]; then
  python3 -m venv .rag-venv
fi
source .rag-venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements-rag.txt

SERVICE_LOG="logs/lightrag-service_${SLURM_JOB_ID}.log"
SMOKE_LOG="logs/lightrag-smoke_${SLURM_JOB_ID}.log"
echo "LightRAG service log: $SERVICE_LOG"
lightrag-server > "$SERVICE_LOG" 2>&1 &
LIGHTRAG_PID=$!

cleanup() {
  kill "$LIGHTRAG_PID" 2>/dev/null || true
  wait "$LIGHTRAG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:9621/health >/dev/null; then
    echo "LightRAG health check passed"
    break
  fi
  if ! kill -0 "$LIGHTRAG_PID" 2>/dev/null; then
    echo "LightRAG exited before becoming healthy" >&2
    tail -n 100 "$SERVICE_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:9621/health >/dev/null

if [[ ! -f data/lightrag/.ingestion-complete ]]; then
  pixi run -e streamer python scripts/rag_admin.py ingest

  # Ingestion is asynchronous. Wait for every queued document to finish before
  # allowing the AVTR UI to become an evaluation target.
  for _ in $(seq 1 160); do
    if curl -fsS http://127.0.0.1:9621/documents/pipeline_status \
      | python -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if not d.get("busy") and not d.get("pending_enqueues", 0) else 1)'; then
      echo "LightRAG document pipeline is idle"
      break
    fi
    sleep 15
  done
  curl -fsS http://127.0.0.1:9621/documents/pipeline_status \
    | python -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if not d.get("busy") and not d.get("pending_enqueues", 0) else 1)'

  # Require non-empty retrieved context using a question answered by the
  # controlled breast-cancer corpus.
  for _ in $(seq 1 80); do
    SMOKE_OUTPUT=$(pixi run -e streamer python scripts/rag_admin.py smoke \
      --query "What breast changes should I report to a doctor?")
    printf '%s\n' "$SMOKE_OUTPUT" | tee "$SMOKE_LOG"
    if printf '%s\n' "$SMOKE_OUTPUT" \
      | python -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d.get("status") == "ok" and d.get("context_chars", 0) > 0 else 1)'; then
      touch data/lightrag/.ingestion-complete
      echo "LightRAG ingestion and retrieval smoke test passed"
      break
    fi
    sleep 15
  done
fi

test -f data/lightrag/.ingestion-complete
pixi run -e streamer interactive-demo
