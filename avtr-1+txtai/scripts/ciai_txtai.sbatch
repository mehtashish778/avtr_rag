#!/bin/bash
#SBATCH -J avtr-txtai
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

PROJECT=/home/ashish.mehta/avtr_rag/avtr-1+txtai
cd "$PROJECT"
mkdir -p logs data/txtai-openai-index .cache/pip
chmod 600 .env

export PIP_CACHE_DIR="$PROJECT/.cache/pip"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
set -a
source .env
set +a

echo "host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# The AVTR environment is shared with the working LightRAG checkout. txtai has
# its own isolated Python environment because its API dependencies are separate.
if [[ ! -x .rag-venv/bin/python ]]; then
  python3 -m venv .rag-venv
fi
source .rag-venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements-rag.txt

export CONFIG="$PROJECT/service-configs/txtai-openai.yml"
SERVICE_LOG="logs/txtai-service_${SLURM_JOB_ID}.log"
SMOKE_LOG="logs/txtai-smoke_${SLURM_JOB_ID}.log"
echo "txtai service log: $SERVICE_LOG"
python -m uvicorn "txtai.api:app" --host 127.0.0.1 --port 8001 --no-access-log \
  > "$SERVICE_LOG" 2>&1 &
TXTAI_PID=$!

cleanup() {
  kill "$TXTAI_PID" 2>/dev/null || true
  wait "$TXTAI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/openapi.json >/dev/null; then
    echo "txtai API health check passed"
    break
  fi
  if ! kill -0 "$TXTAI_PID" 2>/dev/null; then
    echo "txtai exited before becoming healthy" >&2
    tail -n 100 "$SERVICE_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8001/openapi.json >/dev/null

if [[ ! -f data/txtai-openai-index/.ingestion-complete ]]; then
  pixi run -e streamer python scripts/rag_admin.py ingest
  SMOKE_OUTPUT=$(pixi run -e streamer python scripts/rag_admin.py smoke \
    --query "What breast changes should I report to a doctor?")
  printf '%s\n' "$SMOKE_OUTPUT" | tee "$SMOKE_LOG"
  printf '%s\n' "$SMOKE_OUTPUT" \
    | python -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d.get("status") == "ok" and d.get("context_chars", 0) > 0 else 1)'
  touch data/txtai-openai-index/.ingestion-complete
  echo "txtai ingestion and retrieval smoke test passed"
fi

test -f data/txtai-openai-index/.ingestion-complete
pixi run -e streamer interactive-demo
