#!/bin/bash
#SBATCH -J avtr-haystack
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -t 03:00:00
#SBATCH -o /l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/slurm/%x_%j.out
#SBATCH -e /l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/slurm/%x_%j.err

set -euo pipefail

PROJECT=/home/ashish.mehta/avtr_rag/avtr-1+haystack
RUNTIME_ROOT=/l/users/ashish.mehta/avtr_rag-runtime
RUN_LOG_DIR="$RUNTIME_ROOT/logs/haystack/$SLURM_JOB_ID"
cd "$PROJECT"
mkdir -p "$RUN_LOG_DIR"

test -f .env
chmod 600 .env
set -a
source .env
set +a

: "${QDRANT_SIF:?Set QDRANT_SIF to a pre-pulled Qdrant Singularity image}"
: "${QDRANT_STORAGE:?Set QDRANT_STORAGE to a persistent writable directory}"
mkdir -p "$QDRANT_STORAGE"

echo "host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

QDRANT_LOG="$RUN_LOG_DIR/qdrant.log"
HAYHOOKS_LOG="$RUN_LOG_DIR/hayhooks.log"

singularity run --bind "$QDRANT_STORAGE:/qdrant/storage" "$QDRANT_SIF" \
  > "$QDRANT_LOG" 2>&1 &
QDRANT_PID=$!

pixi run --frozen -e haystack hayhooks-service > "$HAYHOOKS_LOG" 2>&1 &
HAYHOOKS_PID=$!

cleanup() {
  kill "$HAYHOOKS_PID" "$QDRANT_PID" 2>/dev/null || true
  wait "$HAYHOOKS_PID" "$QDRANT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:6333/readyz >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:6333/readyz >/dev/null

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:1416/status >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:1416/status
curl -fsS http://127.0.0.1:1416/openapi.json >/dev/null

pixi run --frozen -e haystack haystack-ingest
pixi run --frozen -e haystack haystack-smoke
pixi run --frozen -e streamer interactive-demo
