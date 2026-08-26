#!/bin/bash
set -euo pipefail

PROJECT=/home/ashish.mehta/avtr_rag/avtr-1+lightrag
RUN_ID=${1:?usage: run_ciai_scale_test.sh RUN_ID}
OUTPUT_DIR="$PROJECT/results/scalability/$RUN_ID/lightrag"
SERVICE_LOG="$PROJECT/logs/lightrag-scale-service_${SLURM_JOB_ID:-manual}.log"

cd "$PROJECT"
mkdir -p "$OUTPUT_DIR" logs
set -a
source .env
set +a

.rag-venv/bin/lightrag-server > "$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!

cleanup() {
  kill "$SERVICE_PID" 2>/dev/null || true
  wait "$SERVICE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:9621/health >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    tail -n 100 "$SERVICE_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:9621/health >/dev/null

.rag-venv/bin/python scripts/rag_scale_test.py \
  --backend lightrag \
  --base-url http://127.0.0.1:9621 \
  --questions knowledge_base/evaluation/breast_cancer_questions.jsonl \
  --concurrency 1,4,8,16,32 \
  --requests-per-level 120 \
  --warmup 8 \
  --top-k 3 \
  --timeout-seconds 30 \
  --service-match lightrag-server \
  --output-dir "$OUTPUT_DIR"
