#!/bin/bash
set -euo pipefail

PROJECT=/home/ashish.mehta/avtr_rag/avtr-1+lightrag
RUN_ID=${1:?usage: run_ciai_accuracy_test.sh RUN_ID}
OUTPUT_ROOT="$PROJECT/results/retrieval-accuracy/$RUN_ID"
SERVICE_LOG="$PROJECT/logs/lightrag-accuracy-service_${SLURM_JOB_ID:-manual}.log"

cd "$PROJECT"
mkdir -p "$OUTPUT_ROOT/lightrag_c1" "$OUTPUT_ROOT/lightrag_c8" logs
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

.rag-venv/bin/python scripts/rag_accuracy_test.py \
  --backend lightrag \
  --base-url http://127.0.0.1:9621 \
  --dataset knowledge_base/evaluation/breast_cancer_retrieval_extensive.jsonl \
  --top-k 5 \
  --repeats 3 \
  --concurrency 1 \
  --timeout-seconds 60 \
  --output-dir "$OUTPUT_ROOT/lightrag_c1"

.rag-venv/bin/python scripts/rag_accuracy_test.py \
  --backend lightrag \
  --base-url http://127.0.0.1:9621 \
  --dataset knowledge_base/evaluation/breast_cancer_retrieval_extensive.jsonl \
  --top-k 5 \
  --repeats 3 \
  --concurrency 8 \
  --timeout-seconds 30 \
  --output-dir "$OUTPUT_ROOT/lightrag_c8"
