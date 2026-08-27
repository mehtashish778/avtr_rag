#!/bin/bash
#SBATCH -J avtr-migration-smoke
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH --exclude=gpu-24
#SBATCH -t 00:30:00
#SBATCH -o /l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/%x_%j.out
#SBATCH -e /l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/%x_%j.err

set -euo pipefail

ROOT=/home/ashish.mehta/avtr_rag
RUNTIME=/l/users/ashish.mehta/avtr_rag-runtime
LOG_ROOT="$RUNTIME/migration-smoke/logs"
LIGHTRAG="$ROOT/avtr-1+lightrag"
TXTAI="$ROOT/avtr-1+txtai"

mkdir -p "$LOG_ROOT"

LIGHTRAG_PID=""
ORCHESTRATOR_PID=""
TXTAI_PID=""

cleanup() {
  for pid in "$TXTAI_PID" "$ORCHESTRATOR_PID" "$LIGHTRAG_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url=$1
  local pid=$2
  local log_file=$3
  local retries=${4:-180}
  if curl -fsS --retry "$retries" --retry-delay 2 --retry-connrefused \
    --retry-all-errors --retry-max-time "$((retries * 2))" "$url" >/dev/null; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Process $pid exited before $url became healthy" >&2
  fi
  echo "Timed out waiting for $url" >&2
  tail -n 120 "$log_file" >&2 || true
  return 1
}

echo "MIGRATION_SMOKE_START"
echo "host=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

test -d "$ROOT/.git"
test -x "$LIGHTRAG/.rag-venv/bin/python"
test -x "$TXTAI/.rag-venv/bin/python"
test -d "$LIGHTRAG/.pixi"
test -d "$TXTAI/.pixi"
test -f "$LIGHTRAG/.env"
test -f "$TXTAI/.env"

cd "$LIGHTRAG"
set -a
source .env
set +a

export PYTHONPATH="$LIGHTRAG/src${PYTHONPATH:+:$PYTHONPATH}"
pixi run -e default python -c \
  'import pathlib, torch, avtr1_renderer; root=pathlib.Path("/home/ashish.mehta/avtr_rag/avtr-1+lightrag"); module=pathlib.Path(avtr1_renderer.__file__).resolve(); print(f"CUDA_OK={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} RENDERER_MODULE={module}"); assert torch.cuda.is_available(); assert module.is_relative_to(root)'
pixi run -e streamer python -c \
  'import pathlib, avaturn_live_streamer; root=pathlib.Path("/home/ashish.mehta/avtr_rag/avtr-1+lightrag"); module=pathlib.Path(avaturn_live_streamer.__file__).resolve(); print(f"STREAMER_MODULE={module}"); assert module.is_relative_to(root)'

LIGHTRAG_LOG="$LOG_ROOT/lightrag_${SLURM_JOB_ID}.log"
.rag-venv/bin/lightrag-server >"$LIGHTRAG_LOG" 2>&1 &
LIGHTRAG_PID=$!
wait_for_url http://127.0.0.1:9621/health "$LIGHTRAG_PID" "$LIGHTRAG_LOG" 120

LIGHTRAG_RESULT=$(pixi run -e streamer python scripts/rag_admin.py smoke \
  --query "What breast changes should I report to a doctor?")
printf 'LIGHTRAG_RESULT=%s\n' "$LIGHTRAG_RESULT"
printf '%s\n' "$LIGHTRAG_RESULT" | .rag-venv/bin/python -c \
  'import json,sys; d=json.load(sys.stdin); assert d.get("status") == "ok"; assert d.get("context_chars", 0) > 0; assert d.get("references", 0) > 0'

ORCHESTRATOR_LOG="$LOG_ROOT/avtr_orchestrator_${SLURM_JOB_ID}.log"
pixi run -e streamer python scripts/run_local_stream.py >"$ORCHESTRATOR_LOG" 2>&1 &
ORCHESTRATOR_PID=$!
wait_for_url http://127.0.0.1:8000/health "$ORCHESTRATOR_PID" "$ORCHESTRATOR_LOG" 180
wait_for_url http://127.0.0.1:7860/ "$ORCHESTRATOR_PID" "$ORCHESTRATOR_LOG" 30
echo "AVTR_RENDERER_AND_STREAMER_OK"
kill "$ORCHESTRATOR_PID" 2>/dev/null || true
wait "$ORCHESTRATOR_PID" 2>/dev/null || true
ORCHESTRATOR_PID=""
kill "$LIGHTRAG_PID" 2>/dev/null || true
wait "$LIGHTRAG_PID" 2>/dev/null || true
LIGHTRAG_PID=""

cd "$TXTAI"
set -a
source .env
set +a
export PYTHONPATH="$TXTAI/src${PYTHONPATH:+:$PYTHONPATH}"
export CONFIG="$TXTAI/service-configs/txtai-openai.yml"

pixi run -e streamer python -c \
  'import pathlib, avaturn_live_streamer; root=pathlib.Path("/home/ashish.mehta/avtr_rag/avtr-1+txtai"); module=pathlib.Path(avaturn_live_streamer.__file__).resolve(); print(f"TXTAI_STREAMER_MODULE={module}"); assert module.is_relative_to(root)'

TXTAI_LOG="$LOG_ROOT/txtai_${SLURM_JOB_ID}.log"
.rag-venv/bin/python -m uvicorn txtai.api:app \
  --host 127.0.0.1 --port 8001 --no-access-log >"$TXTAI_LOG" 2>&1 &
TXTAI_PID=$!
wait_for_url http://127.0.0.1:8001/openapi.json "$TXTAI_PID" "$TXTAI_LOG" 120

TXTAI_RESULT=$(pixi run -e streamer python scripts/rag_admin.py smoke \
  --query "What breast changes should I report to a doctor?")
printf 'TXTAI_RESULT=%s\n' "$TXTAI_RESULT"
printf '%s\n' "$TXTAI_RESULT" | .rag-venv/bin/python -c \
  'import json,sys; d=json.load(sys.stdin); assert d.get("status") == "ok"; assert d.get("context_chars", 0) > 0; assert d.get("references", 0) > 0'

kill "$TXTAI_PID" 2>/dev/null || true
wait "$TXTAI_PID" 2>/dev/null || true
TXTAI_PID=""

pixi run -e streamer python -c \
  'from avaturn_live_streamer.conversation_engines.rag_client import RagClient, RagSettings; cfg=RagSettings(enabled=True, backend="ragflow", base_url="http://127.0.0.1:9380"); RagClient(cfg); print("RAGFLOW_INTEGRATION_CODE_OK service_runtime=deferred")'

echo "MIGRATION_GPU_SMOKE_PASS"
