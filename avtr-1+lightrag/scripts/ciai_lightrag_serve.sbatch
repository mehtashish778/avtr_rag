#!/bin/bash
#SBATCH -J avtr-lightrag
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH --qos=gpu-debug-qos
#SBATCH --exclude=gpu-24
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

SERVICE_LOG="logs/lightrag-service_${SLURM_JOB_ID}.log"
.rag-venv/bin/lightrag-server > "$SERVICE_LOG" 2>&1 &
LIGHTRAG_PID=$!

cleanup() {
  kill "$LIGHTRAG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The AVTR application remains in the foreground and owns the job lifetime.
# The LightRAG index and both Python environments are already materialized on
# shared Lustre storage by the one-time setup workflow.
pixi run -e streamer interactive-demo
