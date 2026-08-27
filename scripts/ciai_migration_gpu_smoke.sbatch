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
LIGHTRAG="$ROOT/avtr-1+lightrag"

cd "$LIGHTRAG"
export PYTHONPATH="$LIGHTRAG/src${PYTHONPATH:+:$PYTHONPATH}"
pixi run -e streamer python "$ROOT/scripts/ciai_migration_gpu_smoke.py"
