# AVTR-1 + Haystack/Hayhooks on CIAI

Use Slurm for all CIAI compute work. Haystack ingestion and retrieval are
CPU-safe, but renderer startup and GPU evaluation require a GPU allocation
with `--gres=gpu:1`. Never start the renderer on a login node; inside a
one-GPU allocation the visible device is device `0`.

The source checkout is
`/home/ashish.mehta/avtr_rag/avtr-1+haystack`. Environments, model caches,
Qdrant storage, logs, indexes, and generated results belong under
`/l/users/ashish.mehta/avtr_rag-runtime` and must remain untracked.

## One-time setup

Perform installation from a compute allocation (a GPU allocation is required
for installation steps that inspect CUDA):

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:1 -p long -t 02:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+haystack
mkdir -p /l/users/ashish.mehta/avtr_rag-runtime/pixi/avtr-1+haystack
if [ ! -e .pixi ]; then
  ln -s /l/users/ashish.mehta/avtr_rag-runtime/pixi/avtr-1+haystack .pixi
fi
pixi install -e haystack
pixi install -e streamer
```

The ignored `.pixi` symlink keeps the project environments on Lustre. If a
real `.pixi` directory already exists, move or recreate it deliberately rather
than replacing it with the command above.

Create the untracked environment file without overwriting an existing one:

```bash
cp -n ciai.env.example .env
chmod 600 .env
```

Edit `.env` locally on CIAI and replace its placeholders. Never commit or paste
the resulting secrets. The project expects these persistent runtime paths:

```bash
mkdir -p /l/users/ashish.mehta/avtr_rag-runtime/qdrant
mkdir -p /l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/slurm
```

Set `QDRANT_SIF` to a pre-pulled Qdrant container image supported by the
cluster and `QDRANT_STORAGE` to the persistent runtime directory. Do not pull
containers repeatedly from a batch job.

## CPU-only retrieval service

Use a CPU allocation for ingestion, smoke tests, or retrieval without the
avatar. Confirm that your account can use `cscc-cpu-p` before submission:

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=32G \
  -p cscc-cpu-p -t 02:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+haystack
set -a
source .env
set +a

RUNTIME_LOG_DIR=/l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/manual
mkdir -p "$RUNTIME_LOG_DIR" "$QDRANT_STORAGE"
singularity run --bind "$QDRANT_STORAGE:/qdrant/storage" "$QDRANT_SIF" \
  >"$RUNTIME_LOG_DIR/qdrant.log" 2>&1 &
QDRANT_PID=$!
pixi run --frozen -e haystack hayhooks-service \
  >"$RUNTIME_LOG_DIR/hayhooks.log" 2>&1 &
HAYHOOKS_PID=$!

curl -fsS http://127.0.0.1:6333/readyz
curl -fsS http://127.0.0.1:1416/status
pixi run --frozen -e haystack haystack-ingest
pixi run --frozen -e haystack haystack-smoke
```

Stop both services before leaving the allocation:

```bash
kill "$HAYHOOKS_PID" "$QDRANT_PID"
wait "$HAYHOOKS_PID" "$QDRANT_PID" 2>/dev/null || true
```

## Full avatar: interactive

Request exactly one GPU, then start Qdrant and Hayhooks using the service
commands above before starting the renderer:

```bash
salloc -N 1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:1 -p long -t 03:00:00
cd /home/ashish.mehta/avtr_rag/avtr-1+haystack
set -a
source .env
set +a
nvidia-smi

# Start Qdrant and Hayhooks as shown in "CPU-only retrieval service", then:
pixi run --frozen -e haystack haystack-ingest
pixi run --frozen -e haystack haystack-smoke
pixi run --frozen -e streamer interactive-demo
```

Tunnel port `7860` and verify TURN using [CIAI_RUNBOOK.md](CIAI_RUNBOOK.md).
Keep the service PIDs from the startup shell so you can stop them cleanly.

## Full avatar: batch

The checked-in batch script requests one GPU and starts Qdrant, Hayhooks,
ingestion, a smoke query, and the interactive streamer in order. Create its
Slurm output directory before submitting because Slurm opens output files
before the script begins:

```bash
mkdir -p /l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/slurm
cd /home/ashish.mehta/avtr_rag/avtr-1+haystack
sbatch scripts/ciai_haystack.sbatch
```

Monitor without entering the job:

```bash
squeue -u "$USER"
tail -f /l/users/ashish.mehta/avtr_rag-runtime/logs/haystack/slurm/avtr-haystack_<JOB_ID>.out
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Verification and privacy

The CPU-safe integration suite does not start the renderer. The first real
pipeline startup may download the pinned embedding and ranker models:

```bash
pixi run --frozen -e haystack haystack-test
curl -fsS http://127.0.0.1:1416/openapi.json
```

The generated OpenAPI document must contain
`/avtr_haystack_index/run` and `/avtr_haystack_query/run`. Keep `.env`, API
keys, TURN credentials, medical conversations, patient-related data, audit
logs, and generated evaluation results out of Git.
