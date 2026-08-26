# AVTR-1 on CIAI: setup, start, stop, and restart

This runbook records the working AVTR-1 setup on the MBZUAI CIAI cluster.
It assumes the project and model artifacts already exist at:

```text
Project:       /home/ashish.mehta/avtr_rag/avtr-1+txtai
Artifacts:     /l/users/ashish.mehta/avtr_rag-runtime/shared/avtr1-storage
Hugging Face:  /l/users/ashish.mehta/avtr_rag-runtime/shared/huggingface-cache
```

The GPU node name changes between allocations. Replace `<GPU_NODE>` and
`<JOB_ID>` below with the values Slurm gives you.

## Quick restart checklist

Use this section for normal future runs after the one-time setup is complete.

### 1. Connect to CIAI from the laptop

Connect to the MBZUAI VPN first when off campus. In PowerShell:

```powershell
ssh ciai
```

This opens a login-node shell. The login node does not have a GPU, so
`nvidia-smi` failing there is expected.

### 2. Start a three-hour A100 allocation

For this account, interactive GPU jobs use the debug QoS and are limited to
three hours:

```bash
salloc \
  -N 1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  -p long \
  -t 03:00:00
```

Wait for output similar to:

```text
salloc: Granted job allocation <JOB_ID>
salloc: Nodes <GPU_NODE> are ready for job
```

The prompt should change from `ciai-login-*` to a node such as `gpu-12`.
Confirm that this is a compute node:

```bash
hostname
nvidia-smi
```

Optional but recommended: start `tmux` on the login node before `salloc` so a
laptop disconnection does not immediately lose the interactive shell:

```bash
tmux new -s avtr1
```

Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t avtr1`.

### 3. Start AVTR-1 on the GPU node

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
chmod 600 .env
pixi run -e streamer interactive-demo
```

The Pixi manifest deprecation warning and ONNX Runtime
`pthread_setaffinity_np` messages are non-fatal on this Slurm setup. Wait for:

```text
renderer healthy
local-stream server starting host=0.0.0.0 port=7860
Application startup complete.
Uvicorn running on http://0.0.0.0:7860
```

Keep this terminal open. `Ctrl+C` stops the application but does not release
the parent allocation.

### 4. Create the laptop SSH tunnel

Open a second PowerShell window. Use the GPU node assigned in step 2:

```powershell
ssh -N -L 7860:<GPU_NODE>:7860 ciai
```

Example only:

```powershell
ssh -N -L 7860:gpu-12:7860 ciai
```

The tunnel command normally shows no output and stays running. Keep that
PowerShell window open, then visit:

```text
http://127.0.0.1:7860/
```

Enter the current OpenAI API key in the UI when requested. Never put the
OpenAI key in this document or commit it to Git.

### 5. Confirm Cloudflare TURN

After opening the UI, the server log should contain:

```text
ice: using Cloudflare TURN
```

From another shell inside the same allocation, this safe check verifies that
the application returns TURN URLs without printing short-lived credentials:

```bash
curl -s http://127.0.0.1:7860/ice-servers | grep -q turn.cloudflare.com
echo $?
```

Exit code `0` means TURN is present. In the browser, Connectivity should show:

```text
✓ relay via TURN
```

## Re-enter an existing allocation

From the login node, list your jobs:

```bash
squeue -u $USER -o "%.18i %.9T %.20R %.10L"
```

If the allocation is still running, enter it without requesting another GPU:

```bash
srun --jobid=<JOB_ID> --overlap --pty bash
```

Then start AVTR-1 using the command from the quick restart checklist.

To find the assigned GPU node for the laptop tunnel:

```bash
squeue -j <JOB_ID> -h -o "%N"
```

## Stop or restart safely

### Foreground application

In the terminal running AVTR-1, press `Ctrl+C`. Check that its ports are free:

```bash
fuser -n tcp 7860 8000
```

No output with exit code `1` means neither port is in use. Restart with:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
pixi run -e streamer interactive-demo
```

### Application launched as a separate Slurm step

List steps without canceling the allocation:

```bash
squeue -s -j <JOB_ID>
```

Cancel only the application step, for example:

```bash
scancel <JOB_ID>.<STEP_ID>
```

Do not run `scancel <JOB_ID>` unless you intend to release the A100 and end
the entire allocation.

### Finish and release the GPU

Stop AVTR-1 with `Ctrl+C`, then leave the compute shell:

```bash
exit
```

From the login node, confirm that the allocation ended:

```bash
squeue -j <JOB_ID>
```

If it is still running and you are completely finished, release it explicitly:

```bash
scancel <JOB_ID>
```

## One-time environment setup

The file below is stored only on CIAI and must never be committed:

```text
/home/ashish.mehta/avtr_rag/avtr-1+txtai/.env
```

The repository includes the secret-free [`ciai.env.example`](ciai.env.example)
template. On CIAI, create `.env` from it only when `.env` does not already
exist:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
cp -n ciai.env.example .env
nano .env
chmod 600 .env
```

`cp -n` protects an existing `.env` from being overwritten. Replace only the
TURN placeholders in the new file. The essential structure is:

```dotenv
STREAMER_HOST="0.0.0.0"
STREAMER_PORT="7860"
RENDERER_PORT="8000"

AVTR1_LOCAL_STORAGE="/l/users/ashish.mehta/avtr_rag-runtime/shared/avtr1-storage"
HF_HOME="/l/users/ashish.mehta/avtr_rag-runtime/shared/huggingface-cache"
HF_TOKEN="<HUGGINGFACE_TOKEN>"

CLOUDFLARE_TURN_KEY_ID="<TURN_KEY_ID>"
CLOUDFLARE_TURN_KEY_TOKEN="<TURN_API_TOKEN>"
```

The TURN Key ID and API token must come from the same active Cloudflare
Realtime TURN key. A `404` response containing `cannot find specified key`
means the ID is not an active TURN Key ID for that key/account.

`HF_TOKEN` must be a Hugging Face token with read access to the gated
`avaturn-live/avtr-1` repository. As an alternative to keeping it in `.env`,
authenticate once with `hf auth login` while `HF_HOME` points to the shared
CIAI cache, then remove or comment out `HF_TOKEN`. Never put the real token in
`ciai.env.example`.

The upstream repository code is sufficient for the verified setup; no source
change to `src/avaturn_live_streamer/localrtc/ice.py` is required.

## One-time install, download, and TensorRT build

These steps are already complete for the current CIAI project. Run them only
when rebuilding the environment or storage from scratch.

On the GPU node:

```bash
cd /home/ashish.mehta/avtr_rag/avtr-1+txtai
export AVTR1_LOCAL_STORAGE=/l/users/ashish.mehta/avtr_rag-runtime/shared/avtr1-storage
export HF_HOME=/l/users/ashish.mehta/avtr_rag-runtime/shared/huggingface-cache

pixi install
pixi run download
pixi run build-trt-engines
```

The Hugging Face account must have accepted access to the gated
`avaturn-live/avtr-1` repository. Authenticate when prompted. TensorRT engines
are GPU-architecture-specific; the existing engines were built for the CIAI
A100 (`sm80`) and can be reused across future A100 allocations through shared
storage.

## Troubleshooting

### `nvidia-smi: command not found`

You are on a login node. Request or enter a Slurm GPU allocation first.

### SSH tunnel says `connect failed: Connection refused`

Port `7860` is not listening on the selected GPU node, the application is
still starting, or the tunnel uses the wrong node name. Confirm:

```bash
squeue -j <JOB_ID> -h -o "%N"
fuser -n tcp 7860
```

Also confirm `.env` contains `STREAMER_HOST="0.0.0.0"`.

### `address already in use` on port 7860 or 8000

Another AVTR-1 process is still running. Find it without killing the parent
allocation:

```bash
fuser -n tcp 7860 8000
squeue -s -j <JOB_ID>
```

Stop the foreground process with `Ctrl+C` or cancel only its Slurm step.

### Hugging Face `401 Unauthorized` or gated-repository error

The new shell is not using the persistent cache/storage paths, or Hugging Face
authentication is missing. Confirm the three non-secret path variables in
`.env`, then authenticate the same `HF_HOME` if required. Do not paste the
token into chat or commit it.

### UI says no TURN is configured

1. Confirm both `CLOUDFLARE_TURN_KEY_*` names exist in `.env`.
2. Confirm the key ID and API token were created together for the same TURN key.
3. Restart AVTR-1; `.env` is loaded only when the process starts.
4. Run the safe `/ice-servers` check above.
5. Look for `ice: using Cloudflare TURN` in the streamer log.

### ICE/session timeout despite STUN success

STUN only proves the browser can discover its public address. CIAI cannot
usually receive the browser's direct UDP media through the SSH TCP tunnel, so
the session needs a successful TURN relay. Confirm the browser shows
`✓ relay via TURN`.

### ONNX Runtime thread-affinity errors

Messages such as `pthread_setaffinity_np failed` are caused by Slurm CPU
affinity. They were non-fatal in the verified run when both Uvicorn services
subsequently reported `Application startup complete`.

### Allocation ends after three hours

Interactive debug QoS jobs expire at their requested walltime. Files, model
artifacts, Pixi environments, and caches under `/l/users/ashish.mehta` persist;
request a new allocation and repeat the quick restart checklist.

## Security reminders

- Never commit `.env`.
- Never paste OpenAI, Hugging Face, Cloudflare, or TURN secrets into chat,
  screenshots, logs, or documentation.
- Use `chmod 600 .env`.
- Revoke and rotate any credential that was accidentally exposed.
- TURN short-lived credentials returned by `/ice-servers` should not be pasted
  into tickets or chat.
