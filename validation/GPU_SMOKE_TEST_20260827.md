# CIAI migration GPU smoke test — 2026-08-27

## Execution

- Slurm job: `183630`
- Partition: `long`
- QOS: `gpu-debug-qos`
- Node: `gpu-13`
- Allocation: one NVIDIA A100-SXM4-40GB GPU
- Canonical workspace: `/home/ashish.mehta/avtr_rag`
- Runtime root: `/l/users/ashish.mehta/avtr_rag-runtime`

The job ran all checks through the bounded foreground driver
`scripts/ciai_migration_gpu_smoke.py`. The final marker was:

```text
MIGRATION_GPU_SMOKE_PASS
```

## Results

| Check | Result |
|---|---|
| CUDA available through PyTorch | Pass |
| AVTR renderer imported from canonical workspace | Pass |
| AVTR streamer imported from canonical LightRAG workspace | Pass |
| LightRAG service startup and retrieval | Pass |
| AVTR renderer `/health` | HTTP 200 |
| AVTR local-stream UI `/` | HTTP 200 |
| AVTR streamer imported from canonical txtai workspace | Pass |
| txtai service startup and retrieval | Pass |
| RAGFlow integration code construction | Pass; service runtime deferred |

### LightRAG retrieval

```json
{
  "backend": "lightrag",
  "context_chars": 3712,
  "latency_ms": 3434.34,
  "reason": null,
  "references": 3,
  "status": "ok"
}
```

### txtai retrieval

```json
{
  "backend": "txtai",
  "context_chars": 3220,
  "latency_ms": 2278.72,
  "reason": null,
  "references": 3,
  "status": "ok"
}
```

## Logs

```text
/l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/avtr-migration-smoke_183630.out
/l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/avtr-migration-smoke_183630.err
/l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/avtr_orchestrator_183630.log
/l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/lightrag_183630.log
/l/users/ashish.mehta/avtr_rag-runtime/migration-smoke/logs/txtai_183630.log
```

## Non-blocking warnings

- Pixi reports that `[system-requirements]` is deprecated in favor of platform
  virtual packages.
- Pixi redirects its network-filesystem repodata cache to the Slurm temporary
  directory. This is expected and does not affect correctness.
- ONNX Runtime emitted CPU thread-affinity warnings under Slurm. Renderer and
  streamer health checks still passed.
- The orchestrator reports a terminated streamer during cleanup because the
  smoke driver intentionally stops the process group after both HTTP checks
  pass.
- `sacct` was temporarily unavailable because the Slurm accounting database
  connection was refused. Job completion was confirmed by disappearance from
  `squeue` and the final pass marker in stdout.

## Verdict

The migrated canonical workspace is GPU-operational for AVTR-1 + LightRAG and
AVTR-1 + txtai. RAGFlow still requires an approved CIAI service/container
runtime before its live backend can be tested.
