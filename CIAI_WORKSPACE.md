# CIAI canonical workspace

## Source workspace

```text
/home/ashish.mehta/avtr_rag
```

This parent directory is the only source workspace to open in VS Code Remote
SSH. It is one Git repository containing the LightRAG, txtai, and RAGFlow
variants plus their shared evaluation materials.

## Runtime storage

Large and generated assets stay on Lustre:

```text
/l/users/ashish.mehta/avtr_rag-runtime
```

The workspace exposes those assets through ignored symbolic links. They include
Pixi environments, RAG virtual environments, model artifacts, Hugging Face
caches, vector indexes, service data, logs, and raw results.

## Legacy locations

The migration preserves the existing locations until the canonical workspace
passes GPU smoke tests:

```text
/l/users/ashish.mehta/avatar_project
/home/ashish.mehta/avatar_project
```

Do not delete either legacy location until their server-only content has been
verified and a separate deletion approval has been given.

## Execution rule

Use the CIAI login node only for Git, editing, file synchronization, and Slurm
submission. Run AVTR, CUDA setup, renderer tests, and end-to-end evaluation in
a Slurm allocation with an explicitly requested GPU.
