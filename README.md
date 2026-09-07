# AVTR-RAG workspace

## About

This repository is part of the **AVATAR PROJECT POC** at the
**Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)**. It is the
canonical CIAI workspace for developing and comparing independent AVTR-1
retrieval-augmented generation integrations.

## Repository variants

This workspace compares four independent AVTR-1
integrations:

- `avtr-1+haystack`
- `avtr-1+lightrag`
- `avtr-1+txtai`
- `avtr-1+ragflow`

Shared evaluation inputs and comparison outputs live at the repository root.
Large models, service environments, indexes, caches, logs, and private `.env`
files are intentionally excluded from Git and stored on CIAI shared storage.

Open this repository through VS Code Remote SSH at:

```text
/home/ashish.mehta/avtr_rag
```

See `CIAI_WORKSPACE.md` for storage, migration, and execution details.
