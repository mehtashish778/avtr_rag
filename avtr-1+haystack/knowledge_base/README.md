# Knowledge-base data

The repository includes a de-identified breast-cancer evaluation corpus under
`documents/breast_cancer/`. Its provenance and review date are recorded in
[`BREAST_CANCER_SOURCES.md`](BREAST_CANCER_SOURCES.md). Use the exact same files
and filenames in all three AVTR variants for comparable retrieval results.

Haystack accepts UTF-8 `.txt` and `.md` files. With Qdrant and Hayhooks already
running, index and query the corpus with:

```bash
pixi run -e haystack haystack-ingest
pixi run -e haystack python scripts/rag_admin.py smoke --query "your test question"
```

Pass `--documents-dir /path/to/approved/documents` to the `ingest` subcommand
to use another corpus. Do not mix the older fictional clinic-operations data
with the breast-cancer evaluation set.

Only use reviewed sources suitable for the intended evaluation. This avatar
must not diagnose or replace a qualified clinician.
