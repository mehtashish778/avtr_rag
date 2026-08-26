# Knowledge-base data

Place the approved breast-cancer corpus in `documents/` as UTF-8 `.txt` or
`.md` files. Use the exact same files and filenames in all three AVTR
variants, then run:

```bash
pixi run -e streamer python scripts/rag_admin.py ingest
pixi run -e streamer python scripts/rag_admin.py smoke --query "your test question"
```

The workspace does not currently contain the breast-cancer corpus. The older
`Voice_RAG_Package_Eval/case_study` data is a fictional clinic-operations
corpus, so it is not copied here or mislabeled as breast-cancer data.

Only use reviewed sources suitable for the intended evaluation. This avatar
must not diagnose or replace a qualified clinician.
