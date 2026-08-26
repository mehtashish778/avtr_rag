# CIAI migration manifest

- Migration ID: `20260826T185944`
- Canonical workspace: `/home/ashish.mehta/avtr_rag`
- Runtime root: `/l/users/ashish.mehta/avtr_rag-runtime`
- Source archive SHA-256:
  `c38bba4e8e76d65d0507bba783d50f0a1f828082e67c2cd63abf064086ae78e4`

## Source repositories consolidated

The three local variants were separate modified clones of
`https://github.com/avaturn-live/avtr-1.git` at upstream commit `eb1e5a8`.
Their nested `.git` directories were deliberately excluded so the canonical
workspace can be maintained as one parent repository.

- `avtr-1+lightrag`
- `avtr-1+txtai`
- `avtr-1+ragflow`

The original nested repositories remain available in the pre-migration local
workspace and legacy CIAI locations.

## CIAI-only source preserved

Files present only in the active CIAI copies were merged without overwriting
the transferred source. This includes the retrieval accuracy runner used by
the LightRAG and txtai evaluations:

- `avtr-1+lightrag/scripts/rag_accuracy_test.py`
- `avtr-1+txtai/scripts/rag_accuracy_test.py`

Existing LightRAG and txtai `.env` files were copied with permission mode
`600`. They are ignored by Git.

## Runtime compatibility

The new runtime tree links to the already-materialized environments and data
under `/l/users/ashish.mehta/avatar_project`. The canonical workspace links to
that runtime tree. This preserves the working indexes, model artifacts, caches,
logs, and results without duplicating gigabytes or changing legacy paths before
GPU validation.

## Validation state

- Source archive checksum verified after upload.
- No nested `.git` directory exists in the canonical source tree.
- Secret `.env` files are excluded from Git and retain mode `600`.
- Existing CIAI copies have not been deleted or renamed.
- Login-node validation can run without a GPU.
- Renderer and end-to-end smoke tests remain pending until a GPU is available.
