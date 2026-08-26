## Workspace

`/home/ashish.mehta/avtr_rag` is the canonical source workspace. Maintain all
three AVTR-RAG variants in the single parent Git repository. Do not initialize
nested repositories inside variant directories.

## Compute

Training, renderer startup, and GPU evaluation run on MBZUAI CIAI through
Slurm, never on the login node. Follow the personal `mbzuai-hpc` skill. Always
request GPUs with `--gres=gpu:N`; a one-GPU job uses device `0`.

Large environments, models, caches, indexes, logs, and generated results stay
under `/l/users/ashish.mehta/avtr_rag-runtime` and remain untracked.

Never commit `.env`, API keys, TURN credentials, medical conversation logs, or
patient-related data.
