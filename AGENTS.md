# Project Agent Instructions

## Execution model

- The coding agent runs in the local checkout. Make source, dependency, lockfile, documentation, and test changes locally.
- Do not run local `uv sync`, local package installation, or local GPU/import validation. After committing and pushing, validate the environment directly on `h100-server`.
- Use `h100-server` as the GPU execution target after local development has been committed and pushed.
- Do not maintain a separate uncommitted source version on `h100-server`; the Git repository is the source of truth.

## Required delivery workflow

For every completed development change:

1. Inspect `git status` and `git diff`. Locally, only run cheap static checks needed to catch malformed edits; never run `uv sync`.
2. Stage only task-related files with `git add`. Never commit virtual environments, agent state, credentials, datasets, checkpoints, generated outputs, or caches.
3. Create a descriptive Git commit and push the current branch to `origin`.
4. Connect to `h100-server` using one reusable SSH/tmux terminal for the task.
5. In the matching repository checkout on `h100-server`, require a clean working tree, then run `git pull --ff-only origin <branch>`.
6. Run `uv sync --frozen` and execute the requested GPU validation, inference, training, or evaluation there.
7. If H100 validation fails because code must change, fix it locally and repeat the add/commit/push/pull cycle. Do not patch tracked source files directly on the server.

Stop instead of resetting or overwriting files when the local or remote checkout contains unexpected changes, has diverged history, or is on the wrong branch.

## Verification economy

- Minimize testing aggressively. Run only checks that can fail because of the files changed or that are explicitly required by the user.
- Do not run broad suites, duplicate smoke tests, repeated status probes, or "just in case" checks.
- Run each required check once. Do not rerun an unchanged passing check.
- On `h100-server`, use one `uv sync --frozen` followed by one focused command that verifies the requested path. Add further checks only after a concrete failure requires diagnosis.
- Run `bash tests/test.sh` only when the changed path requires the repository test suite; do not run it for documentation, dependency metadata, or workflow-only changes.

## Environment and commands

- Supported environment: Linux x86_64, Python 3.10, CUDA 12.1.
- Manage dependencies with `uv`; keep `pyproject.toml` and `uv.lock` synchronized.
- Reproducible H100 install: `uv sync --frozen`.
- Minimal dependency smoke test on H100:
  `uv run python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)"`
