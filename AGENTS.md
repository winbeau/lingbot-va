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

## Attention extraction and density workflow

- Treat the current `robotwin_i2av` execution path and `lingbot-va-posttrain-robotwin` checkpoint as LingBot-VA 1.0-style velocity FlowMatch inference, not the LingBot-VA 2.0 distilled student. The VA2 student checkpoint and deployment sampler are not wired into this repository path.
- RoboTwin video inference performs 25 denoising updates at zero-based step indices `0..24`, then appends step index `25` at scheduler timestep `0` as a cache-only Transformer forward. Step `25` still executes VV attention but does not call `scheduler.step()` or update the video latent.
- Use `summary.json`, not the step index alone, as the source of truth for the actual shifted scheduler timestep.
- To extract six positions spanning the 25-step denoising trajectory (`0,4,9,14,19,24`) across all 30 layers and 24 heads on eight GPUs, run on `h100-server`:

  ```bash
  cd /root/wenbiao_zhao/lingbot-va && \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NGPU=8 CONFIG_NAME='robotwin_i2av' \
  uv run bash script/run_launch_va_server_sync.sh \
    --model-path /root/wenbiao_zhao/models/lingbot-va-posttrain-robotwin \
    --num-chunks-to-infer 10 \
    --save_root train_out-vv-10-6 \
    --attention-analysis \
    --attention-layers all \
    --attention-timesteps 0,4,9,14,19,24 \
    --attention-output attn-exp-vv-10-6
  ```

- To compute top-p=0.9 per-layer density for those six output directories, run:

  ```bash
  cd /root/wenbiao_zhao/lingbot-va && \
  INPUT_ROOT='attn-exp-vv-10-6/grab-the-medium-sized-white-mug-rotate-it-place-it-on-the-table-and-hook-it-onto-the-smooth-dark-gray-rack' && \
  OUTPUT_DIR='attn-density-vv-10-6' && \
  for step in 000 004 009 014 019 024; do
    uv run python script/compute_attention_density.py \
      --input-dir "$INPUT_ROOT/step_$step" \
      --output-dir "$OUTPUT_DIR"
  done && \
  ls -lh "$OUTPUT_DIR"
  ```

- The density script defaults to three quarters of available CPUs, capped by the 30 layer tasks. Each output is named after its input directory, for example `attn-density-vv-10-6/step_004.csv`.
