# Minimal UV Diffusion Training Environment Design

## Goal

Create a reproducible CUDA-enabled environment in the repository-local `.venv` that can load a LeRobot Dataset v3.0 dataset and run Diffusion Policy training.

## Environment

- Use the repository's existing `pyproject.toml` and `uv.lock` without changing dependency declarations.
- Materialize the environment at `.venv` with `UV_PROJECT_ENVIRONMENT=.venv`.
- Sync only the `training` and `diffusion` optional dependency sets, excluding development dependencies.
- Keep the locked Linux CUDA stack: PyTorch 2.11.0 with CUDA 12.8 wheels.

The environment command is:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync \
  --locked \
  --no-dev \
  --extra training \
  --extra diffusion
```

## Validation

Validation has three layers:

1. Import the installed training, dataset, video, and diffusion packages.
2. Confirm that PyTorch can use the NVIDIA RTX 5090 through CUDA.
3. Load episode 0 from `lerobot/pusht`, confirm its dataset codebase version is v3.0, read a sample, and run one Diffusion Policy optimizer step on CUDA.

The training smoke test uses a small batch, episode 0 only, no evaluation, no WandB logging, and no checkpoint saving. This exercises the normal `lerobot-train` path while limiting download and runtime costs.

## Outputs

- `.venv/` contains the local environment and remains ignored by Git.
- Hugging Face data remains in the normal user cache rather than being committed to the repository.
- Smoke-test output is written under `/tmp` so verification does not leave training artifacts in the worktree.

## Failure Handling

- A locked dependency resolution failure is reported rather than bypassing `uv.lock`.
- A dataset version other than v3.0 fails validation.
- CUDA validation checks both `torch.cuda.is_available()` and a real tensor operation on the GPU.
- The environment is not considered usable until the one-step training command exits successfully.
