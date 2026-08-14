# Minimal UV Diffusion Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a repository-local CUDA environment that loads `lerobot/pusht` Dataset v3.0 and completes one Diffusion Policy training step.

**Architecture:** Reuse the repository's locked dependency graph and combine only the existing `training` and `diffusion` extras. Validate the environment from the lowest layer upward: imports, a real CUDA tensor operation, one decoded v3.0 dataset sample, and finally the normal `lerobot-train` entry point.

**Tech Stack:** uv, Python 3.12, PyTorch 2.11.0+cu128, Hugging Face Datasets/Hub, LeRobotDataset v3.0, Diffusers, Accelerate

---

### Task 1: Materialize The Locked Local Environment

**Files:**
- Create locally (Git-ignored): `.venv/`
- Verify unchanged: `pyproject.toml`
- Verify unchanged: `uv.lock`

- [ ] **Step 1: Verify the starting state**

Run:

```bash
git status --short
test ! -e .venv
uv --version
```

Expected: the committed design and plan are the only repository changes, `.venv` does not exist, and uv 0.11.21 or newer is available.

- [ ] **Step 2: Synchronize only training and Diffusion Policy dependencies**

Run:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync \
  --locked \
  --no-dev \
  --extra training \
  --extra diffusion
```

Expected: uv creates `.venv`, installs the editable `lerobot` workspace, and does not change `uv.lock`.

- [ ] **Step 3: Verify the local interpreter**

Run:

```bash
.venv/bin/python --version
.venv/bin/python -c "import lerobot; print(lerobot.__version__)"
git diff --exit-code -- pyproject.toml uv.lock
```

Expected: Python 3.12+, LeRobot 0.5.2, and no dependency-file diff.

### Task 2: Verify Training Imports And CUDA Execution

**Files:**
- Use: `.venv/`

- [ ] **Step 1: Import the exact runtime dependency surface**

Run:

```bash
.venv/bin/python -c "import accelerate, av, datasets, diffusers, torch, torchvision; print('imports_ok', torch.__version__, torchvision.__version__, diffusers.__version__, datasets.__version__, accelerate.__version__, av.__version__)"
```

Expected: all imports succeed and PyTorch reports a `+cu128` build.

- [ ] **Step 2: Execute a real CUDA operation**

Run outside the restricted device sandbox:

```bash
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); x=torch.arange(8, device='cuda'); assert x.sum().item() == 28; print('cuda_ok', torch.cuda.get_device_name(0), torch.version.cuda, x.device)"
```

Expected: `cuda_ok NVIDIA GeForce RTX 5090 12.8 cuda:0`.

### Task 3: Load And Decode LeRobot Dataset v3.0

**Files:**
- Download through the normal Hugging Face user cache: `lerobot/pusht`, revision `v3.0`

- [ ] **Step 1: Load only episode 0 and inspect one sample**

Run:

```bash
.venv/bin/python -c "from lerobot.datasets import LeRobotDataset; ds=LeRobotDataset('lerobot/pusht', episodes=[0]); assert ds.meta.info.codebase_version == 'v3.0'; sample=ds[0]; assert 'observation.state' in sample and 'action' in sample; camera=ds.meta.camera_keys[0]; assert camera in sample; print('dataset_ok', ds.meta.info.codebase_version, len(ds), camera, tuple(sample[camera].shape), tuple(sample['observation.state'].shape), tuple(sample['action'].shape))"
```

Expected: `dataset_ok v3.0 ...` followed by valid image, state, and action tensor shapes. This step must decode an actual camera frame, not only metadata.

### Task 4: Run One Diffusion Policy Optimizer Step

**Files:**
- Write transient output: `/tmp/lerobot-diffusion-pusht-smoke-<timestamp>/`

- [ ] **Step 1: Run the normal training CLI on episode 0**

Run:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --dataset.episodes='[0]' \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --num_workers=0 \
  --steps=1 \
  --log_freq=1 \
  --eval_freq=0 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir="/tmp/lerobot-diffusion-pusht-smoke-$(date +%s)"
```

Expected: the CLI reports the CUDA device, creates the Diffusion Policy and optimizer, loads one batch from episode 0, and exits after `Training: 100% 1/1` with finite loss and gradient norm.

- [ ] **Step 2: Confirm the environment remains local and source files remain clean**

Run:

```bash
du -sh .venv
git status --short
```

Expected: `.venv` exists under the repository and is ignored; no source, dependency, or lock file changed during installation and validation.
