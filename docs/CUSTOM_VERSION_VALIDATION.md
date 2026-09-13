# Migration Validation (Historical Full Migration)

This records `0.6.2+droid.1` before the release scope was narrowed. Franka-specific
tools/tests and the dedicated server were subsequently excluded, and IMF-AttnRes
was added. See `MAIN_RELEASE_VALIDATION.md` for the current main release results.

Validation date: 2026-09-09. Source and target refs are recorded in
[CUSTOM_VERSION.md](CUSTOM_VERSION.md). Tests below use a separate target `.venv`:
Python 3.12.13, Torch 2.11.0+cu128, Transformers 5.5.4, Draccus 0.11.6,
Datasets 4.8.5, Diffusers 0.39.0.

## Executed Checks

- Final consolidated custom and upstream regression (`bash scripts/test_custom_version.sh`):
  **1,246 passed, 21 skipped, 2 deselected, 20 subtests passed** in 59.50 seconds.
  Only the two gated OpenPI reference tests are deselected; both files are collected.
- Broader initial upstream run: **645 passed, 14 skipped, 2 failed**. Both failures
  were OpenPI reference tests requiring gated `google/paligemma-3b-pt-224`, returning
  HTTP 401. Those tests were not changed to conceal the failure.
- Base dependency tier: **174 passed, 29 skipped, 20 subtests passed**. Tests that
  require absent optional dependencies skip at collection; core algorithm tests run.
- Camera cache: actual shared H264 MP4, selected episode offsets, temporal padding,
  raw uint8/float output, transforms, disk corruption/identity checks, atomic build,
  external cache root, spawn DataLoader, train/eval split and unsupported readers.
- Training: actual tiny CPU Diffusion updates/checkpoint writes; EMA, gradient
  accumulation, saved stats preservation/overwrite, reusable/fresh evaluation
  environments, exception cleanup, zero-video logging and matching checkpoint hashes.
- RL: native SmolVLA DynamicCache, repeated denoising, actor gradients, frozen shared
  encoder, IQL/dynamics/PPO/AM-Q paths, checkpoint validation and iterative workflow.
- `uv lock --check`, wheel build, out-of-repository wheel import of RL/DP3 passed.
  Wheel contains the RL package and attribution notice, not datasets/model weights.
- Scoped Ruff checks pass excluding inherited docstring rules; original RL code
  has historical missing-docstring issues. No claim of a clean whole-upstream lint
  run is made. Migrated code formatting and `git diff --check` are checked separately.

## Real Checkpoint Compatibility

Read-only offline CPU smokes used existing source-checkout artifacts:

| Checkpoint | Result |
| --- | --- |
| `outputs/franka_duo_dp3_action20_pc_only_dit_il_50k/train/checkpoints/050000/pretrained_model` | Strict load; 19,543,828 parameters; finite action `[1,20]` |
| `outputs/train/franka_duo_smolvla_vlm_only_512_h64_a32_50k/checkpoints/050000/pretrained_model` | Strict load; 450,046,176 parameters; finite chunk `[1,64,20]`; cached VLM only |

DP3 source/target comparison used synthetic 34D state, 2,048 XYZ points, identical
seeded noise and one diffusion step: `torch.equal=True`, maximum absolute difference
`0.0`. SmolVLA used synthetic three-camera observations and one flow step. These are
interface/numerical smokes, not real-scene behavior or rollout-success tests. No
models were downloaded, checkpoint files edited, training jobs or servers launched.

## Reproduce

```bash
uv sync --locked --extra training --extra dp3 --extra imf-attnres-vlm --extra test --extra dev
bash scripts/test_custom_version.sh
uv lock --check
uv build --wheel
```

The consolidated script excludes only the two gated reference tests, retaining
the rest of both PI processor files. CUDA/real-data opt-in tests stay skipped unless
their explicit environment variables are supplied. Hardware, full simulator assets,
multi-GPU execution, exact legacy optimizer resume and policy quality remain outside
this migration's acceptance scope. Known inherited warnings include multiprocessing
`fork`, single-process distributed-checkpoint fallback and EMA scheduler ordering.
