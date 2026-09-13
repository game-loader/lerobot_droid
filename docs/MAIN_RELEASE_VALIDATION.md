# Main Release Validation

This records the validation for the `0.6.2+droid.2` release scope on
`release/dp3-rl-main`. It excludes the historical Franka ROS/TMR acquisition and
deployment tools and the dedicated SmolVLA server; those remain on
`feat/upstream-0.6-custom` and the original source branch.

Validation date: 2026-09-13.

## Executed Checks

- Consolidated custom-version regression with the IMF-AttnRes integration:
  **1,271 passed, 21 skipped, 2 deselected** in 63.74 seconds.
- The two deselected tests are gated OpenPI reference tests; both test files
  remain collected by the suite.
- `git diff --check` passed.

## Scope Covered

- Upstream LeRobot 0.6.2 behavior and RL-100 regressions.
- DP3 and state-only Diffusion policy paths.
- IMF-AttnRes policy, differential transformer, processor, SmolVLM encoder,
  training schedule, teacher updates, and checkpoint restoration.
- Dataset, processor, environment, distributed checkpoint, EMA, and training
  integration tests selected by `scripts/test_custom_version.sh`.

Known warnings are inherited multiprocessing/fork, single-process distributed
checkpoint fallback, EMA scheduler ordering, and the documented IMF DCT padding
mask behavior. This validation does not claim hardware, CUDA, simulator-asset,
multi-GPU, real-scene, or policy-quality coverage.

## Reproduce

```bash
bash scripts/test_custom_version.sh
uv lock --check
uv build --wheel
```
