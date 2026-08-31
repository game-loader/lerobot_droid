# DP3 Dual-Wrist Validation Follow-up

Date: 2026-08-26

The earlier 40D state description is superseded by the current 34D model
contract. The state/action field ordering is supplied by the upstream adapter;
this model validates only the `(34,)` state and `(18,)` action shapes and keeps
pose conversion outside the policy.

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py tests/processor/test_diffusion_processor.py -q --disable-warnings --maxfail=1
40 passed, 4 skipped, 1 warning in 1.25s

UV_CACHE_DIR=.uv-cache uv run ruff check src/lerobot/policies/dp3 tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py
All checks passed!

git diff --check
exit 0
```

The added point-cloud regression confirms that a frame with no valid depth
points returns a fixed-size all-zero cloud, matching RL-100's padding path.
