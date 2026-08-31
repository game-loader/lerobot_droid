# DP3 Dual-Wrist Validation

Date: 2026-08-26

## Real-robot scope

This model variant is prepared for real Franka Duo recordings from the local
ROS 2 recorder. ZED Mini RGB+depth supplies the point-cloud source through the
depth sidecar, and the left/right D405 streams supply the two wrist RGB inputs.
The recorder's raw 17D state/action fields are deliberately separate from the
model's 34D/18D contract; an explicit dataset and controller mapping remains
required.

## Implemented contract

- `observation.point_cloud`: fixed-size XYZ point cloud encoded by the RL-100-style shared MLP, global max pooling, and projection path.
- `observation.images.wrist_left` and `observation.images.wrist_right`: independent ResNet18 encoders with `weights=None`.
- `observation.state`: 34 dimensions.
- `action`: 18 dimensions.
- Sparse point clouds use RL-100-compatible zero padding; unrelated RGB features such as the head camera are excluded from DP3 conditioning.

## Verification

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py tests/processor/test_diffusion_processor.py -q --disable-warnings --maxfail=1
39 passed, 4 skipped, 1 warning in 1.33s

UV_CACHE_DIR=.uv-cache uv run ruff check src/lerobot/policies/dp3 tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py
All checks passed!

git diff --check
exit 0
```

The focused model test verifies configuration rejection for missing wrist RGB, wrong state/action dimensions, or pretrained weights; distinct RGB encoder instances; RGB-branch gradients; 18-dimensional inference output; and strict checkpoint round-trip.
