# DP3 34D/18D and PointNet Parameter Count

Date: 2026-08-27

## Contract change

The DP3 configuration now fixes `observation.state` to `(34,)` and `action` to
`(18,)`, and defaults `point_cloud_num_points` to `2048` so the complete input
cloud is processed. The point-cloud and dual-wrist RGB architecture is
unchanged.

## Point-cloud parameter count

The count was computed from the checked-in `PointNetEncoder` with
`point_cloud_num_points=2048`, point feature shape `(2048, 3)`, default hidden
dimensions `(64, 128, 256)`, LayerNorm enabled, and a 64-dimensional projection:

```text
pointnet_total_params 59072
```

This count is independent of the number of points because the encoder applies
the same per-point linear layers and global pooling to every point. Increasing
or lowering the default 2048 points changes runtime activation/memory cost but
adds no trainable parameters.

## Verification

```text
config_default_point_count 2048
helper_default_shape (2048, 3)
```

The direct default-value probe used the checked-in `DP3Config` and
`depth_to_point_cloud` implementations with a one-pixel valid depth image.

```text
UV_CACHE_DIR=.uv-cache uv run pytest tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py tests/processor/test_diffusion_processor.py -q
41 passed, 4 skipped, 1 warning

UV_CACHE_DIR=.uv-cache uv run ruff check src/lerobot/policies/dp3 tests/policies/test_dp3.py tests/test_franka_duo_real_recorder.py
All checks passed!

git diff --check
exit 0
```
