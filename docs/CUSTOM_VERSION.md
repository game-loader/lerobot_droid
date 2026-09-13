# LeRobot Droid 0.6.2+droid.2

## Lineage and Scope

- Upstream: Hugging Face LeRobot 0.6.2, commit `2774d9bddcbbda50e697e162e89e7eaada8d7105`.
- Custom source: `feat/franka-duo-real-recorder`, commit `088e1ba7`, including its
  working-tree SmolVLA RL/server, mixed-resolution Diffusion and point-encoder changes.
- Common ancestor: `bd9619dfc3fadd3647408537d89e82f83f770851`.
- Migration branch: `feat/upstream-0.6-custom`. The original checkout, local
  datasets, weights, running services and training outputs are left in place.
- `origin` remains the personal repository; `upstream` points to Hugging Face.
- Main release: `release/dp3-rl-main`, with IMF-AttnRes from
  `origin/feat/imf-attnres-smolvla-layerwise` at `4da33bb2`. Franka ROS/TMR
  acquisition, conversion and deployment tools, and the dedicated SmolVLA server
  are excluded. They remain in `feat/upstream-0.6-custom` and the old source branch.

This is a selective forward port, not a merge of old infrastructure over new
infrastructure. Upstream model families, native depth/Lance storage, distributed
training, EMA, validation splits and existing SAC/async inference remain present.
No claim is made that the migrated policies improve task success.

## Install

From this branch's checkout:

```bash
uv sync --locked --extra training --extra dp3 --extra imf-attnres-vlm --extra test --extra dev
uv run --no-sync lerobot-info
```

The package remains named `lerobot`, with local version `0.6.2+droid.2`; it replaces
the official package only in the environment where it is installed. Use a separate
virtual environment from the historical branch. The `RL` Python package is also
included in wheels, unlike the original source-only layout.

Optional extras: `dp3`, `imf-attnres`, `imf-attnres-vlm`, `rl100`, `smolvla-rl`, `moya_newton`.
Moya is intentionally not installed by the command above: it additionally needs
the private simulator submodule and site assets. Hardware drivers are outside
this release's scope.

## Custom Modules

| Module | Integration |
| --- | --- |
| DP3 | Native convention-based policy discovery; PointNet, optional wrist RGB, U-Net/Transformer denoising |
| Diffusion | State-only conditioning and per-camera resize before stacking; upstream batch inference and gradient checkpointing retained |
| IMF-AttnRes | MeanFlow/JVP action policy, AttnRes backbones, optional SmolVLM flat-token/layerwise conditioning |
| RL-100 | Separate `RL/` package; IQL, dynamics/AM-Q, masked DDIM/PPO, iterative IL and real-environment adapter |
| SmolVLA RL | Upstream native `DynamicCache` adapter, frozen shared conditioning, token-query critics and prefix dynamics |
| Moya Newton | Pinned submodule, fused vector environment, headless evaluation and sparse collection |
| Data cache | Opt-in raw RGB RAM/disk cache below transforms for local Parquet/video datasets |
| Training | Preserve checkpoint normalization during iterative IL, environment reuse, periodic evaluation JSON and checkpoint hashes |

See [RL migration](../RL/MIGRATION.md), [SmolVLA RL](../RL/smolvla/README.md),
[IMF-AttnRes](source/imf_attnres.mdx), and
[experimental encoders](EXPERIMENTAL_POINT_ENCODERS.md).

## Compatibility Boundaries

- Current DP3 is **34D state / 20D action**. Historical 18D DP3 checkpoints are
  not automatically remapped: changing an action head would change learned
  semantics. A separate explicit model/data migration is required.
- The generic RL real-environment callback adapter is retained; it is not a
  Franka driver. Provide a calibrated observation/control adapter before deployment.
- PointNet DP3 checkpoints with the current 34D/20D contract have tiny and actual
  production-checkpoint strict-load smokes. The tested production DP3 also matches
  the old branch exactly for a fixed-input, fixed-noise, one-step sample. An actual
  SmolVLA checkpoint strictly loads and predicts offline; real-scene behavior is not audited.
- New training commands use `--env_eval_freq`, not `--eval_freq`. Saved JSON
  training configs migrate the old field on load and reject conflicting values.
  The upstream 0.6 optimizer/distributed resume format is otherwise retained;
  use an old standalone `pretrained_model` bundle for warm-start, not a promise
  of exact cross-version optimizer/RNG continuation.
- `preserve_pretrained_processor_stats=true` requires a pretrained policy and
  is incompatible with `resume`, reward-model training or relative-action overrides.
  Ordinary resume keeps upstream's saved-stat behavior.
- Evaluation results are always saved. Checkpoint hashes are written only for
  a same-step standalone `model.safetensors` bundle with supported provenance
  fields. EMA evaluations do not receive a misleading live-weight hash; DCP-only
  directories are not treated as standalone policies.
- RL data adapters require local LeRobot v3 Parquet data. They do not implicitly
  acquire support for upstream Lance or bucket streaming.
- PTv3/Sonata files in the historical branch contain MLP placeholders rather
  than real backbones. The migration fails closed by default, rejects real
  pretrained weights in those placeholders, and disables misleading launchers.
  Production pretrained PTv3/Sonata support remains unimplemented.
- Private Moya simulator/assets licensing must be resolved before redistribution.
  The Apache-2.0 upstream license and RL attribution notice are retained.

## Validation

Run the migration-focused regression set from the new checkout:

```bash
uv run --no-sync pytest tests/rl100 tests/policies/test_dp3.py \
  tests/policies/test_diffusion_state_only.py tests/envs/test_moya_newton.py \
  tests/policies/imf_attnres tests/processor/test_imf_attnres_processor.py \
  tests/scripts/test_lerobot_eval_rendering.py \
  tests/configs/test_custom_train_migration.py -q
```

Validation results are recorded in `CUSTOM_VERSION_VALIDATION.md`. Hardware,
full-size checkpoints, multi-GPU execution and task-success measurements require
separate acceptance. Do not infer those from CPU software tests.

## Future Upstream Updates

Keep the original branch as a historical reference. On a new branch, fetch
`upstream/main`, integrate upstream changes, re-run custom and upstream regression
tests, then promote only the verified branch. Do not copy the old dataset reader,
policy factory or training script over their newer upstream versions.
