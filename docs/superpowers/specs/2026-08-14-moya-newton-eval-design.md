# Moya Newton LeRobot Evaluation Integration Design

## Goal

Add the Moya Newton simulator as a pinned Git submodule and make its fused CUDA
charger-grasp environment available through the standard `lerobot-eval` path in
the repository-local uv environment.

The first supported task is state-only Diffusion Policy evaluation against the
100-episode randomized charger-grasp dataset contract collected on 2026-08-13.
Evaluation is headless and does not record videos.

## Scope

This integration will:

- add the Moya repository at `third_party/moya_newton_sim` as a Git submodule;
- add an opt-in `moya_newton` dependency extra to the current uv project;
- register `--env.type=moya_newton` with LeRobot;
- adapt Moya's existing fused `gym.vector.VectorEnv` to LeRobot's evaluation
  contract without wrapping it in `SyncVectorEnv` or `AsyncVectorEnv`;
- reproduce and validate the environment settings used to collect the local
  LeRobot v3 dataset;
- report the agreed dataset-style success metric through LeRobot's
  `is_success` and `pc_success` fields; and
- run CPU, CUDA, and state-only Diffusion Policy smoke checks.

This integration will not add image observations, rendering, a Newton viewer,
per-world CUDA-native policy actions, a separate process bridge, or an upstream
Python packaging refactor for Moya.

## Submodule

The submodule will use:

```text
path:   third_party/moya_newton_sim
remote: git@gitlab.com:game-loader/moya_newton.git
commit: 7dd6af2952e1153d8943bc7b255e877b96ed0e55
```

Only tracked repository content is included. The source checkout's ignored
`runs/` data and untracked `imgui.ini` are not copied into the submodule.

The Moya repository and its robot/charger assets have no tracked `LICENSE`,
`COPYING`, or `NOTICE` file at the pinned revision. This does not block internal
technical integration, but the submodule must not be assumed redistributable
until its license is clarified.

The SSH remote requires GitLab credentials wherever submodules are initialized,
including CI.

## Dependencies

Add an optional dependency group named `moya_newton`:

```toml
moya_newton = [
  "newton[sim,importers]==1.3.0",
  "warp-lang==1.14.0",
]
```

`sim` provides the MuJoCo solver used by Moya. `importers` provides the MJCF and
USD import paths needed by the tracked assets. Warp is pinned to the version
used by the verified Moya CUDA environment. The extra is not added to `all`, so
base LeRobot commands remain lightweight and do not import Newton.

The intended local environment command is:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync \
  --extra training \
  --extra diffusion \
  --extra moya_newton
```

The lockfile will be updated and committed with the dependency declaration.

## Architecture

### Environment configuration

Register a `MoyaNewtonEnvConfig` choice under `moya_newton`. Its relevant
defaults are:

```text
task                           randomized_grasp_charger
task_description               grasp and lift randomized charger
fps                            60
episode_length                 930
device                         cuda:0
headless                       true
sim_substeps                   8
preset                         randomized_grasp_v1
success_min_final_lift_height  0.015
supports_rendering             false
```

Its policy feature contract is:

```text
agent_pos  -> observation.state  float32[39]
action     -> action             float32[14]
```

`create_envs()` constructs one fused Moya vector environment containing
`n_envs` worlds and returns:

```python
{"moya_newton": {0: vector_env}}
```

`use_async_envs=True` is ignored with a warning because Moya already owns the
batch and CUDA context. It never creates worker processes or a nested vector
environment.

### Lazy Moya loader

Moya has no packaging metadata and uses top-level imports such as
`moya_model` and `rewards.reward_api`. The adapter therefore resolves the
submodule relative to the LeRobot checkout, temporarily prepends its root to
`sys.path`, imports the backend lazily, and restores `sys.path` afterward.

The loader verifies that imported Moya modules resolve inside the pinned
submodule. A conflicting pre-imported top-level module fails with a clear error
instead of silently using another checkout. Importing `lerobot.envs` without the
`moya_newton` extra remains valid.

### Dataset environment preset

The dataset was not collected with Moya's source defaults. The
`randomized_grasp_v1` preset applies these values before importing or
constructing the backend:

```text
MOYA_IK_ITERS                                  24
MOYA_RIGHT_IK_ROTATION_WEIGHT                  1.0
MOYA_CHARGER_GRASP_WRIST_ACTION_SCALE          0.02
MOYA_CHARGER_GRASP_ROTATION_ACTION_SCALE       0.1
MOYA_HAND_CLOSE_CONTROL_RADIUS                 0.03
MOYA_CHARGER_X                                 0.37
MOYA_CHARGER_Y                                 0.00
MOYA_CHARGER_Z                                 1.095
MOYA_CHARGER_MASS                              0.5
MOYA_CHARGER_BOX_SCALE                         2.0
MOYA_CHARGER_CROSS_HALF                        0.0125
MOYA_CHARGER_GRASP_REFERENCE_X_OFFSET         -0.04
MOYA_CHARGER_GRASP_REFERENCE_Y_OFFSET         -0.03
```

The adapter applies this immutable preset around module import and environment
construction. It then compares `environment_contract()` against schema 14 and
the expected SHA-256:

```text
a06cb5e5a51c3b53a1972e77d803a15c0916395f9c15ecd7172433cc1e6c293e
```

An environment with a different contract is closed and rejected. Charger mass
and box scale are also asserted from the preset because the upstream contract
does not currently encode both values.

### Vector environment facade

The facade retains Moya's native `num_envs`, action timing, SAME_STEP
autoreset, and one-action-per-1/60-second-step behavior. It provides the
additional interface expected by LeRobot:

- `reset(seed=int | list[int] | None)`;
- `step(action)`;
- `call()` and `get_attr()`;
- `_max_episode_steps`, `task`, and `task_description`;
- `single_observation_space`, `observation_space`, `single_action_space`, and
  `action_space`;
- `state()`, `environment_contract()`, and idempotent `close()`; and
- `metadata["render_fps"] = 60`.

Moya owns one NumPy RNG for the fused batch. A LeRobot seed list is validated
to have `num_envs` entries and deterministically folded into one scalar batch
seed. This guarantees reproducibility for a fixed seed list and batch size. It
does not claim batch-size-independent per-world seeding.

Raw Moya observations are validated as finite `float32[num_envs,39]` and
returned as:

```python
{"agent_pos": observation}
```

Actions are validated as finite `float32[num_envs,14]` and passed to Moya
without scaling or semantic conversion. The policy postprocessor already
restores the raw action representation recorded in the dataset; Moya performs
the configured translation and rotation scaling internally.

### Terminal information

Moya returns terminal information as an object array of per-world dictionaries.
LeRobot requires Gymnasium 1.x-style dictionaries of batched arrays. The facade
recursively collates `final_info`, preserves `_final_info` and final-observation
masks, and provides boolean arrays for all worlds.

Moya only truncates at the episode horizon and performs SAME_STEP autoreset.
Terminal metrics are therefore computed from `final_info`, not from the reset
state returned as the next observation.

## Success Metric

The LeRobot `is_success` metric uses the user-approved dataset-style benchmark:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height >= 0.015 m
and final_table_contacts == 0
and final_hand_contacts > 0
```

`true_grasp_ever` means that Moya established a true grasp for 20 consecutive
steps. Each true-grasp step requires the configured finger contacts and closure,
object integrity, bounded charger velocity, bounded charger-wrist relative
velocity, and stable palm-to-charger distance.

`clear_table_ever` retains Moya's collection meaning: after true grasp was
established, the charger reached at least 5 mm lift with zero table contacts at
some point in the episode.

The facade persists both history flags across steps and computes the remaining
conditions from terminal information before SAME_STEP reset. It publishes the
individual conditions as diagnostic info fields. Moya's existing strict
`success` value is preserved as `native_success`, but it does not drive
LeRobot's `pc_success`.

The new 15 mm terminal threshold is intentionally stricter than the original
5 mm collection threshold. Of the 100 accepted demonstrations, 94 meet the new
threshold and 6 would be failures under this evaluation metric.

## Rendering

Add `supports_rendering: bool = True` to the base `EnvConfig`. Set it to `False`
for Moya. Standalone evaluation and training-time evaluation pass zero rendered
episodes when the selected environment does not support rendering, while
retaining their existing video counts for other environments.

The Moya adapter does not construct a viewer and its `render()` result is never
used. No fake RGB frames or empty video files are created.

## Error Handling

The integration fails early with actionable messages for:

- a missing or uninitialized submodule;
- missing Newton or Warp dependencies;
- a Moya module imported from another path;
- an unknown environment preset;
- environment-contract schema or SHA mismatch;
- invalid world count, episode length, seed-list length, or device;
- observation/action shape, dtype, or finite-value violations; and
- malformed Moya terminal information.

Environment construction failures close any partially created backend.
`close()` is safe to call more than once. CUDA errors are not converted into a
CPU fallback because that would hide a requested evaluation configuration.

## Verification

### Unit tests without Newton

Use a fake backend to cover:

- config registration and the 39D state/14D action feature map;
- lazy dependency behavior;
- direct fused environment creation with no nested vector wrapper;
- integer, list, and invalid seeds;
- observation and action validation with zero action transformation;
- `call()`, `get_attr()`, task text, state, contract, and idempotent close;
- object-array to dict-of-arrays terminal conversion;
- SAME_STEP terminal metrics rather than reset metrics;
- every success subcondition and the exact 15 mm boundary; and
- rendering disabled for standalone and training-time evaluation.

### Real backend checks

1. Construct a small CPU batch with the dataset preset and verify the exact
   environment-contract SHA plus finite 39D reset/step observations.
2. Construct a CUDA batch, verify the device path, randomized reset range,
   finite values, and one 14D step.
3. Run the state-only Diffusion preprocessor, policy action selection,
   postprocessor, and Moya step as one CUDA loop.
4. Run a short-horizon standard `lerobot-eval` smoke with a state-only
   Diffusion checkpoint and confirm that no video directory or video file is
   produced.
5. Run focused regression tests for the existing state-only Diffusion Policy
   changes.

A full 930-step benchmark is not required for the integration smoke test. The
production command retains the 930-step default:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run lerobot-eval \
  --policy.path=/path/to/pretrained_model \
  --env.type=moya_newton \
  --env.device=cuda:0 \
  --eval.batch_size=16 \
  --eval.n_episodes=16 \
  --eval.use_async_envs=false \
  --policy.device=cuda
```

## Acceptance Criteria

The integration is complete when:

- the submodule is pinned at the approved revision;
- uv resolves and syncs the training, Diffusion, and Moya extras together;
- base LeRobot imports and tests still work without the Moya extra;
- `moya_newton` produces the exact dataset environment contract;
- the standard LeRobot rollout consumes 39D state and emits raw 14D actions;
- LeRobot reports the agreed success metric from terminal episode data;
- standalone and training-time Moya evaluation do not attempt rendering; and
- focused unit, CPU, CUDA, and policy-loop smoke checks pass.
