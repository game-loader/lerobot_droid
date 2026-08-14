# Moya Newton LeRobot Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pinned Moya Newton fused simulator to the local uv environment and expose a headless, state-only moya_newton environment through the standard LeRobot evaluation CLI.

**Architecture:** Keep Moya's native GPU-batched gym.vector.VectorEnv inside a LeRobot VectorWrapper facade. Load the un-packaged submodule lazily under the dataset collection preset, normalize observations and terminal info to LeRobot contracts, and disable video callbacks for this environment. Validate first with a fake backend, then with the real CPU/CUDA backend and a tiny state-only Diffusion checkpoint.

**Tech Stack:** Python 3.12, uv, Gymnasium 1.3, PyTorch/CUDA, Newton 1.3, Warp 1.14, LeRobot processors, pytest

---

## File Map

- Create: .gitmodules and gitlink third_party/moya_newton_sim for the pinned Moya checkout.
- Modify: pyproject.toml and uv.lock for the opt-in moya_newton dependency group.
- Modify: src/lerobot/policies/diffusion/configuration_diffusion.py and src/lerobot/policies/diffusion/modeling_diffusion.py; preserve the already-written state-only conditioning change.
- Create: src/lerobot/envs/moya_newton.py containing the preset loader, terminal-info collation, success tracker, and MoyaNewtonVectorEnv facade.
- Modify: src/lerobot/envs/configs.py and src/lerobot/envs/__init__.py to register and export MoyaNewtonEnvConfig.
- Modify: src/lerobot/envs/configs.py, src/lerobot/scripts/lerobot_eval.py, and src/lerobot/scripts/lerobot_train.py for the generic supports_rendering capability.
- Create: tests/envs/test_moya_newton.py with fake-backend TDD coverage.
- Create: tests/scripts/test_lerobot_eval_rendering.py for rendering capability resolution.
- Preserve and run: tests/policies/test_diffusion_state_only.py.
- Create: docs/source/moya_newton.mdx and modify docs/source/_toctree.yml with install/eval instructions and metric definitions.

### Task 1: Land The Existing State-Only Diffusion Support

**Files:**
- Modify: src/lerobot/policies/diffusion/configuration_diffusion.py (already changed in the worktree)
- Modify: src/lerobot/policies/diffusion/modeling_diffusion.py (already changed in the worktree)
- Test: tests/policies/test_diffusion_state_only.py (already present in the worktree)

- [ ] Step 1: Inspect the existing change and run its focused test

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/policies/test_diffusion_state_only.py -q
```

Expected: one passing test. If the environment is missing a dependency, run the base training sync from the earlier uv plan before changing source files.

- [ ] Step 2: Run the Diffusion policy regression subset

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/processor/test_diffusion_processor.py -q
```

Expected: all existing Diffusion tests pass; image and environment-state paths remain valid while a state-only input is now accepted.

- [ ] Step 3: Commit only the prerequisite policy change

Run:

```
git add src/lerobot/policies/diffusion/configuration_diffusion.py \
  src/lerobot/policies/diffusion/modeling_diffusion.py \
  tests/policies/test_diffusion_state_only.py
git commit -m "feat: allow state-only diffusion policy conditioning"
```

Expected: the commit contains exactly the two policy files and their focused test. Do not stage the design/plan documents, submodule, or unrelated worktree files in this commit.

### Task 2: Add And Pin The Moya Submodule

**Files:**
- Create: .gitmodules
- Create: third_party/moya_newton_sim as a gitlink

- [ ] Step 1: Verify the remote and approved revision before writing the gitlink

Run:

```
GIT_SSH_COMMAND='ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new' \
git ls-remote git@gitlab.com:game-loader/moya_newton.git \
  HEAD refs/heads/main
```

Expected: the main revision includes 7dd6af2952e1153d8943bc7b255e877b96ed0e55.

- [ ] Step 2: Add the submodule at the agreed path

Run:

```
git submodule add git@gitlab.com:game-loader/moya_newton.git \
  third_party/moya_newton_sim
git -C third_party/moya_newton_sim checkout \
  7dd6af2952e1153d8943bc7b255e877b96ed0e55
```

Expected: .gitmodules records the SSH URL and the working tree is detached at the approved full SHA. The checkout must contain moya_batched_env.py, moya_model.py, rewards/, and the MJCF/USD assets, with no ignored runs/ data.

- [ ] Step 3: Verify the gitlink and commit it separately

Run:

```
git submodule status -- third_party/moya_newton_sim
git diff --check -- .gitmodules
git add .gitmodules third_party/moya_newton_sim
git commit -m "build: pin Moya Newton simulator submodule"
```

Expected: git submodule status prints the approved SHA with no leading plus or minus, and the commit contains only .gitmodules and the gitlink.

### Task 3: Add The Opt-In Newton Dependency Extra

**Files:**
- Modify: pyproject.toml near the simulation extras
- Modify: uv.lock via uv, never by hand

- [ ] Step 1: Add the dependency declaration without changing the default extras

Insert this exact group after the existing simulation extras and do not add it to all:

```
moya_newton = [
    "newton[sim,importers]==1.3.0",
    "warp-lang==1.14.0",
]
```

Keep the existing PyTorch CUDA index/source configuration unchanged.

- [ ] Step 2: Resolve the lockfile with the project-local uv cache

Run:

```
UV_CACHE_DIR=.uv-cache uv lock
```

Expected: uv resolves newton==1.3.0 and warp-lang==1.14.0 together with the existing Torch 2.11/CUDA 12.8 graph and updates only uv.lock plus the declared pyproject.toml change. If the resolver selects another Warp version, stop and inspect the Newton metadata rather than weakening the pin.

- [ ] Step 3: Verify both opt-in and base lock behavior

Run:

```
UV_CACHE_DIR=.uv-cache uv lock --check
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync \
  --locked --extra training --extra diffusion --extra moya_newton \
  --extra test --extra dev
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python \
  -c "import newton, warp; print('moya_deps_ok', newton.__version__)"
```

Expected: lock validation succeeds, the local environment imports Newton and Warp, and importing lerobot.envs without constructing moya_newton still does not import Newton.

- [ ] Step 4: Commit dependency metadata

Run:

```
git add pyproject.toml uv.lock
git commit -m "build: add optional Moya Newton dependencies"
```

### Task 4: Build The Adapter With Fake-Backend TDD

**Files:**
- Create: tests/envs/test_moya_newton.py
- Create: src/lerobot/envs/moya_newton.py

The production facade must subclass gym.vector.VectorWrapper, not construct a second vector environment. Its constructor accepts an already-created native backend so unit tests can run without Newton:

```
MoyaNewtonVectorEnv(
    env: gym.vector.VectorEnv,
    *,
    episode_length: int,
    task: str,
    task_description: str,
    success_min_final_lift_height: float = 0.015,
)
```

- [ ] Step 1: Write a deterministic fake backend and failing contract tests

In tests/envs/test_moya_newton.py, define a FakeMoyaVectorEnv with num_envs=2, raw Box spaces (39,)/(14,), and reset/step behavior that returns a terminal object array containing charger_lift_height 0.015, zero table contacts, one hand contact, and positive true_grasp/clear_table reward components.

Add these tests with complete setup and assertions:

- test_reset_wraps_agent_pos_and_folds_seed_list
- test_action_shape_and_dtype_are_checked
- test_final_info_is_dict_of_arrays_and_15mm_is_success
- test_lift_just_below_15mm_is_not_success
- test_call_get_attr_task_and_close_are_vector_compatible

The tests must assert observation["agent_pos"].shape == (2, 39), is_success in info["final_info"], exact boolean success at 0.015, and false success at 0.014999. Run before implementation:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/envs/test_moya_newton.py -q
```

Expected: collection fails because lerobot.envs.moya_newton and the facade do not yet exist.

- [ ] Step 2: Implement the wrapper's spaces, reset, and validation

Implement these exact core rules in MoyaNewtonVectorEnv:

```
super().__init__(env)
self.single_observation_space = gym.spaces.Dict(
    {"agent_pos": gym.spaces.Box(-np.inf, np.inf, (39,), np.float32)}
)
self.observation_space = gym.vector.utils.batch_space(
    self.single_observation_space, env.num_envs
)
self.single_action_space = env.single_action_space
self.action_space = env.action_space
self._max_episode_steps = int(episode_length)
self.task = task
self.task_description = task_description
self._true_grasp_ever = np.zeros(env.num_envs, dtype=np.bool_)
self._clear_table_ever = np.zeros(env.num_envs, dtype=np.bool_)
self._closed = False
```

reset() must accept an integer, a Sequence[int], or None. For a sequence, validate exactly num_envs entries and derive a stable scalar with np.random.SeedSequence(np.asarray(seed, dtype=np.uint32)).generate_state(1)[0]. Reset both history arrays, call the backend, validate finite float32[N,39], and return {"agent_pos": obs}.

step() must require shape (N,14), finite numeric values, cast to float32, call the backend once, and return the wrapped observation plus the backend reward/termination arrays unchanged.

- [ ] Step 3: Implement terminal-info collation and success tracking

Add private helpers named _update_history, _collate_final_info, and
_terminal_success. Their return types are None, dict[str, Any], and bool,
respectively; the following paragraphs define their complete behavior.

Update history from positive reward_components["true_grasp"] and reward_components["clear_table"] values on every step, including scalar dicts inside terminal object-array entries. Compute terminal is_success as:

```
bool(self._true_grasp_ever[i])
and bool(self._clear_table_ever[i])
and float(final_info["charger_lift_height"]) >= self.success_min_final_lift_height
and int(final_info["charger_table_contacts"]) == 0
and int(final_info["right_hand_charger_contacts"]) > 0
```

Convert the object array into a dict whose leaves have a leading N dimension; recursively collate nested dictionaries and use object arrays for heterogeneous optional fields. Inject is_success, true_grasp_ever, clear_table_ever, and native_success into the collated terminal dictionary. Set top-level info["is_success"] to the terminal boolean array (false for non-terminal worlds), preserve _final_info, final_obs, and _final_obs, and reset only the history slots for worlds that autoreset.

- [ ] Step 4: Implement vector API delegation and run the tests

Implement:

```
def call(self, name: str, *args, **kwargs):
    value = getattr(self, name)
    value = value(*args, **kwargs) if callable(value) else value
    return tuple(copy.deepcopy(value) for _ in range(self.num_envs))

def get_attr(self, name: str):
    return self.call(name)

def state(self):
    return {"agent_pos": np.asarray(self.env.state(), dtype=np.float32)}

def environment_contract(self):
    return copy.deepcopy(self.env.environment_contract())

def close(self, **kwargs):
    if not self._closed:
        self.env.close(**kwargs)
        self._closed = True
```

render() raises NotImplementedError with a message that rendering is disabled for Moya. Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/envs/test_moya_newton.py -q
```

Expected: all fake-backend tests pass, including the exact 15 mm boundary and the dict-of-arrays final_info shape.

- [ ] Step 5: Commit the adapter and its tests

Run:

```
git add src/lerobot/envs/moya_newton.py tests/envs/test_moya_newton.py
git commit -m "feat: adapt Moya fused vector env for LeRobot"
```

### Task 5: Add The Preset Loader And Environment Config

**Files:**
- Modify: src/lerobot/envs/moya_newton.py
- Modify: src/lerobot/envs/configs.py
- Modify: src/lerobot/envs/__init__.py
- Extend: tests/envs/test_moya_newton.py

- [ ] Step 1: Add loader tests before the real loader

Add tests that a fake constructor observes every preset variable in os.environ,
that an unknown preset raises ValueError, and that a backend returning a
different contract SHA raises RuntimeError and is closed. Also assert that
importing lerobot.envs and constructing MoyaNewtonEnvConfig do not add newton to
sys.modules. Add explicit missing-submodule and missing-Newton tests that assert
the error includes the exact initialization or uv sync command.

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/envs/test_moya_newton.py -q
```

Expected: the new loader tests fail because the preset factory and config are
not implemented yet.

- [ ] Step 2: Implement the immutable preset and lazy loader

Define this exact mapping:

```
MOYA_PRESETS = {
    "randomized_grasp_v1": {
        "MOYA_IK_ITERS": "24",
        "MOYA_RIGHT_IK_ROTATION_WEIGHT": "1.0",
        "MOYA_CHARGER_GRASP_WRIST_ACTION_SCALE": "0.02",
        "MOYA_CHARGER_GRASP_ROTATION_ACTION_SCALE": "0.1",
        "MOYA_HAND_CLOSE_CONTROL_RADIUS": "0.03",
        "MOYA_CHARGER_X": "0.37",
        "MOYA_CHARGER_Y": "0.00",
        "MOYA_CHARGER_Z": "1.095",
        "MOYA_CHARGER_MASS": "0.5",
        "MOYA_CHARGER_BOX_SCALE": "2.0",
        "MOYA_CHARGER_CROSS_HALF": "0.0125",
        "MOYA_CHARGER_GRASP_REFERENCE_X_OFFSET": "-0.04",
        "MOYA_CHARGER_GRASP_REFERENCE_Y_OFFSET": "-0.03",
    }
}
EXPECTED_MOYA_CONTRACT_SHA256 = (
    "a06cb5e5a51c3b53a1972e77d803a15c0916395f9c15ecd7172433cc1e6c293e"
)
```

Use a context manager to save and restore the caller's environment and
temporarily prepend third_party/moya_newton_sim to sys.path. Import
moya_batched_env.MoyaBatchedChargerGraspEnv and moya_model only inside that
context, then construct the backend before restoring the environment. Reject
pre-imported moya_batched_env, moya_model, or rewards modules whose files are
outside the pinned submodule. Translate missing newton, warp, or importer
modules into an error that includes:

```
UV_PROJECT_ENVIRONMENT=.venv uv sync --extra moya_newton
```

After construction, require:

```
contract["schema_version"] == 14
contract["sha256"] == EXPECTED_MOYA_CONTRACT_SHA256
moya_model.CHARGER_MASS == 0.5
```

Close the backend before raising on a mismatch. The box scale is guaranteed by
the construction-time environment variable and covered by the fake-constructor
test because the upstream contract does not expose it.

- [ ] Step 3: Register MoyaNewtonEnvConfig

Add a dataclass registered as moya_newton with these defaults:

```
task = "randomized_grasp_charger"
task_description = "grasp and lift randomized charger"
fps = 60
episode_length = 930
device = "cuda:0"
headless = True
sim_substeps = 8
preset = "randomized_grasp_v1"
success_min_final_lift_height = 0.015
supports_rendering = False
```

Its feature definitions must be exactly:

```
features = {
    "action": PolicyFeature(FeatureType.ACTION, (14,)),
    "agent_pos": PolicyFeature(FeatureType.STATE, (39,)),
}
features_map = {
    "action": "action",
    "agent_pos": "observation.state",
}
```

Reject headless=False, nonpositive episode lengths/substeps, non-finite or
negative lift thresholds, and unknown presets. create_envs() calls the lazy
factory and returns {"moya_newton": {0: facade}}. If use_async_envs is true,
log one warning and still create only the native fused environment.

Export MoyaNewtonEnvConfig from src/lerobot/envs/__init__.py. The export imports
only configs.py and must not import the Newton backend.

- [ ] Step 4: Verify config registration without constructing Newton

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python - <<'PY'
import sys
from lerobot.envs import MoyaNewtonEnvConfig, make_env_config

cfg = make_env_config("moya_newton")
assert isinstance(cfg, MoyaNewtonEnvConfig)
assert cfg.features["agent_pos"].shape == (39,)
assert cfg.features["action"].shape == (14,)
assert cfg.features_map["agent_pos"] == "observation.state"
assert "newton" not in sys.modules
print("moya_config_ok")
PY
```

Expected: moya_config_ok without importing newton.

- [ ] Step 5: Run the adapter/config tests and commit

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/envs/test_moya_newton.py -q
git add src/lerobot/envs/moya_newton.py src/lerobot/envs/configs.py \
  src/lerobot/envs/__init__.py tests/envs/test_moya_newton.py
git commit -m "feat: register Moya Newton environment config"
```

Expected: all fake-backend and loader tests pass. The commit does not modify
evaluation scripts or documentation.

### Task 6: Disable Moya Rendering In Generic Evaluation

**Files:**
- Modify: src/lerobot/envs/configs.py
- Modify: src/lerobot/scripts/lerobot_eval.py
- Modify: src/lerobot/scripts/lerobot_train.py
- Create: tests/scripts/test_lerobot_eval_rendering.py

- [ ] Step 1: Write the capability-resolution test

Add tests for a renderable base config and Moya:

```
assert resolve_max_episodes_rendered(renderable_cfg, 10) == 10
assert resolve_max_episodes_rendered(moya_cfg, 10) == 0
assert resolve_max_episodes_rendered(moya_cfg, 0) == 0
```

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/scripts/test_lerobot_eval_rendering.py -q
```

Expected: import or assertion failure until the helper exists.

- [ ] Step 2: Add the generic capability and helper

Add supports_rendering: bool = True to EnvConfig. Moya overrides it with false.
In lerobot_eval.py add:

```
def resolve_max_episodes_rendered(env_cfg: EnvConfig, requested: int) -> int:
    if requested < 0:
        raise ValueError("requested rendered episodes must be non-negative")
    return requested if env_cfg.supports_rendering else 0
```

Import EnvConfig for the annotation. Use the helper at the standalone eval call
site instead of the literal 10. Import the helper in lerobot_train.py and use it
at the training-time eval call site instead of the literal 4. Keep all other
environments' video behavior unchanged.

- [ ] Step 3: Run rendering and config regressions

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/scripts/test_lerobot_eval_rendering.py tests/envs/test_moya_newton.py -q
```

Expected: all tests pass and no Moya render callback is selected.

- [ ] Step 4: Commit the rendering behavior

Run:

```
git add src/lerobot/envs/configs.py src/lerobot/scripts/lerobot_eval.py \
  src/lerobot/scripts/lerobot_train.py tests/scripts/test_lerobot_eval_rendering.py
git commit -m "feat: disable rendering for headless environments"
```

### Task 7: Document, Sync, And Run Real Smoke Checks

**Files:**
- Create: docs/source/moya_newton.mdx
- Modify: docs/source/_toctree.yml

- [ ] Step 1: Add user-facing installation and evaluation documentation

Document:

- git submodule update --init --recursive;
- the uv sync command with training, diffusion, and moya_newton extras;
- Linux, CUDA, and headless requirements;
- the pinned randomized_grasp_v1 preset and contract SHA;
- the 39D observation.state and 14D raw action contract;
- the success definition with true_grasp_ever, clear_table_ever, final lift at
  least 15 mm, zero final table contacts, and positive final hand contacts;
- the fact that 94 of the 100 accepted demonstrations satisfy the new 15 mm
  benchmark;
- fixed-batch seed reproducibility limitations;
- no-video behavior; and
- the missing upstream Moya license warning.

Add this production command:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run lerobot-eval \
  --policy.path=/path/to/pretrained_model \
  --env.type=moya_newton \
  --env.device=cuda:0 \
  --eval.batch_size=16 \
  --eval.n_episodes=16 \
  --eval.use_async_envs=false \
  --policy.device=cuda
```

Add the page under Benchmarks in docs/source/_toctree.yml.

- [ ] Step 2: Run static and focused unit verification

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync ruff check \
  src/lerobot/envs/moya_newton.py src/lerobot/envs/configs.py \
  src/lerobot/scripts/lerobot_eval.py src/lerobot/scripts/lerobot_train.py \
  tests/envs/test_moya_newton.py tests/scripts/test_lerobot_eval_rendering.py
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/policies/test_diffusion_state_only.py \
  tests/envs/test_moya_newton.py \
  tests/scripts/test_lerobot_eval_rendering.py -q
```

Expected: Ruff exits zero and all focused tests pass.

- [ ] Step 3: Verify the real CPU backend and contract

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python - <<'PY'
import numpy as np
from lerobot.envs import MoyaNewtonEnvConfig

cfg = MoyaNewtonEnvConfig(device="cpu", episode_length=2)
env = cfg.create_envs(n_envs=2, use_async_envs=True)["moya_newton"][0]
try:
    obs, info = env.reset(seed=[11, 12])
    assert obs["agent_pos"].shape == (2, 39)
    assert np.isfinite(obs["agent_pos"]).all()
    charger_position = np.asarray(info["charger_position"], dtype=np.float32)
    assert np.all((charger_position[:, 0] >= 0.355) & (charger_position[:, 0] <= 0.385))
    assert np.all((charger_position[:, 1] >= -0.015) & (charger_position[:, 1] <= 0.015))
    obs, reward, terminated, truncated, info = env.step(
        np.zeros((2, 14), dtype=np.float32)
    )
    assert obs["agent_pos"].shape == (2, 39)
    assert np.isfinite(obs["agent_pos"]).all()
    assert env.environment_contract()["sha256"] == (
        "a06cb5e5a51c3b53a1972e77d803a15c0916395f9c15ecd7172433cc1e6c293e"
    )
    print("moya_cpu_smoke_ok", reward.tolist(), truncated.tolist())
finally:
    env.close()
PY
```

Expected: one moya_cpu_smoke_ok line and no viewer/window creation.

- [ ] Step 4: Create a tiny state-only Diffusion checkpoint for CLI smoke

Write the checkpoint only under /tmp:

```
SMOKE_DIR="/tmp/moya-diffusion-state-only-$(date +%s)"
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync \
  python - "$SMOKE_DIR" <<'PY'
from pathlib import Path
import sys
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)

out = Path(sys.argv[1])
config = DiffusionConfig(
    device="cuda",
    n_obs_steps=2,
    horizon=4,
    n_action_steps=2,
    down_dims=(32, 64),
    diffusion_step_embed_dim=16,
    n_groups=8,
    num_train_timesteps=2,
    num_inference_steps=2,
    pretrained_backbone_weights=None,
    normalization_mapping={
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
        "VISUAL": NormalizationMode.IDENTITY,
    },
    input_features={
        "observation.state": PolicyFeature(FeatureType.STATE, (39,))
    },
    output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
)
policy = DiffusionPolicy(config)
pre, post = make_diffusion_pre_post_processors(config)
out.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(out)
pre.save_pretrained(out)
post.save_pretrained(out)
print(out)
PY
```

Expected: the directory contains config.json, model.safetensors, and both
processor JSON files.

- [ ] Step 5: Run the no-video CUDA LeRobot eval smoke

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync lerobot-eval \
  --policy.path="$SMOKE_DIR" \
  --env.type=moya_newton \
  --env.device=cuda:0 \
  --env.episode_length=2 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --output_dir="$SMOKE_DIR/eval"
```

Expected: the rollout exits successfully with finite reward/metric output, does
not call a viewer, and creates no videos/ directory or MP4 file.

- [ ] Step 6: Run the final focused suite and commit docs

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync pytest \
  tests/policies/test_diffusion_state_only.py \
  tests/envs/test_moya_newton.py \
  tests/scripts/test_lerobot_eval_rendering.py -q
git add docs/source/moya_newton.mdx docs/source/_toctree.yml
git commit -m "docs: document Moya Newton evaluation"
```

Expected: the focused suite is green, the documentation commit contains only
the two documentation files, and earlier commits remain independently
reviewable.

- [ ] Step 7: Return the repository-local environment to runtime-only extras

Run:

```
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv sync \
  --locked --no-dev --extra training --extra diffusion --extra moya_newton
UV_CACHE_DIR=.uv-cache UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python \
  -c "import newton, torch; assert torch.cuda.is_available(); print('runtime_env_ok')"
```

Expected: the final .venv remains in the repository, contains the runtime
training/Diffusion/Newton stack, and no longer requires test/dev packages.

## Self-Review Checklist

- [x] The plan covers the pinned submodule, optional dependencies, lazy import,
  dataset preset, 39D/14D mapping, seed handling, terminal-info conversion,
  success history, no-video eval, docs, and CPU/CUDA smoke requirements from the
  approved design.
- [x] No step relies on a placeholder, an unspecified file, or an unstated
  dependency; every command has an expected result.
- [x] The facade constructor, helper names, config fields, and test references
  use the same names across tasks.
- [x] Existing uncommitted user changes are isolated in Task 1 and are never
  reverted or staged with unrelated files.
