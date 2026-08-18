# Moya IL Rollout Sparse Offline RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect the first 100 Moya Diffusion Policy episodes without label filtering into a canonical sparse-reward LeRobot v3 dataset, then train offline IQL/diffusion PPO with optional SwanLab mirroring.

**Architecture:** A pure collector state machine owns per-world histories and SAME_STEP terminal extraction; a thin CLI owns policy/environment construction and atomic dataset publication. The v3 adapter treats `next.reward`, `next.done`, and `next.truncated` as truth for new datasets and cross-checks the audit summary. Offline metrics are persisted to JSONL first and mirrored to SwanLab with the same one-based step.

**Tech Stack:** Python 3.12, PyTorch/CUDA, LeRobot Dataset v3, Gymnasium, Newton/Warp, uv, SwanLab 0.9.4, pytest, ruff

---

## File map

- Create `RL/collectors/__init__.py`: collector exports.
- Create `RL/collectors/moya_il.py`: state machine, SAME_STEP parsing, rollout, and publication.
- Create `RL/cli/collect_moya_il.py`: policy/Moya construction, hashes, seeds, and CLI.
- Modify `RL/adapters/lerobot_v3.py`: canonical sparse fields plus legacy fallback.
- Create `RL/tracking.py`: finite scalar tracker and SwanLab implementation.
- Modify `RL/trainers/offline.py`: post-update progress rows and tracker steps.
- Modify `RL/cli/train_offline.py`: SwanLab lifecycle and flags.
- Create `tests/rl100/test_moya_il_collector.py`.
- Create `tests/rl100/test_tracking.py`.
- Modify `tests/rl100/test_lerobot_v3_adapter.py`.
- Modify `tests/rl100/test_offline_trainer.py`.
- Modify `RL/README.md` and `RL/MIGRATION.md`.

### Task 1: Canonical LeRobot v3 RL fields

**Files:**
- Modify: `tests/rl100/test_lerobot_v3_adapter.py`
- Modify: `RL/adapters/lerobot_v3.py`

- [ ] **Step 1: Write failing canonical-field tests**

Extend the fake episode helper with this exact schema:

~~~python
def _add_rl_fields(
    episode: dict[str, torch.Tensor], *, success: bool
) -> dict[str, torch.Tensor]:
    length = episode["action"].shape[0]
    episode["next.reward"] = torch.zeros(length, 1, dtype=torch.float32)
    episode["next.done"] = torch.zeros(length, 1, dtype=torch.bool)
    episode["next.truncated"] = torch.zeros(length, 1, dtype=torch.bool)
    episode["next.done"][-1] = True
    if success:
        episode["next.reward"][-1] = 1.0
    else:
        episode["next.truncated"][-1] = True
    return episode
~~~

Test that a 35-frame success produces decision rewards `[0, 1]` and done flags `[False, True]`. Add explicit rejection tests for early reward, early done, partial fields, success marked truncated, failure not marked truncated, nonbinary reward, malformed shapes, and disagreement with the five summary fields.

- [ ] **Step 2: Verify the tests fail**

Run:

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest   tests/rl100/test_lerobot_v3_adapter.py -q --maxfail=10
~~~

Expected: new tests fail because the adapter neither loads nor validates the three fields.

- [ ] **Step 3: Implement all-or-none field discovery**

Add:

~~~python
_RL_FIELDS = ("next.reward", "next.done", "next.truncated")


def _canonical_rl_fields(features: Mapping[str, Any]) -> tuple[str, ...]:
    present = tuple(key for key in _RL_FIELDS if key in features)
    if present and len(present) != len(_RL_FIELDS):
        missing = sorted(set(_RL_FIELDS) - set(present))
        raise ValueError(f"canonical RL fields are partially present; missing={missing}")
    return present
~~~

Load the fields when present. Require reward `float32[frames,1]`, flags `bool[frames,1]`, exactly one final done, zero nonterminal rewards, and one of the two terminal tuples:

~~~text
success: (reward=1, done=True, truncated=False)
failure: (reward=0, done=True, truncated=True)
~~~

Canonical rows drive chunk reward/done without relabeling. A legacy dataset with none of the three fields retains the current summary-derived path.

- [ ] **Step 4: Enforce completed collection summaries**

For canonical datasets require `complete is True`, `episodes_saved == dataset.num_episodes`, and per-episode terminal reward equality with `terminal_success(summary_record)`. Reject a staging path whose name contains `.incomplete`.

- [ ] **Step 5: Verify and commit**

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_lerobot_v3_adapter.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check   RL/adapters/lerobot_v3.py tests/rl100/test_lerobot_v3_adapter.py
git diff --check
git add RL/adapters/lerobot_v3.py tests/rl100/test_lerobot_v3_adapter.py
git commit -m "feat(rl): load canonical sparse rewards from v3 datasets"
~~~

### Task 2: Pure Moya sparse-episode state machine

**Files:**
- Create: `RL/collectors/__init__.py`
- Create: `RL/collectors/moya_il.py`
- Create: `tests/rl100/test_moya_il_collector.py`

- [ ] **Step 1: Write failing first-success and SAME_STEP tests**

Create fake two-world info where the top-level horizon values are reset values and `final_info` contains the actual terminal values. Test:

~~~python
def test_first_acceptance_step_freezes_episode() -> None:
    batch = EpisodeBatch.create(np.array([True, True]))
    batch.append_step(states(1.0), actions(1.0), diagnostics(lift=(0.014, 0.0)))
    batch.append_step(states(2.0), actions(2.0), diagnostics(lift=(0.015, 0.0)))
    batch.append_step(states(3.0), actions(3.0), diagnostics(lift=(0.020, 0.0)))
    episode = batch.finalized()[0]
    assert episode.states.shape[0] == 2
    assert episode.rewards[:, 0].tolist() == [0.0, 1.0]
    assert episode.dones[:, 0].tolist() == [False, True]
~~~

Also test exact 15 mm, each false subcondition, native-success independence, horizon success precedence, final-info mask mismatch, missing component keys, wrong shapes, and non-finite diagnostics.

- [ ] **Step 2: Verify imports fail**

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest   tests/rl100/test_moya_il_collector.py -q --maxfail=10
~~~

Expected: failure because `RL.collectors.moya_il` does not exist.

- [ ] **Step 3: Implement the public data contracts**

Implement:

~~~python
@dataclass(frozen=True)
class StepDiagnostics:
    true_grasp: np.ndarray
    clear_table: np.ndarray
    lift_height: np.ndarray
    table_contacts: np.ndarray
    hand_contacts: np.ndarray
    native_done: np.ndarray


@dataclass(frozen=True)
class CollectedEpisode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    metadata: dict[str, Any]
~~~

Implement `EpisodeBatch.create(recording_mask)`, `append_step`, `complete`, and `finalized`. Validate finite float32 `[N,39]` states and `[N,14]` actions. Fold the current step's grasp/clear-table components before success testing. Freeze each recorded world on its first success; a non-success native horizon becomes truncated failure.

- [ ] **Step 4: Implement terminal-aware diagnostic extraction**

Add:

~~~python
def extract_step_diagnostics(
    info: Mapping[str, Any],
    *,
    terminated: np.ndarray,
    truncated: np.ndarray,
    num_envs: int,
) -> StepDiagnostics:
~~~

Read top-level reward components and contact/lift values for ordinary rows. For native-done rows require `_final_info == terminated | truncated` and replace every required value from collated `final_info`. Reject reset leakage, absent keys, malformed masks, and non-finite values.

- [ ] **Step 5: Implement tail masks and seed derivation**

~~~python
def recording_mask(remaining: int, num_envs: int) -> np.ndarray:
    if remaining <= 0 or num_envs <= 0:
        raise ValueError("remaining and num_envs must be positive")
    result = np.zeros(num_envs, dtype=np.bool_)
    result[: min(remaining, num_envs)] = True
    return result


def batch_seeds(base_seed: int, batch_index: int, num_envs: int) -> tuple[list[int], int]:
    env_seeds = [
        base_seed + batch_index * num_envs + index for index in range(num_envs)
    ]
    return env_seeds, base_seed + 1_000_000 + batch_index
~~~

Test `remaining=4,num_envs=16` and exact seed results.

- [ ] **Step 6: Verify and commit**

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_moya_il_collector.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check   RL/collectors tests/rl100/test_moya_il_collector.py
git diff --check
git add RL/collectors tests/rl100/test_moya_il_collector.py
git commit -m "feat(rl): add Moya sparse rollout state machine"
~~~

### Task 3: Policy rollout, official v3 writer, and atomic CLI

**Files:**
- Modify: `RL/collectors/moya_il.py`
- Create: `RL/cli/collect_moya_il.py`
- Modify: `tests/rl100/test_moya_il_collector.py`

- [ ] **Step 1: Add a fake vector rollout test**

Use four fake worlds and request three episodes. The fake policy must expose the postprocessed action it returns and the fake environment must retain the exact action passed to `step`. Assert three contiguous episode indices, equality between stored and executed action, one policy reset per fused batch, and no fourth saved world.

- [ ] **Step 2: Implement rollout orchestration**

Define:

~~~python
class PolicyRunner(Protocol):
    def reset(self) -> None:
        raise NotImplementedError

    def select_action(self, raw_states: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def collect_rollouts(
    env: Any,
    policy: PolicyRunner,
    *,
    target_episodes: int,
    episode_length: int,
    base_seed: int,
) -> CollectionResult:
~~~

For every fused batch: derive full environment and policy-noise seeds, reset policy/environment, infer a full N-world action even in the four-world tail, bind diagnostics to the same `env.step`, stop when every recording-mask row is logically terminal, and return only the requested count in batch/world order.

- [ ] **Step 3: Add an official writer round-trip test**

Write a two-frame success and failure to a temporary official `LeRobotDataset` with `use_videos=False`, finalize, reload, and assert v3.0, two episodes, shapes `(39,)/(14,)/(1,)`, exact terminal values, and no camera/video keys. Point `HF_HOME` and `HF_DATASETS_CACHE` at test-local writable paths.

- [ ] **Step 4: Implement atomic publication**

Implement:

~~~python
def publish_collection(
    output_dir: Path,
    *,
    repo_id: str,
    episodes: Sequence[CollectedEpisode],
    summary: Mapping[str, Any],
    fps: int = 60,
) -> Path:
~~~

Reject an existing final path. Write to sibling `name.incomplete-UUID` with layout `dataset/` and `collection_summary.json`. Write an initial `complete=False` summary, use the official writer, finalize, reload/validate all rows, replace summary with `complete=True`, fsync it, and atomically rename staging to final. Preserve staging and never create final on failure.

The reload validator reads the raw columns once and checks them in linear
frame order. It also enforces the fixed 60 Hz rate, the float32 15 mm
threshold semantics, and the five summary conditions before publication.

- [ ] **Step 5: Implement the collection CLI**

Expose:

~~~text
--checkpoint PATH                 required
--output-dir PATH                 required
--repo-id TEXT                    required
--episodes INT                    default 100
--num-envs INT                    default 16
--episode-length INT              default 930
--device TEXT                     default cuda
--sim-device TEXT                 default cuda:0
--inference-steps INT             default 100
--seed INT                        default 1000
--smoke                           2 episodes, 2 envs, 2 inference steps
~~~

The real runner loads the exact checkpoint/pre/post processors, validates state `(39,)` and action `(14,)`, stores raw pre-action state and postprocessed action, and never creates a viewer. Record model/config SHA-256, processor fingerprint, Moya schema/hash/full contract, seeds, episode lengths, terminal fields, and counts.

- [ ] **Step 6: Verify and commit**

~~~bash
HF_HOME="$PWD/.hf-datasets-cache/hf" HF_DATASETS_CACHE="$PWD/.hf-datasets-cache/datasets" UV_CACHE_DIR=.uv-cache uv run pytest   tests/rl100/test_moya_il_collector.py   tests/rl100/test_lerobot_v3_adapter.py -q --maxfail=10
UV_CACHE_DIR=.uv-cache uv run ruff check   RL/collectors RL/cli/collect_moya_il.py tests/rl100/test_moya_il_collector.py
git diff --check
git add RL/collectors/moya_il.py RL/cli/collect_moya_il.py   tests/rl100/test_moya_il_collector.py
git commit -m "feat(rl): collect IL rollouts as sparse v3 data"
~~~

### Task 4: Durable SwanLab mirroring for offline training

**Files:**
- Create: `RL/tracking.py`
- Create: `tests/rl100/test_tracking.py`
- Modify: `RL/trainers/offline.py`
- Modify: `RL/cli/train_offline.py`
- Modify: `tests/rl100/test_offline_trainer.py`

- [ ] **Step 1: Write failing tracker lifecycle tests**

Use a fake SwanLab module. Test explicit `step=1`, non-strict log failure warns/disables after one call, strict failure propagates, disabled mode does not import, missing package fails before output creation, idempotent finish, and finish failure never masks a training exception.

- [ ] **Step 2: Implement the tracking boundary**

~~~python
class ScalarTracker(Protocol):
    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class SwanLabConfig:
    project: str
    run_name: str | None
    mode: Literal["online", "offline", "local", "disabled"]
    log_dir: Path
    strict: bool = False
~~~

Implement `create_swanlab_tracker(config, run_config)`. Use SwanLab 0.9.4 names `name` and `log_dir`; validate positive steps and finite scalar metrics. In non-strict mode, warn once then disable after init/log failure. Make finish idempotent and warning-only.

- [ ] **Step 3: Write failing offline row tests**

After one IQL and one actor update, assert JSONL `progress/metrics_rows == [1,2]`, `progress/phase_id == [0,1]`, tracker steps `[1,2]`, and counter rows `2`.

- [ ] **Step 4: Persist rows before mirroring**

Add `tracker: ScalarTracker | None` to `OfflineTrainer`. Change `record_metrics(metrics, phase_id)` to build a finite row from post-update counters, append/flush it, update `metrics_rows`, then call `tracker.log(row, step=row_number)`. Preserve `train_step` with a numeric combined phase id.

- [ ] **Step 5: Add CLI flags and lifecycle**

Add `--swanlab-project`, `--swanlab-run-name`, `--swanlab-mode online|offline|local|disabled`, and `--swanlab-strict`. Initialize before `metrics.jsonl`, log dataset counts/hashes/args as config, pass phase 0 for IQL and phase 1 for actor, and finish exactly once in `finally`. Run with `uv run --with swanlab==0.9.4`.

- [ ] **Step 6: Verify and commit**

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest   tests/rl100/test_tracking.py tests/rl100/test_offline_trainer.py -q --maxfail=10
UV_CACHE_DIR=.uv-cache uv run ruff check   RL/tracking.py RL/trainers/offline.py RL/cli/train_offline.py   tests/rl100/test_tracking.py tests/rl100/test_offline_trainer.py
git diff --check
git add RL/tracking.py RL/trainers/offline.py RL/cli/train_offline.py   tests/rl100/test_tracking.py tests/rl100/test_offline_trainer.py
git commit -m "feat(rl): mirror offline training metrics to SwanLab"
~~~

### Task 5: Integration, review, CUDA collection, and training

**Files:**
- Modify: `RL/README.md`
- Modify: `RL/MIGRATION.md`
- Verify: `tests/rl100/`, Moya env tests, Diffusion state-only test

- [ ] **Step 1: Document exact commands and schema**

Document first-success termination, no label filtering, canonical truth table, 60 Hz, no video, atomic incomplete staging, the 080000 checkpoint, and SwanLab 0.9.4 invocation.

- [ ] **Step 2: Run full CPU regression**

~~~bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100 -q --maxfail=10
UV_CACHE_DIR=.uv-cache uv run pytest   tests/policies/test_diffusion_state_only.py tests/envs/test_moya_newton.py -q
UV_CACHE_DIR=.uv-cache uv run ruff check RL tests/rl100
git diff --check
~~~

- [ ] **Step 3: Complete independent spec and code-quality reviews**

Give reviewers the design, this plan, and all task commits. Resolve every blocking/important finding and repeat Step 2.

- [ ] **Step 4: Commit docs/review fixes without user files**

~~~bash
git add RL/README.md RL/MIGRATION.md RL tests/rl100
git commit -m "docs(rl): document sparse IL rollout training"
~~~

Do not stage `AGENTS.md`, `.hf-datasets-cache/`, or unrelated user plan files.

- [ ] **Step 5: Run CUDA collector and offline smoke outside the sandbox**

~~~bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.collect_moya_il   --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model   --output-dir outputs/rl100/collections/moya_il_smoke   --repo-id local/moya-il-smoke --device cuda --sim-device cuda:0 --smoke
UV_CACHE_DIR=.uv-cache uv run --with swanlab==0.9.4   python -m RL.cli.train_offline   --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model   --dataset-root outputs/rl100/collections/moya_il_smoke/dataset   --repo-id local/moya-il-smoke   --summary outputs/rl100/collections/moya_il_smoke/collection_summary.json   --output-dir outputs/rl100/offline_moya_il_smoke   --device cuda --smoke   --swanlab-project moya-rl100 --swanlab-run-name offline-moya-il-smoke   --swanlab-mode online
~~~

Expected: two episodes, loadable v3 data, one IQL plus one actor step, two SwanLab/JSONL rows, loadable checkpoint, and no video.

- [ ] **Step 6: Collect and validate the formal first 100 episodes**

Set `RUN_ID` once and execute outside the sandbox:

~~~bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
COLLECTION=outputs/rl100/collections/moya_diffusion_080000_sparse_100_$RUN_ID
UV_CACHE_DIR=.uv-cache uv run python -m RL.cli.collect_moya_il   --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model   --output-dir "$COLLECTION"   --repo-id local/moya-diffusion-080000-sparse-100   --episodes 100 --num-envs 16 --episode-length 930   --inference-steps 100 --seed 1000 --device cuda --sim-device cuda:0
~~~

Validate complete flag, 100 indices, success+failure=100, lengths 1..930, finite `(39,)/(14,)`, terminal reward sums, done/truncated semantics, and no video extension or videos directory.

- [ ] **Step 7: Launch bounded full offline training with SwanLab**

Use the formal collection path and a new `TRAIN_ID`:

~~~bash
TRAIN_ID=$(date +%Y%m%d-%H%M%S)
UV_CACHE_DIR=.uv-cache uv run --with swanlab==0.9.4   python -m RL.cli.train_offline   --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model   --dataset-root "$COLLECTION/dataset"   --repo-id local/moya-diffusion-080000-sparse-100   --summary "$COLLECTION/collection_summary.json"   --output-dir "outputs/rl100/offline_moya_il_sparse_$TRAIN_ID"   --device cuda --batch-size 32   --iql-steps 100000 --actor-steps 300000 --inference-steps 10   --swanlab-project moya-rl100   --swanlab-run-name "offline-moya-il-sparse-$TRAIN_ID"   --swanlab-mode online
~~~

Persist the PID, stdout/stderr log, resolved paths, and SwanLab URL. Do not claim completion until the final RL checkpoint reloads and its manifest and metric-row count validate.
