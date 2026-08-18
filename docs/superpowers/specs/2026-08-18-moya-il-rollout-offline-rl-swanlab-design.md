# Moya IL Rollout → Sparse Offline RL Dataset → SwanLab Design

**Date:** 2026-08-18  
**Status:** Approved by user; implementation pending

## Problem and fixed decisions

The existing 080000 state-only Diffusion Policy checkpoint should first be
rolled out in the pinned Moya Newton simulator. The resulting trajectories are
then used for offline RL-100-style training. Exactly the first 100 episodes in
deterministic candidate order are retained, without filtering, rejecting, or
replacement based on success. Successful and failed trajectories are both
eligible, but the collector does not artificially guarantee that both classes
appear.

The observation/action contract is fixed to state `(39,)` and action `(14,)`.
The success label is the dataset acceptance rule:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height_m >= 0.015
and final_table_contacts == 0
and final_hand_contacts > 0
```

The lift comparison uses the simulator's float32 representation, so the
stored float32 value for exactly 15 mm is accepted consistently by the
collector, summary validator, and v3 adapter. Moya may omit sparse
`reward_components.true_grasp` or `reward_components.clear_table` on steps
where those events did not fire; an omitted component is interpreted as zero.
If a component key is present, its vector shape and finite numeric values are
still validated strictly.

Only the terminal transition receives a sparse reward. A successful terminal
transition has reward `1`; every other transition, including a failed terminal
transition, has reward `0`. No video is recorded.

## Goals

1. Add a dedicated, headless, CUDA Moya collector for an arbitrary local
   Diffusion Policy checkpoint.
2. End each logical episode at the first step at which the five success
   conditions are all true; a failed episode ends at the configured horizon.
3. Write a native LeRobot Dataset v3 with state, action, and canonical RL
   fields `next.reward`, `next.done`, and `next.truncated`.
4. Preserve an auditable JSON summary containing per-episode conditions,
   lengths, seeds, checkpoint hash, and Moya contract hash.
5. Make offline IQL and diffusion-PPO metrics optionally mirror to SwanLab
   without making SwanLab the checkpoint source of truth.

## Non-goals

- Do not change the Diffusion Policy architecture or add an image encoder.
- Do not migrate PointNet/point-cloud paths from RL-100.
- Do not use Moya's dense/native reward or native `success` as the dataset
  label.
- Do not record RGB frames or create video files.
- Do not change the pinned Moya environment contract in this iteration.

## Architecture

### Collector

Use this implementation surface:

```text
RL/collectors/__init__.py
RL/collectors/moya_il.py
RL/cli/collect_moya_il.py
RL/tracking.py
RL/adapters/lerobot_v3.py
RL/trainers/offline.py
RL/cli/train_offline.py
```

The collector loads the checkpoint with the existing processor contract,
constructs the synchronous fused Moya vector environment (`use_async_envs` is
false), and runs policy inference on CUDA. It resets policy history at the
start of each batch and applies the saved pre/post processors exactly once.
The stored state is the raw, finite pre-action `s_t`; the stored action is the
finite, postprocessed 14D action actually passed to `env.step`; and all
`next.*` fields describe the result of that same call.

The native Moya backend currently only truncates at its horizon. To avoid
changing its pinned contract, the collector maintains an explicit active mask
per world. When a world first satisfies the acceptance rule, it records that
step as its terminal transition, stops appending frames for that world, and
does not allow later autoreset observations to enter its episode. Remaining
worlds continue in the fused batch; the batch is reset after all worlds are
logically terminal. This gives exact episode data semantics without private
backend mutation. A future backend-level partial-reset feature is out of scope.

For each active world, store the pre-action observation and action. After the
step, update `true_grasp_ever` and `clear_table_ever` from reward components,
evaluate the five conditions from the current simulator info, and mark the
terminal frame. On an ordinary step, diagnostics come from the top-level info.
On a native done/horizon step, diagnostics and last-step reward components
must come from `final_info` under its matching `_final_info` mask because Moya
uses SAME_STEP autoreset. The last-step components are folded into the two
history flags before success is evaluated. If the horizon action first makes
all five conditions true, success takes precedence and the row is
`(reward=1, done=True, truncated=False)`; only a non-successful horizon is
`(0, True, True)`.

Every active step must expose correctly shaped per-world values for
`true_grasp`, `clear_table`, lift height, table contacts, and hand contacts.
Missing values, malformed masks, or non-finite diagnostics fail immediately,
not only at terminal time.

The two sparse event components are the intentional exception to the missing
value rule above: their absent keys mean zero, while malformed present values
remain fatal.

The terminal post-action simulator state is not written as another actionable
frame. Its diagnostic values are retained in the episode summary, while
`next.done=True` prevents value bootstrapping beyond the stored terminal
transition. The current offline CLI does not enable the optional dynamics
ensemble. A future dynamics run must either mask terminal state-delta loss or
add an explicit terminal next-state feature; it must not train a state target
from the repeated terminal pre-action state.

The target count may be smaller than the fused batch in the final batch. A
deterministic recording mask activates only the lowest world indices needed
for the remaining episode slots. Episode indices are assigned in batch order
and then ascending world index. Non-recording and already completed worlds
still receive the policy's valid postprocessed action so the fused backend is
never given a fabricated rotation/no-op encoding, but their buffers and frozen
episode diagnostics are never changed. This produces exactly 100 saved
episodes rather than rounding up to 112 with 16 worlds.

The CLI accepts one base seed. Batch `b` resets the full fused environment with
the seed list `[base + b*N + i for i in range(N)]`; Moya's existing wrapper
deterministically folds that list to its scalar backend seed. Before resetting
policy history for the batch, CPU and CUDA policy RNGs are seeded with
`base + 1_000_000 + b`. The full environment seed list and policy-noise seed
are recorded per batch. Tail batches still infer a full `N`-world batch, so
their RNG schedule does not depend on the four-world recording mask. This
defines candidate ordering and RNG inputs; it does not claim cross-driver
bitwise CUDA reproducibility.

### Dataset schema

The collection CLI receives one new run directory and publishes this layout:

```text
<output-dir>/
  dataset/                 # official LeRobot Dataset v3 root
  collection_summary.json # audit/completion metadata
```

The writer uses `LeRobotDataset.create(..., use_videos=False)` and declares:

```python
{
    "observation.state": {"dtype": "float32", "shape": (39,), ...},
    "action": {"dtype": "float32", "shape": (14,), ...},
    "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    "next.truncated": {"dtype": "bool", "shape": (1,), "names": None},
}
```

Every frame includes all declared features and a task string. The terminal
success frame is `(1, True, False)`, a failed horizon frame is `(0, True,
True)`, and ordinary frames are `(0, False, False)`. Each episode is saved in
episode-index order and the dataset is finalized before it is reported as
usable. Dataset fps is fixed to and validated against Moya's 60 Hz control
rate. The collector writes no image/video features.

`collection_summary.json` is retained as an audit artifact, not as the only
source of reward. It records the five raw conditions, `success`, terminal
reason, frame count, seed/world identity, checkpoint SHA-256, policy processor
fingerprint, and Moya environment contract.

Collection uses a sibling staging directory whose name contains
`.incomplete`. Only after `finalize()`, reload validation, target-count checks,
and summary validation succeed is `collection_summary.json` written with
`"complete": true` and the staging directory atomically renamed to the final
dataset-run path. An interruption leaves only the staging path; the requested
final path does not exist. Offline inspection/training requires the complete
summary and rejects a staging path, a missing completion flag, or a count
mismatch.

The checkpoint model hash is SHA-256 over `model.safetensors`; config SHA-256
is over the exact `config.json` bytes; processor identity uses the existing
combined processor-artifact fingerprint. The Moya contract records the
backend-provided schema version, SHA-256, and full payload after the existing
adapter has verified them; the collector does not invent a second JSON hash.

The LeRobot v3 offline adapter reads the canonical `next.*` fields when they
are present. It requires exactly one `next.done=True` on the last frame,
requires reward zero on all non-terminal frames, and accepts terminal reward
only in `{0, 1}`. It also requires `next.truncated=False` everywhere for a
success and only on the last row for a failed horizon. It cross-checks the
terminal reward against the five summary conditions and rejects any
disagreement. Decision construction aggregates those raw rows without
relabeling: the terminal chunk receives its binary terminal reward and done
flag, while all preceding chunks receive zero/false. The existing
summary-derived path is retained only for legacy datasets that have none of
the three canonical RL fields; partially present fields are invalid.

### Offline training and SwanLab

Add a shared optional scalar tracking adapter used by offline (and compatible
with the existing online) trainers. The local `metrics.jsonl` is written and
flushed first and remains the authoritative checkpoint input. SwanLab is a
best-effort mirror unless strict mode is explicitly requested.

The offline CLI gains project/run-name and tracking-mode options. SwanLab is
injected reproducibly with `uv run --with swanlab==0.9.4`; it is not a core
project dependency. If tracking is requested but SwanLab is unavailable, fail
before changing the output directory and print that exact command. If a
non-strict SwanLab log call fails during training, warn once, disable remote
logging, and continue writing JSONL/checkpoints. `finish()` runs exactly once
in a `finally` block after any successful SwanLab initialization. On success
this is after checkpoint publication; on training/save failure it runs during
exception unwinding. A `finish()` failure is warned and cannot replace an
earlier training/checkpoint exception. SwanLab initialization failure is
fail-fast in strict mode and warns then disables remote tracking in non-strict
mode, before training output is created.

SwanLab steps are one-based canonical JSONL row numbers, including both IQL and
actor rows. Progress fields are part of the locally persisted row, not only the
remote mirror. `metrics_rows` is the post-write 1-based row number; every other
counter is its value after the just-completed update; and SwanLab receives
`step == metrics_rows`. Each row retains current metrics and adds finite
progress fields (`metrics_rows`, IQL updates, actor updates,
samples/decisions seen, and a numeric phase id: `0` for IQL and `1` for actor).
Dataset counts, hashes, seed, and resolved configuration are logged as run
config/summary rather than extra metric rows. No SwanLab files are placed
inside the hashed RL checkpoint bundle.

In strict mode a remote log failure happens only after the corresponding local
row has been committed. Training stops, no final checkpoint is published, and
the partial output remains auditable but is not automatically rerunnable in
place. A retry uses a new output directory. Non-strict mode remains the
default and continues locally.

## Error handling and reproducibility

- Refuse an existing non-empty dataset/output path unless an explicit resume
  mode is implemented; this prevents metric-row mismatches.
- Validate checkpoint dimensions, processor fingerprint, action mask, Moya
  contract, finite tensors, episode indices, and the requested saved-episode
  count. The formal run requests exactly 100 episodes; smoke runs may request
  fewer without weakening formal acceptance.
- Preserve deterministic seed and per-world seed metadata.
- On interruption, discard an open episode and leave only the `.incomplete`
  staging directory; do not present it as a valid v3 dataset.
- Keep success labels independent from Moya native dense reward and native
  success.

## Verification plan

Unit tests cover: each success-condition counterexample and the exact 15 mm
boundary; first-success-only termination; sparse reward sequences for success
and failure; SAME_STEP horizon success precedence and final-info masking;
raw-state/actual-action/result alignment; inactive-world masking; the 4-world
tail of a 100-episode/16-world run; variable episode lengths; canonical v3
feature dtype/shape; contiguous 100 episode indices; native-success mismatch;
SwanLab disabled/missing/non-strict failure/strict failure and finish behavior;
and one-based post-update metric steps.

CUDA integration smoke collects 2–4 episodes with 2–4 worlds, reloads the v3
dataset, checks finite `(39,)`/`(14,)` tensors and reward/done consistency, then
runs offline smoke training and reloads the final RL checkpoint. Formal
acceptance collects exactly 100 episodes, reports success+failure = 100 and
both class counts without requiring either to be nonzero, checks per-episode
terminal semantics and reward sums, and asserts no video files/directories
were created.
