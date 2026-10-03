# LIBERO LPWM-FM B: common40 production, all130 compatibility — 2026-09-21

## Scope and source contract

Current production scope is the common package: **Spatial10 + Object10 + Goal10 +
LIBERO10 = 40 tasks**, with canonical global IDs **0..39** in that suite/local-ID order.
The catalog is generated separately from local `HuggingFaceVLA/libero`, matching exact
instruction text and preserving task names from the old130 reference catalog. The
LIBERO10 global IDs are now 30..39, not the old130 IDs 120..129. Original dataset task
indices remain provenance, not conditioning tokens. Duplicate language texts may share
only byte-identical frozen pretrained tokens; no artificial task-ID embeddings.

This is joint imitation learning on all selected suites, **not** a LIBERO90-to10
held-out-task generalization experiment. Publish per-suite and aggregate results.
The former all130 scope (including LIBERO90) remains supported: `SUITES` and supervisor
`--task-count` still default to 130. Training infers exactly 40 or 130 from the manifest
and catalog; evaluation infers suites from the checkpoint's own `task_catalog.json`.
Unsupported, partial, duplicate, misnumbered or mismatched catalogs fail closed.

The common-cache producer is `prepare_common40.py` (maintained separately). Its
`lpwm_libero_full_zlib_v1` manifest preserves source rows/episode IDs, native state8 and
action7, and the existing converted RGB orientation. Metadata fps=10 is recorded as
such: do not claim raw20Hz demonstrations, infer physical frequency, filter no-ops,
resample, or rotate converted source images a second time. The old official130 HDF5
path remains separate; its raw-OpenGL rotation/provenance must not be attributed to
this common package. The old Spatial94% is recipe-selection history, not a matched
baseline on common40.

## Model, optimization and measured memory gate

- Fresh initialization, same architecture/initial-weight SHA256 as selected B:
  `3116e036247b84884b6c259de7bf5a5fbdfa6737a8e763c8a2157f5c01d2e13e`.
- Loss weights remain world1 / reconstruction1 / dynamics1 / particleprior0.001.
- Ordinary FM expert; GT-clean action dynamics, repeated action-condition tokens3;
  no AdaLN, IMF/JVP, predicted-action world supervision or action-latent alignment.
- Native DLP; scene2/expert4/world4 layers, width256; state8/action7, RGB128x2,
  history2, prediction16, actionqueue8, Euler10 inference.
- 30000 updates, **microbatch8 x accumulation4 = effective32**, AdamW1e-4,
  warmup500, cosine to1e-5, gradclip10, seed42, world ramp1000, unchanged float32/TF32.
- Supervisor exposes `--batch-size {8,16,32}` and `--grad-accumulation {1,2,4}`,
  forwards them to the trainer and rejects products other than32. Defaults stay8/4.
  These options do not authorize larger production microbatches on the current L40S.

User-reported real-cache/full-world-strength L40S profile (2026-09-21):

| Configuration | Outcome | Peak GiB | Steady update |
| --- | --- | --- | --- |
| 32x1 | OOM | 42.92 | — |
| 16x2 | OOM | 43.20 | — |
| **8x4** | **passed** | **29.57 allocated / 36.64 reserved** | **2.324s** |

Production remains8x4 without changing precision or losses. Optional `--profile-gate`
requires the selected profile JSON to have `status=passed`, matching batch/accumulation,
`effective_batch=32`, `world_scale_step=1000`, `formal_training=false`. A profile is not
formal training or policy-success evidence. The profile JSON itself was not produced
by these local adapter tests.

Offline validation remains every500 updates, balanced across selected tasks. Its
batch size is8 independently of the training microbatch: minimum5 batches for40,
17 for130. Supervisor requests40/130 batches respectively (320/1040 windows).
Keep the fixed30k budget; sufficiency/convergence must be measured, not assumed.

## Cache integrity and launch gates

Per-task seed42 episode-random90/10 train/validation split; train-only state mean/std;
identity native-action normalization. Sampling is episode-safe: action[t] conditions
image[t] to image[t+1]; future images are supervision only, never policy input.
Frozen language embeddings retain exact pretrained provenance.

Images use lossless XOR16/zlib blocks at source-episode boundaries, bounded decoding
and worker-local handles. Materialize NPZ indices before worker fork (CRC regression).
Common40 has episode-level shards (currently1693), potentially many references to the
same task image file. The producer's verified export contract is:

```text
image_shards[i]:
  start/end: contiguous global source frame interval for one episode
  path: task_NNN/images.bin                  # shared across that task's episodes
  index: task_NNN/episode_NNNNNN.npz         # episode-local index
  files_sha256: {images.bin: SHA, episode_NNNNNN.npz: SHA}
task_NNN/complete.json:
  files_sha256: {images.bin: SHA, every episode index: SHA, other task files: SHA}
```

Supervisor verifies scalar/language hashes, contiguous image coverage and explicit
image/index hashes in every shard. It also verifies every file declared by each
existing per-task `complete.json`. Manifest hashes alone remain supported for caches
without completion records; completion records cannot replace missing shard hashes.
Hash checks are deduplicated by resolved path plus expected digest: each file is read
once, and any second declaration with a different digest fails. Each completion
record is read/processed once, not once per episode. Root-relative hash keys and
explicit shard `sha256`/`index_sha256` are also accepted. Paths outside the cache fail.

Production requires actual episodes for all40 (or all130), exact manifest/catalog
identity, no `preflight_only`, correct codec, frozen source hashes and disk reserve.
No partial dataset can be promoted to production. Per-task completion/hash provenance
and source-catalog matching are not substitutes for the producer's source audit.

## Simulation protocol and preflight

Pause training at5k/10k/15k/20k/25k/30k for immutable-checkpoint evaluation on the same GPU.

| Scope | Formal rollout/checkpoint | Six checkpoints | Nonformal preflight |
| --- | --- | --- | --- |
| **common40** | **40x10 = 400 episodes** | **2400 episodes** | **4 suites x task0 x1 episode, max2 control steps** |
| all130 compatibility | 130x10 =1300 | 7800 | 5 suites x task0 x1 episode, max2 control steps |

Protocol declares suites, selected task count, full scope count, episodes/task and
formal eligibility. `validate_result` recomputes exact suite/local/global-task coverage,
episode indices/counts, simulator outcomes, per-task/per-suite totals, aggregate and
suite-macro rates. Explicit protocol contradictions fail; old130 results with only
the preflight marker retain the historical130 default. Formal evaluation never accepts
preflight results. Upload receipts bind the actual result SHA and formal/nonformal role.

Native simulator step caps remain Spatial280/Object280/Goal300/LIBERO90 400/LIBERO10 520.
Use native relative OSC_POSE7, clip[-1,1], no gripper inversion, actionqueue8, reset each
episode, exclude10 settling steps and use actual env success only. Simulator images
retain the existing raw OpenGL rotate180-once preprocessing. Initialization states
use seeded no-replacement validation-firsthalf selection with catalog-global-ID seeds;
final-secondhalf remains reserved. Repeated validation is not independent final evidence.

The launch gate is configurable through `--preflight-run` (absolute or relative to
root) and `--preflight-step` (default2). The backward-compatible directory default is
`preflight_run_v3`; **the current run explicitly selects `preflight_adapter_run`**.
The user-prepared `preflight_adapter_cache` has old official Spatial task0 images and
40-catalog/remapped language, marked `preflight_only` with `source_note`. It may exercise
the four-suite adapter/GPU/upload smoke path before common-cache completion, but it is
**not production common40 data, not a common-cache validation and not a formal result**.
If ready, common-cache-specific smoke checks can be run separately. The supervisor
checks completed preflight step, matching scope, short rollout, verified nonformal
receipt bound to the result, and selected initialization hash before production.

## Run layout and command (not executed by this local code change)

New common40 root: `/root/gpufree-data/lpwm_40/20260921`.
Keep the old `/root/gpufree-data/lpwm_full/20260921` run untouched.

```bash
# Run only by the authorized launch operator after cache/source verification.
# Replace profile8.json with the actual passed8x4 profile artifact's path.
/root/lpwm_ab/.venv/bin/python -u -m scripts.lpwm_full.supervise \
  --root /root/gpufree-data/lpwm_40/20260921 \
  --task-count 40 --batch-size 8 --grad-accumulation 4 \
  --preflight-run preflight_adapter_run --preflight-step 2 \
  --profile-gate profile8.json
```

- `cache/`: production common cache; `preflight_adapter_cache/` stays separate.
- `repo/`, `frozen_source_sha256.json`: immutable checked training source.
- `pipeline_status.json`: preparing_data/starting_training/training/completed/failed.
- `run/`: fresh production output, no accidental overwrite/restart.
- `run/checkpoints/step_005000` through `step_030000`: retain all six complete bundles.
- `run/eval/step_*.json`: complete400-episode payloads and server-verified hashed receipts.
- `supervisor.log`, `training.log`, preparation logs: separate lifecycle records.

Private online SwanLab project `lpwm-fm-b-full-libero`; no embedded credentials.
Exclusive supervisor lock; failed data/preflight/profile/eval/upload gates stop progress.
No automatic restart, checkpoint deletion, unrelated service interruption or remote
process manipulation is part of this local code/test task. No ARA edits.

## Execution handoff (common40)

The local common40 cache completed exhaustive round-trip verification: 1693 episodes,
273465 frames, 546930 camera images; source state8/action7 rows unchanged. Episode split
is train1525/validation168, using the existing seeded per-task10% holdout. Source metadata
fps=10 is recorded without inferring a physical resampling operation. Transfer only the
9.22GiB prepared cache, not the33GiB embedded-PNG source.

`await_transfer.py` accepts the remote cache only after its manifest hash equals the local
exhaustive verification receipt, then runs an additional actual-common40 batch8/full-world
GPU profile and two-update/four-suite preflight (`preflight_common40_run`). Neither is a
formal training result. Only after these gates does the supervisor start the fresh30k run.
The common-cache profile, rather than the earlier adapter-cache profile, is the final
launch gate. Training batch remains8×4 after batch16 and32 failed the isolated GPU probes.
