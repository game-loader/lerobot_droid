# Single-GPU LPWM-FM A/B experiments on LIBERO Spatial

## Experiment definition

- **A:** shared real DLP visual encoder + causal scene/proprioception tokens + frozen language features + ordinary Flow Matching action expert.
- **B:** the same model and initialization, plus native DLP reconstruction/static priors and action-token-conditioned one-step latent dynamics. Dynamics **always** receives clean demonstration actions, never predicted actions.
- Action conditioning repeats an identical token three times per transition; no AdaLN/FiLM in the robot path. The complete upstream LPWM is separately retained as a native reference.
- No IMF/JVP, action-latent alignment or predicted-action world-consistency objective is enabled in these two experiments.
- Visual/policy weights start from the same random initialization. Language is cached from the actual pretrained SmolVLM2-500M text model, not task IDs. No claim of pretrained DLP features is made.

## Data and time alignment

The row-preserving Spatial subset has 432 episodes, 52,970 rows and 10 task instructions. It is split by episode within each task: seed 42 gives 389 training episodes and 43 validation episodes. State normalization uses only training rows. Native 7-D LIBERO action values are kept unchanged.

Use the transition `(image[i], action[i]) -> image[i+1]` only within the same episode. The data's 10-FPS export timestamps are separate from 20-Hz simulator control. Do not repeat actions, skip rows, or rotate stored images again. The observation histories contain only past/current images; B's future images live in a separate `world.images` batch field.

Cache preparation preserves each global row and episode/frame indices, records hashes of the source Parquet files, and generates real frozen text features:

```bash
uv run --no-sync python scripts/lpwm_ab/prepare_cache.py \
  --root /path/to/libero_spatial --output /path/to/cache \
  --language-model /path/to/cached/SmolVLM2-500M/snapshot
```

Task-balanced validation interleaves held-out tasks so the bounded validation budget is not dominated by the first task. The comparison uses the same validation windows/noise and selects checkpoints by action FM loss, not by total loss with incomparable world terms.

## Actual paired launch

The formal paired launch on 2026-09-17 uses:

| Setting | Both variants |
|---|---|
| Dataset | LIBERO Spatial, 10 tasks |
| Seed | 42 |
| Observations | 2 frames, 2 RGB cameras, 128x128 |
| State/action dimensions | 8 / 7 |
| Language features | 48 tokens x 960, frozen pretrained text model |
| Scene/expert/dynamics width | 256 |
| Scene/expert/dynamics layers | 2 / 4 / 4 |
| Attention heads | 8 |
| Action chunk / execution chunk | 16 / 8 |
| FM inference | 10 Euler steps |
| Parameters instantiated | 18,677,373 |
| Micro-batch / accumulation | 8 / 4 (effective batch 32) |
| Optimizer steps | 20,000 |
| Learning rate / LR warmup | 1e-4 / 500 steps |
| Gradient clip | 10 |
| World supervision | A disabled; B outer weight ramps to 0.1 over 1,000 steps |
| World terms | reconstruction 1, static prior 1e-3, dynamic KL 1 |
| World rollout | one-step GT teacher forcing |
| Validation / checkpoints | every 500 / 2,000 updates |
| Checkpoint retention | latest + best, not every historical checkpoint |
| Precision | float32 with TF32 allowed |

Each job owns one NVIDIA L40S. Preflight measured peak allocated memory ~7.8 GiB for A and ~29.6 GiB for B; reserved memory was ~9.8 / 36.7 GiB. These are measurements at the above configuration, not guarantees for larger models/batches.

Entry points:
- `train.py`: actual training, online SwanLab, finite-value/gradient guards, validation and checkpoint exports.
- `launch_remote.sh`: paired independent background jobs after successful CUDA preflight.
- `data.py`: complete-window sampling, episode splits, train-only normalization and balanced validation ordering.

Credentials are supplied by environment or a restrictive file **outside the repository**. Do not put keys in source, run configuration or command-line literals. The script does not contain any credentials.

The remote root is `/root/lpwm_ab`. Formal outputs:

```text
runs/20260917-183742-A
runs/20260917-183742-B
```

SwanLab project: `game-loader/lpwm-fm-libero-spatial-ab`.

```text
A: https://swanlab.cn/@game-loader/lpwm-fm-libero-spatial-ab/runs/dsksnhn7
B: https://swanlab.cn/@game-loader/lpwm-fm-libero-spatial-ab/runs/biysw016
```

The earlier `20260917-183234-*` online startup checks were stopped and explicitly marked superseded when task-balanced validation was added. Do not mix their metrics with the formal runs.

## Scope of the result

The jobs are training experiments, not completed results. Logged validation loss/action error is **not LIBERO task success**. Simulator rollouts and multi-seed comparisons are still required before judging policy quality. The full repository suite remains separately limited by pre-existing checkpoint serialization failures in the local environment; passing focused tests is not full-release certification.

## Asynchronous LIBERO simulator evaluation (added 2026-09-17)

Offline FM validation is not simulator success. The running training processes are
unchanged; `watch_eval.py` now captures newly saved `latest` checkpoints and queues
real simulator rollouts. Two-pass export checks and immutable snapshots avoid
loading half-written weights or losing a checkpoint while it is being evaluated.
Only currently retained and subsequently saved models can be evaluated; overwritten
historical checkpoints cannot be recovered by this watcher.

`evaluate.py` uses the actual policy action queue and saved training normalization /
language cache, with the same camera orientation and PIL resize convention. No GT
actions or future observations are used. Native 7D commands are checked for finite
values and clipped to [-1,1], without gripper inversion; clipping rate is recorded.

The periodic selection protocol is 10 tasks x 10 episodes, the repository's
280-control-step Spatial limit, 20-Hz control, and 10 settling steps outside the
horizon. A/B use paired seeds and initial states. Validation and final-evaluation
initial-state pools are disjoint. This is the explicitly recorded local protocol,
not an unqualified claim to match every external LIBERO benchmark table.

Metrics are completed-episode successes divided by completed episodes, never
normalized by action chunks. Per-task/overall fractions and percentages are saved
in atomic result JSONs. Simulator failures abort evaluation rather than becoming
false task failures or silently shrinking the denominator. The first episode of
each task is recorded when the video backend is available.

The watcher runs one evaluator at a time on GPU0, alongside A training, to avoid
adding memory load to the heavier B training job. It continues capturing new
checkpoints while a rollout job is running. Initial frozen models: A step16000 and
B step4000; comparisons must account for these different update counts.

SwanLab simulator metrics are written **after each full checkpoint evaluation** to
separate resumable A/B evaluation runs in the same project, avoiding concurrent
writers to live training runs. Fields include `eval/success_rate`, `eval/pc_success`
and per-task success; x-axis step is the checkpoint's training update.

- `checkpoints/best`: existing offline FM-loss selection (unchanged).
- `checkpoints/best_success`: a symlink to the best immutable evaluated model.
- `checkpoints/best_success.json`: result, checkpoint hash/protocol pointer and score.

Start on the configured remote host with `bash scripts/lpwm_ab/launch_eval_remote.sh`.
It uses the independent `.venv-eval` environment and local LIBERO assets; it does not
restart training or modify the training environment's installed packages.

Operational paths under `/root/lpwm_ab/evaluations/20260917-183742/`:

```text
watcher.log / watcher_state.json
snapshots/A/step_XXXXXX/
snapshots/B/step_XXXXXX/
results/A-step_XXXXXX.json
results/B-step_XXXXXX.json
results/*_videos/
```

A short `integration_smoke_A.json` with a five-step horizon is only an interface
smoke test, not a valid full-horizon policy-performance result.
