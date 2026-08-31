# RL-100 Diffusion RL Migration

This directory contains a policy-agnostic adaptation of the offline and online
reinforcement-learning stages from RL-100 for LeRobot Diffusion-family
checkpoints. It supports the original state-only Moya policy and the DP3
Franka Duo policy (`34D` state, `18D` action, XYZ point cloud, and two wrist
RGB streams). The LeRobot policy, processor, dataset, and environment
contracts remain the source of truth.

## Scope

- State-only critics and dynamics use `observation.state`; DP3 instead uses a
  frozen multimodal encoder (PointNet + two wrist ResNet18 encoders + state)
  for IQL and AM-Q. Dimensions are read from the checkpoint (Moya is
  39D/14D, DP3 is 34D/18D).
- Offline training uses sparse terminal labels from the LeRobot v3 collection
  summary, with success equal to `1` and failure equal to `0`.
- Online training uses a caller-provided real-robot vector environment,
  stochastic DDIM traces, vector GAE, and masked denoising PPO. No simulator
  is created by the online CLI.
- For the supplied Moya checkpoint, constant action dimensions `3:12` are
  excluded from critic and PPO likelihood reductions; DP3 uses its own saved
  action-range mask.
- Image tensors are preserved by the data and actor paths. An image critic
  encoder is an explicit extension point and is not silently inferred.
- Moya evaluation is headless and never records video.

For a canonical offline dataset, retain the LeRobot v3 tree and declare the
following transition columns in `meta/info.json`:

```text
next.reward    float32 [1]
next.done      bool    [1]
next.truncated bool    [1]
```

Write one value per frame in `data/**/*.parquet`. The three columns are
all-or-none; sparse terminal data uses `(0, false, false)` before the final
frame, `(1, true, false)` for success, and `(0, true, true)` for a failed
timeout. DP3 additionally requires the multimodal observation features
`observation.state[34]`, `observation.point_cloud[2048,3]`,
`observation.images.wrist_left`, and `observation.images.wrist_right`.

The acceptance rule is shared by dataset and simulator paths:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height_m >= 0.015
and final_table_contacts == 0
and final_hand_contacts > 0
```

The comparison is performed in the simulator's `float32` precision, so an
exact stored value of `0.015 m` is accepted. During rollout, missing
`reward_components.true_grasp` or `reward_components.clear_table` entries mean
zero for that step; malformed present entries still fail validation. The
collector stops a world on its first successful terminal step and keeps failed
horizon episodes too. It writes only the canonical v3 fields
`next.reward`, `next.done`, and `next.truncated` at the fixed 60 Hz rate.

To collect the first 100 state-only episodes from the 080000 checkpoint:

```bash
uv run python -m RL.cli.collect_moya_il \
  --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model \
  --output-dir outputs/rl100/collections/moya_diffusion_080000_sparse_100 \
  --repo-id local/moya-diffusion-080000-sparse-100 \
  --episodes 100 --num-envs 16 --episode-length 930 \
  --inference-steps 100 --device cuda --sim-device cuda:0
```

The writer stages data under a sibling `.incomplete-*` directory, reloads and
checks every raw column, then atomically publishes `dataset/` and
`collection_summary.json`. No image or video directory is created.

## Environment

Install the locked project and optional runtime dependencies in the repository
directory:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked \
  --extra dataset --extra diffusion --extra moya_newton
```

The Moya source is pinned as `third_party/moya_newton_sim`. Initialize it when
using a fresh clone:

```bash
git submodule update --init --recursive
```

Use a writable local cache on managed machines:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export HF_HOME="$PWD/.hf-datasets-cache/hf"
export HF_DATASETS_CACHE="$PWD/.hf-datasets-cache/datasets"
export HF_HUB_CACHE="$PWD/.hf-datasets-cache/hub"
```

## Dataset Inspection

```bash
uv run python -m RL.cli.inspect_dataset \
  --dataset-root /path/to/randomized_grasp_100/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /path/to/collection_summary.json
```

The supplied collection should report 100 episodes, 93,000 frames, 3,000
decision chunks, and a 94/6 positive/negative label split.

## Offline Training

Run a wiring smoke test against a standard LeRobot Diffusion checkpoint:

```bash
uv run python -m RL.cli.train_offline \
  --checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model \
  --dataset-root /path/to/randomized_grasp_100/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /path/to/collection_summary.json \
  --output-dir outputs/rl100/offline_smoke \
  --device cuda --smoke
```

A full bounded offline run is explicit:

```bash
uv run python -m RL.cli.train_offline \
  --checkpoint /path/to/pretrained_model \
  --dataset-root /path/to/randomized_grasp_100/dataset \
  --repo-id moya_newton/randomized_grasp_100 \
  --summary /path/to/collection_summary.json \
  --output-dir outputs/rl100/offline \
  --device cuda --batch-size 32 \
  --iql-steps 100000 --actor-steps 0 --old-policy-sync-target 50 --inference-steps 10 \
  --actor-lr 1e-6 --ppo-epochs 4 --probability-sigma-min 0.1 \
  --amq-enabled --dynamics-steps 10000 --amq-rollout-horizon 20 \
  --amq-eval-interval 50 \
  --old-policy-sync-interval 0 \
  --eval-every-old-policy-syncs 5 --eval-episodes 100 --eval-batch-size 16 \
  --swanlab-project moya-rl100 --swanlab-mode online
```

With `--old-policy-sync-target 50`, training is measured in completed
behavior-policy synchronizations rather than actor-loop iterations. Every five
syncs, an atomic checkpoint is written under
`<output-dir>/checkpoints/sync_005`, `sync_010`, ..., each containing a
standard LeRobot `pretrained_model/` bundle used for the headless Newton
evaluation. The final full RL bundle is at
`<output-dir>/checkpoints/final` and remains loadable by standard LeRobot tools.

The compact training metrics (`actor/loss`, `actor/ratio_mean`,
`actor/clip_fraction`, and `actor/approx_kl`) stay at the top level of each
`metrics.jsonl` row. Aggregate diagnostics are isolated under `info/actor/`:
`ratio_q05/q50/q95/max`, aggregate `delta_logprob`, event counts, old-policy
replay error, and post-update KL. Per-denoising-step `denoise_*` diagnostics
are disabled by default to keep metrics compact; pass `--debug` to
`RL.cli.train_offline` (or `--offline-debug` to the iterative orchestrator)
to emit the full per-step ratios, delta log-probabilities, and DDIM sigma
statistics.
The ratio is computed as `exp(sum(T_action x D_active delta_logprob))` for each
denoising transition; it is not an average over action events.
Rollout sampling continues to use `sigma_sample` from the DDIM schedule and
the existing sample floor. PPO likelihood alone uses
`sigma_probability=max(sigma_sample, probability_sigma_min)`, reported as
`sigma_sample_effective` and `sigma_probability` under each denoising step.
`info/actor/post_update/*` is computed after the optimizer step on the exact
stored old-policy transition, before any behavior synchronization. Thus a
nonzero KL is a real candidate/behavior difference rather than a logging
artifact. `ppo_epochs` controls how many candidate updates reuse one behavior
snapshot; when AM-Q is enabled, the snapshot is promoted only by the gate.

AM-Q is separate from the actor advantage: it uses the raw conservative
`min(Q1,Q2)` estimate on paired state-dynamics rollouts. A candidate is
promoted only when its modeled return reaches the behavior AM-Q plus the
configured relative margin (default 5%), and the
dynamics validation loss and ensemble disagreement pass their limits. Rejected
candidates leave `old_policy` unchanged; `info/amq/*` and the promotion
counters are logged to SwanLab and `metrics.jsonl`.
The default AM-Q score is the batch mean of the horizon-summed modeled Q
values used by RL-100; `--amq-discounted` is an explicit experiment-only
alternative.
`--amq-use-critic-reference` enables the optional stricter critic baseline.
For DP3, AM-Q follows RL-100's 3D path: the PointNet + dual-wrist RGB +
state observation is encoded once per history frame, and the learned ensemble
predicts the next encoded latent (plus reward/done). Imagined rollouts update
that latent history directly; raw point clouds/RGB are never hallucinated.

## Online Training (real robot)

Online PPO does **not** create a simulator. You must provide a real-robot
environment factory with:

```python
env.num_envs = 1  # or a safely vectorized number of robots
env.reset(...) -> (observation, info)
env.step(action) -> (observation, reward, terminated, truncated, info)
```

The observation must expose the policy features. For DP3 this is
`observation.state` (34D), `observation.point_cloud` `(2048, 3)`, and both
wrist RGB streams. A single Franka Duo adapter may return unbatched
`[34]`, `[2048, 3]`, and `[3, H, W]` tensors; the trainer wraps them as one
environment.

```bash
uv run python -m RL.cli.train_online \
  --checkpoint /path/to/dp3_or_diffusion_checkpoint \
  --output-dir outputs/rl100/franka_duo_online \
  --device cuda --num-envs 1 \
  --env-factory my_franka_env:make_env \
  --reward-mode terminal_success \
  --updates 1000 --rollout-decisions 30 --ppo-epochs 1 \
  --inference-steps 10 --minibatch-size 1 --actor-lr 1e-6
```

`terminal_success` keeps the RL-100 sparse contract: each active step has
reward `0`, a successful terminal step has reward `1`, and a failed timeout has
reward `0` with `truncated=True`. Set `--reward-mode environment` to accumulate
the scalars returned by `env.step()` instead (dense shaping is supported by the
online PPO buffer). The environment remains responsible for task-specific
success detection and safe robot stopping.

## Iterative Offline Loop

`RL.cli.train_iterative_offline` runs the resumable RL-100-style outer loop:

```text
highest sync checkpoint by real Newton pc_success
-> collect 100 episodes
-> merge with the accumulated IL dataset
-> warm-start IL with the previous IL weights and fixed normalizer
-> retrain IQL, dynamics, AM-Q, and the diffusion actor
```

The initial historical run used 100-episode evaluations. Each new offline run
also evaluates 100 episodes every five accepted old-policy synchronizations and
the next round scans all `sync_*` results instead of using a fixed label.

```bash
uv run python -m RL.cli.train_iterative_offline \
  --output-root outputs/rl100/iterative-moya \
  --base-dataset-root /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/dataset \
  --base-repo-id local/moya-randomized-grasp-100 \
  --base-summary /home/droid/project/Moya_newton_sim/.worktrees/feat-fused-batched-env/runs/lerobot/randomized_grasp_100_20260813-230331/collection_summary.json \
  --il-checkpoint outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model \
  --source-offline-run outputs/rl100/offline-moya-il-amq-sync50-20260819-143809 \
  --source-eval-episodes 100 \
  --canonical-task moya_charger_grasp \
  --rounds 1 --episodes 100 --num-envs 16 \
  --device cuda --sim-device cuda:0 \
  --il-steps 50000 --il-batch-size 256 \
  --offline-iql-steps 100000 --offline-dynamics-steps 10000 \
  --offline-sync-target 50 --offline-eval-every-syncs 5 \
  --offline-eval-episodes 100 --offline-eval-batch-size 16 \
  --final-rollout-merge \
  --swanlab-project moya-rl100 --swanlab-mode online
```

No rollout or evaluation video is requested. Each round writes
`round_XXX/round_manifest.json` plus separate `rollout/`, `merge/`, `il/`, and
`offline/` attempt directories. Resume reuses a stage only after its artifacts,
source hashes, dependency hashes, checkpoint manifest, and offline provenance
still match. A replaced rollout therefore forces merge, IL, and offline RL to
run again. `--dry-run` records the first pending stage without executing it;
repeating the same dry-run is idempotent. `--smoke` bounds the workflow and
uses direct one-update policy synchronization instead of waiting on an AM-Q
promotion gate. `--final-rollout-merge` adds one final dataset-only round: it
selects the highest measured `sync_*` checkpoint from the newly completed
offline run, collects another 100 episodes, merges them into the cumulative
dataset, and stops before another IL/offline stage. With a 100-episode base and
one full round, this publishes the requested 300-episode dataset under
`round_002/merge/`.

## Checkpoints And Resumption

Each RL checkpoint contains:

```text
pretrained_model/     standard LeRobot model, processors, and statistics
rl_state.pt            policy/critic/optimizer/RNG state
rl_config.json         resolved dimensions and schedule
provenance.json        source and dataset hashes
metrics.jsonl          finite scalar metrics
manifest.json          file hashes and completeness marker
```

The online CLI resumes an online checkpoint into a new output directory when
all immutable `RLConfig` fields match:

```bash
uv run python -m RL.cli.train_online \
  --checkpoint outputs/rl100/online/checkpoints/final \
  --output-dir outputs/rl100/online-resumed \
  --device cuda --env-device cuda:0 --num-envs 1 \
  --env-factory my_franka_env:make_env \
  --reward-mode terminal_success \
  --updates 1000 --rollout-decisions 30 --ppo-epochs 4 \
  --gamma 0.99 --inference-steps 10
```

For programmatic or offline-stage resumption, use
`RL.checkpointing.load_rl_checkpoint` and `restore_rl_state`. Loading validates
processor fingerprints, dimensions, action masks, provenance commits,
optimizer structure, sampler state, and file hashes before state is applied.

## Standard Evaluation

The standard LeRobot evaluator consumes the nested policy bundle, not the RL
metadata directory:

```bash
uv run lerobot-eval \
  --policy.path=outputs/rl100/online_smoke/checkpoints/final/pretrained_model \
  --env.type=moya_newton \
  --env.device=cuda:0 --eval.batch_size=16 \
  --eval.n_episodes=16 --eval.use_async_envs=false \
  --policy.device=cuda
```

This environment is state-only, headless, and video-free by design.

## Limitations

Flow policies, distillation stages, MuJoCo runners, and robot-specific drivers
remain out of scope. DP3/PointNet is integrated through the shared diffusion
adapter. DP3 offline AM-Q uses a multimodal latent critic and
`DP3FeatureDynamicsEnsemble` (latent predictions; raw point clouds/RGB are
carried from the initial observation because no visual decoder is learned).
State-only policies continue to use the lightweight state critic/dynamics path.
