# RL-100 Diffusion RL Migration

This directory contains the state-first adaptation of the offline and online
reinforcement-learning stages from RL-100 for the LeRobot Diffusion Policy
checkpoint used by the Moya charger-grasp task. The implementation keeps the
LeRobot policy, processor, dataset, and environment contracts as the source of
truth.

## Scope

- State critics and dynamics use `observation.state` with the current 39D
  Moya observation and 14D action contract.
- Offline training uses sparse terminal labels from the LeRobot v3 collection
  summary, with success equal to `1` and failure equal to `0`.
- Online training uses the fused Moya Newton vector environment, stochastic
  DDIM traces, vector GAE, and masked denoising PPO.
- Action dimensions `3:12` are constant in the supplied checkpoint and are
  excluded from critic and PPO likelihood reductions.
- Image tensors are preserved by the data and actor paths. An image critic
  encoder is an explicit extension point and is not silently inferred.
- Moya evaluation is headless and never records video.

The acceptance rule is shared by dataset and simulator paths:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height_m >= 0.015
and final_table_contacts == 0
and final_hand_contacts > 0
```

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
  --iql-steps 100000 --actor-steps 300000 --inference-steps 10
```

The output is an atomic RL bundle at `<output-dir>/checkpoints/final`; its
`pretrained_model/` remains loadable by standard LeRobot tools.

## Online Training

Online collection and PPO require a CUDA-capable Newton installation:

```bash
uv run python -m RL.cli.train_online \
  --checkpoint outputs/rl100/offline_smoke/checkpoints/final \
  --output-dir outputs/rl100/online_smoke \
  --device cuda --sim-device cuda:0 \
  --num-envs 16 --smoke
```

Smoke mode fixes a short one-decision rollout, two DDIM steps, one PPO epoch,
headless execution, and no video output. The online buffer stores decisions in
`[time, environment]` order. A partial action chunk uses a contiguous validity
mask and a discount of `gamma ** executed_steps`.

A longer run can record metrics to SwanLab when the package is installed in
the active uv environment:

```bash
uv run python -m RL.cli.train_online \
  --checkpoint outputs/rl100/offline/checkpoints/final \
  --output-dir outputs/rl100/online \
  --device cuda --sim-device cuda:0 --num-envs 16 \
  --updates 1000 --rollout-decisions 4 --ppo-epochs 4 \
  --inference-steps 10 --minibatch-size 32 \
  --swanlab-project moya-rl100 --swanlab-run-name online-ppo
```

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
  --device cuda --sim-device cuda:0 --num-envs 16 \
  --updates 1000 --rollout-decisions 4 --ppo-epochs 4 \
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

The migration does not copy DP3/PointNet, point-cloud schemas, flow policies,
distillation stages, MuJoCo runners, real-robot drivers, or automatic image
critic features from RL-100. A concrete image feature encoder can be added
behind `ObservationFeatureEncoder` without changing the trainer contracts.
