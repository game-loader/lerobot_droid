# IMF-AttnRes LIBERO 4-suite Experiment Plan

## Goal
Produce a SmolVLA Table-2-aligned LIBERO result row for `imf-attnres`: Spatial, Object, Goal, Long, and Average success rate.

## Protocol
- Dataset root: `/data/lerobot_datasets/HuggingFaceVLA/libero`.
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Training data: suite-specific physical LeRobot subsets materialized under `/data/lerobot-imf-attnres-exp/datasets/libero_<suite>`.
- Training budget: 20,000 steps per suite.
- Checkpoints: every 5,000 steps.
- Selection metric: periodic rollout with `eval.n_episodes=10`; in LeRobot LIBERO this means 10 trials per task, so 100 rollouts per suite.
- Final metric: best periodic checkpoint by `pc_success`, re-evaluated with `eval.n_episodes=50`; in LeRobot LIBERO this means 50 trials per task, so 500 rollouts per suite.
- Primary metric: `overall.pc_success` from `eval_info.json` / wrapper `final_result.json`.
- W&B: online sync enabled, checkpoint artifact upload disabled (`wandb.disable_artifact=true`).

## Model/training parameters
- `policy.type=imf-attnres`
- `policy.n_obs_steps=2`, `policy.horizon=16`, `policy.n_action_steps=8`
- `policy.n_layer=4`, `policy.n_emb=256`, `policy.n_head=1`, `policy.n_kv_head=1`
- `policy.backbone_type=attnres_full`
- `policy.vision_backbone=resnet18`, pretrained weights disabled
- `policy.resize_shape=[128,128]`
- `policy.spatial_softmax_num_keypoints=32`
- `policy.use_separate_rgb_encoder_per_camera=true`
- `policy.num_inference_steps=1`
- Optimizer: lr `1e-4`, weight decay `1e-6`
- Batch size: 32, workers: 4

## Resource assignment
- Local `droid-z790eagleax` RTX 5090: `spatial` then `object` sequentially.
- Remote `droid@100.73.14.65` dual RTX 5880 Ada: `goal` on GPU0 and `long` on GPU1 in parallel.

## Artifact locations
- Repo: `/data/lerobot-imf-attnres-exp/lerobot-imf-attnres` locally and on remote.
- Run logs: `/data/lerobot-imf-attnres-exp/runs/imf-attnres-libero-<suite>-s20000-eval5000`.
- Training outputs: `/data/lerobot-imf-attnres-exp/outputs/train/imf-attnres-libero-<suite>-s20000-eval5000`.
- Periodic eval outputs: `/data/lerobot-imf-attnres-exp/outputs/eval/...`.
- Final eval outputs: `/data/lerobot-imf-attnres-exp/outputs/final_eval/...`.

## Acceptance
The experiment is complete only when all four suites have `final_result.json` with `final_eval.ok=true`, then a table is produced with Spatial/Object/Goal/Long/Avg.
