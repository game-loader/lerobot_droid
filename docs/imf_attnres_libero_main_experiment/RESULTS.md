# IMF-AttnRes LIBERO Main Results
Generated: 2026-05-17 07:30:42

## Protocol
- Four LIBERO suites: Spatial (`libero_spatial`), Object (`libero_object`), Goal (`libero_goal`), Long (`libero_10`).
- Train 20k steps per suite; checkpoints every 5k.
- Periodic checkpoint selection uses `eval.n_episodes=10`; in LeRobot LIBERO this is 10 rollouts per task = 100 rollouts per suite.
- Final metric uses the selected checkpoint with `eval.n_episodes=50`; this is 50 rollouts per task = 500 rollouts per suite.
- Metric is `overall.pc_success` from LeRobot `lerobot-eval`. W&B logging was online; checkpoint artifacts were disabled.

## SmolVLA-style LIBERO table
| Method | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| IMF-AttnRes | 52.4 | 0.0 | 6.4 | 4.4 | 15.8 |

## Checkpoint selection details
| Suite | Periodic rollout10 success by checkpoint | Selected step | Selected rollout10 success | Final rollout50 success | Final rollouts |
|---|---|---:|---:|---:|---:|
| Spatial | 5000:22.0, 10000:37.0, 15000:34.0, 20000:46.0 | 20000 | 46.0 | 52.4 | 500 |
| Object | 5000:0.0, 10000:0.0, 15000:0.0, 20000:0.0 | 20000 | 0.0 | 0.0 | 500 |
| Goal | 5000:7.0, 10000:3.0, 15000:5.0, 20000:7.0 | 20000 | 7.0 | 6.4 | 500 |
| Long | 5000:0.0, 10000:0.0, 15000:0.0, 20000:3.0 | 20000 | 3.0 | 4.4 | 500 |

## Artifact pointers
- Spatial: run dir `/data/lerobot-imf-attnres-exp/runs/imf-attnres-libero-spatial-s20000-eval5000`, final eval dir `/data/lerobot-imf-attnres-exp/outputs/final_eval/imf-attnres-libero-spatial-s20000-eval5000/step_020000`
- Object: run dir `/data/lerobot-imf-attnres-exp/runs/imf-attnres-libero-object-s20000-eval5000`, final eval dir `/data/lerobot-imf-attnres-exp/outputs/final_eval/imf-attnres-libero-object-s20000-eval5000/step_020000`
- Goal: run dir `/data/lerobot-imf-attnres-exp/remote_results/imf-attnres-libero-goal-s20000-eval5000`, final eval dir `/data/lerobot-imf-attnres-exp/outputs/final_eval/imf-attnres-libero-goal-s20000-eval5000/step_020000`
- Long: run dir `/data/lerobot-imf-attnres-exp/remote_results/imf-attnres-libero-long-s20000-eval5000`, final eval dir `/data/lerobot-imf-attnres-exp/outputs/final_eval/imf-attnres-libero-long-s20000-eval5000/step_020000`
