# RL-100 Migration Notes

The implementation is a focused rewrite for LeRobot rather than a wholesale
copy of the RL-100 workspace.

| RL-100 concept | This repository | Adaptation |
| --- | --- | --- |
| `unidpg/critic.py` IQL Q/V | `RL/algorithms/iql.py` | MLP state encoder; named observation dictionaries; masked action chunks; sparse terminal targets. |
| `unidpg/uni_ppo.py` | `RL/trainers/offline.py`, `RL/trainers/online.py` | LeRobot Diffusion Policy modules and atomic optimizer updates. |
| Diffusion log-prob patch | `RL/policy/ddim.py`, `RL/policy/diffusion_adapter.py` | Private stochastic DDIM schedule over the existing DDPM-trained U-Net; standard LeRobot inference is unchanged. |
| Online replay buffers | `RL/trainers/online.py` | Decision-major/environment-minor vector storage, partial-chunk masks, terminal-aware next states, and per-decision discounts. |
| Dynamics model | `RL/algorithms/dynamics.py` | State-history dynamics ensemble with explicit image-encoder boundary. |
| Point-cloud observation path | `RL/types.py`, `RL/policy/observation_encoder.py` | `observation.state` is complete for the 39D Moya task; image keys are retained and fail explicitly without an encoder. |
| Environment runners | `RL/adapters/moya_newton.py` | Public `MoyaNewtonEnvConfig` and `make_env(..., use_async_envs=False)`; fused CUDA batching is preserved. |
| Episode terminal handling | `RL/adapters/moya_newton.py` | Gymnasium `SAME_STEP` reset observations are never used as terminal next states; `final_obs` and `final_info["is_success"]` are authoritative. |
| Reward | `RL/adapters/lerobot_v3.py`, Moya wrapper | Only the terminal decision receives a binary reward. Success requires the five acceptance conditions and a 15 mm lift. |
| Action representation | `RL/algorithms/iql.py`, `RL/algorithms/ppo.py` | Full 14D actions remain in the actor/environment contract; constant dimensions `3:12` are masked from critic and likelihood reductions. |
| Workspace stages | `RL/cli/*.py` | Python 3.12 and `uv`; no Hydra launcher or vendored package assumptions. |
| Checkpoint output | `RL/checkpointing.py` | Standard LeRobot `pretrained_model` plus hashed RL state, provenance, metrics, and atomic publication. |

## Deliberate Exclusions

The DP3/PointNet encoder, fixed `point_cloud` schema, flow policy, one-step
distillation, old MuJoCo runners, real-robot drivers, and automatic image
critic are not migrated. These assumptions do not match the current state
checkpoint and would obscure the shared LeRobot observation contract.

## Provenance

The algorithm reference is RL-100 source commit
`7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26`, reviewed against checkout
`3d52f73be4a1f7c27ed3bb32280adb36428f57e5`. The checkpoint metadata records
both values and rejects unreviewed replacements.
