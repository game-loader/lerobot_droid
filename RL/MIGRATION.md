# RL-100 Migration Notes

## LeRobot 0.6 Upstream Rebase

The custom `RL/` package coexists with upstream `src/lerobot/rl/`; the upstream
SAC implementation is not replaced. Warmstart training uses the upstream
`env_eval_freq` setting and the custom checkpoint-normalization preservation hook.
SmolVLA prefix conditioning now uses native Transformers `DynamicCache` tensors
in `[batch, heads, tokens, head_dim]` layout. The stored RL condition tensors retain
their prior token-major layout, so this adapter change does not remap actions or
change the v2 RL checkpoint format.

Current DP3 uses 34D state and 20D action. Historical 17D recorder actions and 18D
DP3 checkpoints are different contracts, not interchangeable representations;
they require an explicitly validated data/checkpoint conversion before use.
The RL adapter reads dimensions from the loaded checkpoint and does not silently
pad, truncate, or reinterpret those physical action channels.

RL dataset adapters currently target local Parquet-backed LeRobot v3 datasets.
They use the upstream `hf_dataset` compatibility property and do not yet support
the new LanceDB/storage-bucket reader backends. Migration validation covers the
offline unit suite, including real upstream SmolVLA cache/attention code with
tiny random layers; it does not establish full-size checkpoint, training-quality,
simulator, or real-robot equivalence.

## Algorithm Mapping

The implementation is a focused rewrite for LeRobot rather than a wholesale
copy of the RL-100 workspace.

| RL-100 concept | This repository | Adaptation |
| --- | --- | --- |
| `unidpg/critic.py` IQL Q/V | `RL/algorithms/iql.py`, `RL/policy/observation_encoder.py` | MLP state encoder for standard policies; DP3 uses a frozen multimodal DP3 feature encoder (PointNet + two RGB ResNet18 + state) with masked action chunks and checkpoint-derived dimensions (currently 34D/20D). |
| `unidpg/uni_ppo.py` | `RL/trainers/offline.py`, `RL/trainers/online.py` | LeRobot Diffusion Policy modules and atomic optimizer updates. |
| Diffusion log-prob patch | `RL/policy/ddim.py`, `RL/policy/diffusion_adapter.py` | Private stochastic DDIM schedule over the existing DDPM-trained U-Net; standard LeRobot inference is unchanged. |
| Online replay buffers | `RL/trainers/online.py` | Decision-major/environment-minor vector storage, partial-chunk masks, terminal-aware next states, and per-decision discounts. |
| Dynamics model | `RL/algorithms/dynamics.py` | State-history ensemble for standard policies; `DP3FeatureDynamicsEnsemble` predicts RL-100-style encoded DP3 latent history (PointNet + dual-wrist RGB + state) for multimodal AM-Q. |
| Point-cloud observation path | `src/lerobot/policies/dp3`, `RL/policy/diffusion_adapter.py` | DP3 checkpoints are dispatched by policy type; actor traces consume state, point cloud, and dual-wrist RGB histories. DP3 IQL/AM-Q critics consume the same encoded multimodal latent history. |
| Environment runners | `RL/cli/train_online.py`, user `--env-factory` | Online PPO requires a caller-provided Gymnasium-compatible real-robot environment; no simulator is created implicitly. |
| Episode terminal handling | `RL/adapters/moya_newton.py` | Gymnasium `SAME_STEP` reset observations are never used as terminal next states; `final_obs` and `final_info["is_success"]` are authoritative. |
| Reward | `RL/adapters/lerobot_v3.py`, `RL/collectors/moya_il.py`, Moya wrapper | Only the terminal decision receives a binary reward. Success requires the five acceptance conditions and a 15 mm lift compared in float32 precision. Missing sparse grasp/clear-table event keys are zero; malformed present values are rejected. |
| Action representation | `RL/algorithms/iql.py`, `RL/algorithms/ppo.py` | The full policy action vector remains in the actor/environment contract; constant dimensions are masked from critic and likelihood reductions using checkpoint ranges. |
| Workspace stages | `RL/cli/*.py` | Python 3.12 and `uv`; no Hydra launcher or vendored package assumptions. |
| Checkpoint output | `RL/checkpointing.py` | Standard LeRobot `pretrained_model` plus hashed RL state, provenance, metrics, and atomic publication. |

## Deliberate Exclusions

The original RL-100 native flow-policy classes, one-step distillation, and old
MuJoCo runners remain out of scope. A separate SmolVLA offline flow-RL path now
reuses IQL objectives, the clipped PPO objective and the promotion gate, with a
constant-diffusion reverse SDE and one frozen token/K/V representation. Compact
query-Transformer Q/V and prefix-token dynamics replace the initial SmolVLA
MLP/KV heads (v2 training states are incompatible with v1; IL remains usable). This is not
a wholesale port of the upstream Flow scheduler or a reproduction of its
performance results. See [SmolVLA offline RL](smolvla/README.md).
DP3/PointNet remains supported through the shared LeRobot Diffusion adapter;
robot-specific drivers and task reward evaluators are provided by the caller.

## Moya IL Collection Contract

`RL.cli.collect_moya_il` runs the saved Diffusion Policy in the synchronous
CUDA Moya environment and stores raw pre-action `(39,)` states plus the exact
postprocessed `(14,)` actions passed to `env.step`. The final frame is always
`next.done=True`; successful episodes use `(reward=1, truncated=False)` and
failed horizons use `(reward=0, truncated=True)`. The summary is required for
canonical datasets and must agree with all five terminal conditions. Dataset
fps is fixed at 60 and the writer never records video.

## Provenance

The algorithm reference is RL-100 source commit
`7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26`, reviewed against checkout
`3d52f73be4a1f7c27ed3bb32280adb36428f57e5`. The checkpoint metadata records
both values and rejects unreviewed replacements.
