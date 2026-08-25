# Online RL from sync_165

## Input

| item | value |
|---|---|
| checkpoint | `outputs/rl100/iterative-moya-il7-20260824-000250/round_001/offline/attempt_001/run/checkpoints/sync_165/pretrained_model` |
| offline evaluation | 91/100 (`pc_success=91.0`) |
| policy input | state-only `observation.state`, 39 dimensions |
| action | 14 dimensions, Moya `[-1, 1]` environment Box |
| simulator | Newton CUDA `cuda:0`, 16 vector worlds, headless, no video |

## Online configuration

```text
rollout_decisions=30
episode_length=930
inference_steps=10
updates=202
ppo_epochs=1
minibatch_size=64
actor_lr=1e-6
value_lr=3e-4
probability_sigma_min=0.1
gamma=0.99
gae_lambda=0.95
seed=800000
```

Each update covers `30 * 32 = 960` action slots and reaches the 930-step
terminal boundary for every world. The target is approximately 3M environment
steps (`202 * 16 * 930 = 3,005,760` before early terminal padding).

## Validation

- CUDA full-episode smoke with 2 denoising steps: 14,880 environment steps,
  16 terminal transitions, no action-contract failure.
- CUDA full-episode smoke with the production 10 denoising steps: 14,880
  environment steps, success label rate `0.0270833`, PPO approx KL
  `0.0066768`, ratio q05/q95 `0.8173/1.1809`, ratio max `2.0702`.
- The online trainer now projects only the command sent to a finite Box after
  unnormalization; latent DDIM traces and PPO likelihoods remain unchanged.
- Focused online trainer tests: 14 passed; Ruff and `git diff --check` passed.

## Formal run

```text
output: outputs/rl100/online-moya-sync165-3m-20260824-2002
pid: 811377 (uv parent 811358)
swanlab: https://swanlab.cn/@game-loader/moya-rl100/runs/4t9jybfx
launch: 2026-08-24 20:03 (Asia/Shanghai)
```

At the first monitor point, 4/202 updates had completed with 59,520 valid
environment steps. The latest success rate was `0.025`, approx KL `0.0026796`,
ratio q05/q95 `0.8916/1.1085`, and ratio max `1.6661`.
