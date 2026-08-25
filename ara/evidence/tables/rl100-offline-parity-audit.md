# RL-100 Offline Algorithm Parity Audit

## Paper contract

Source: RL-100, arXiv:2510.14830, Iterative Offline RL section.

- The behavior policy remains the denominator of the clipped PPO ratio while a
  candidate receives several epochs of gradient updates.
- Actor advantage is the IQL estimate `Q_hat(s,a) - V_hat(s)`.
- Approximate model-Q (AM-Q) is a distinct offline-policy-evaluation gate. A
  learned transition model rolls out both candidate and behavior policies and
  scores their modeled states/actions with `Q_hat`.
- The paper advances the behavior policy only when candidate AM-Q improvement
  reaches `0.05 * abs(behavior AM-Q)`.

## Original repository behavior

Sources:

- `/home/droid/project/RL-100/RL-100/train.py:1040`
- `/home/droid/project/RL-100/RL-100/train.py:1120`
- `/home/droid/project/RL-100/RL-100/train.py:1221`
- `/home/droid/project/RL-100/RL-100/rl_100/unidpg/uni_ppo.py:763`
- `/home/droid/project/RL-100/RL-100/rl_100/unidpg/dynamics_eval_batch.py:27`

The repository snapshots the old policy once at offline-stage entry, updates
the candidate against old-policy denoising traces, evaluates modeled rollout Q
every configured `eval_step` (default 50), and promotes the old policy only
when the current modeled Q exceeds the previous best. This code gate is less
strict than the paper equation because it compares `current_mean_qs >
best_mean_qs` without the explicit five-percent margin.

## Current migration behavior

Sources:

- `RL/cli/train_offline.py:245`
- `RL/trainers/offline.py:220`
- `RL/trainers/offline.py:284`
- `RL/algorithms/iql.py:359`
- `RL/algorithms/dynamics.py:397`

The migration correctly computes a double-Q IQL advantage and replays the same
stored denoising transition under old and current policies. However, it performs
one actor optimizer step per batch and unconditionally copies current policy to
old policy after every step. The CLI does not instantiate the available state
dynamics ensemble or promotion gate. There is no model rollout scorer, AM-Q
comparison, validation/disagreement gate, or conditional behavior-policy
promotion in the executed training path.

Therefore the stopped run was an IQL-advantage diffusion policy-gradient loop,
not the paper's complete AM-Q-gated offline policy-improvement loop.
