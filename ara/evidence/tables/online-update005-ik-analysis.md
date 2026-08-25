# Online PPO Update-005 Metric and IK Analysis

## Metric semantics

Source: `outputs/rl100/online-moya-sync165-3m-20260824-2002/metrics.jsonl`, line 5:

```text
"rollout/environment_steps":14880.0,"rollout/reward_mean":0.02708333544433117,"rollout/success_rate":0.02708333544433117,"rollout/terminal_count":16.0
```

The run used 16 environments and 30 action decisions per 930-step episode (`30 * 32 = 960` action steps). The trainer computes `rollout/success_rate` as `rollout.success.float().mean()` in `RL/trainers/online.py:1132`; the success bit is nonzero only on a terminal transition (`RL/trainers/online.py:827-829`). Thus update 5 is `13 / (16 * 30) = 0.0270833`, corresponding to `13 / 16 = 81.25%` episode success. The persisted online run has a maximum raw value of `0.02916666865348816` at other updates, so update 5 is not the run maximum.

## Checkpoint provenance

The original online CLI saves only `checkpoints/final` (`RL/cli/train_online.py:343-345`). The update-005 artifact was reconstructed by replaying the first five updates from the offline `sync_165` checkpoint:

```text
outputs/rl100/online-moya-sync165-update005-replay/checkpoints/final/pretrained_model
```

The reconstructed model SHA256 is `5bb96b816a1b31234f147d28441c5c125f5cd21b6b1a71c1b73dd708ead521ba`. CUDA Newton contact solving is not bitwise deterministic, so this is a same-configuration replay checkpoint, not a byte-for-byte recovery of an unsaved original checkpoint.

## Rollout results

First CUDA rollout (100 episodes, 16 worlds, 930 steps, 10 DDIM steps, probability sigma floor 0.1):

```text
outputs/rl100/online-moya-sync165-update005-analysis-100/summary.json
episode_count=100, successes=81, failures=19, success_rate_percent=81.0
failure_with_2mm_ik_reachable=19, failure_with_2mm_ik_unreachable=0
```

Independent rerun with the same configuration:

```text
outputs/rl100/online-moya-sync165-update005-analysis-100-rerun/summary.json
episode_count=100, successes=85, failures=15, success_rate_percent=85.0
failure_with_2mm_ik_reachable=15, failure_with_2mm_ik_unreachable=0
```

The 81%/85% difference is consistent with the observed CUDA Newton contact-solver nondeterminism. Across both 200 episodes, 166 succeeded and all 34 failures were statically position-IK reachable.

## Failure breakdown

Counts overlap because an episode can violate several terminal criteria. Across the first 100 episodes the counts were:

```text
clear_table_never_established: 13
final_lift_below_15mm: 11
true_grasp_never_established: 11
final_table_contact: 6
no_final_hand_contact: 1
```

Several failures lifted the charger 31--54 mm while `true_grasp_ever` remained false, indicating pushing/contact rather than a stable grasp. Other failures established contact and temporarily lifted the charger but ended below the 15 mm terminal threshold or recontacted the table.

## Static and dynamic IK checks

The initial grasp-reference targets of all 19 first-run failures had native LM wrist FK residuals below 2 mm (median `1.31e-6 m`, maximum `2.98e-6 m`). Failure initial positions were inside the successful position range rather than outside it: failures covered approximately `x=0.3565..0.3830 m`, `y=-0.0143..0.0150 m`, while successes covered `x=0.3551..0.3842 m`, `y=-0.0147..0.0147 m`.

An additional probe sampled 1,122 dynamic palm targets from all 34 failed trajectories (every 30 simulation steps plus each trajectory's maximum tracking-residual step). It found 1 sample at `2.00089e-3 m`, 1,121 samples below 2 mm, and no sample above 5 mm. The borderline sample is a solver-tolerance edge case, not evidence that the whole failure set lies outside the arm workspace.

Conclusion: the evidence does not support initial charger positions being outside the arm's IK workspace as the dominant failure cause. The stronger signal is grasp establishment/stability and execution tracking under contact; joint-limit margins are not discriminative because successful trajectories also frequently reach the normalized limit boundary.
