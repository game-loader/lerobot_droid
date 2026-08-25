# sync_165 evaluation under the latched-success Moya environment

## Checkpoint

- Path: `outputs/rl100/iterative-moya-il7-20260824-000250/round_001/offline/attempt_001/run/checkpoints/sync_165/pretrained_model`
- `model.safetensors` SHA-256: `a724b015d533b0b5a28ab757aac8c201cfe80f977ada37ae5ab40ff0aba35ebe`

## Matched evaluator runs

All runs used the standard `lerobot.scripts.lerobot_eval` path, 100 episodes, batch size 16, `seed=800000`, 10 inference steps, `cuda` policy and `cuda:0` Moya simulation, and a 930-step horizon.

| Environment | Result | Artifact |
|---|---:|---|
| Historical offline `sync_165` eval | 91/100 (91%) | `outputs/rl100/iterative-moya-il7-20260824-000250/round_001/offline/attempt_001/run/eval/sync_165/eval_info.json` |
| Unmodified submodule archived at `7dd6af2952e1153d8943bc7b255e877b96ed0e55` (temporary `/tmp` copy) | 88/100 (88%) | `outputs/rl100/sync165-old-env-standard-eval-seed800000/eval_info.json` |
| Current submodule with latched success and immediate termination | 86/100 (86%) | `outputs/rl100/sync165-new-env-standard-eval-seed800000/eval_info.json` |

The current-environment evaluator reported:

```json
{"avg_sum_reward": 3166.96698677063, "avg_max_reward": 986.9823450851441, "pc_success": 86.0, "n_episodes": 100}
```

The temporary old-environment reproduction reported:

```json
{"avg_sum_reward": 23334.42574584961, "avg_max_reward": 1027.9947934055328, "pc_success": 88.0, "n_episodes": 100}
```

The historical artifact reported `pc_success=91.0` for 100 episodes. Its `eval_info.json` does not persist the seed; `offline.log` records the evaluator configuration with `seed=800000`.

## Interpretation

- Relative to the historical record, the new environment is 5 percentage points lower (91% -> 86%).
- A matched reproduction with the archived old submodule is 88%, so the observed single-run semantic delta is 2 points (88% -> 86%); the remaining difference from 91% is run-to-run simulator/policy rollout variation.
- Immediate termination makes successful episodes shorter, so `avg_sum_reward` is not comparable across old and new semantics. The binary `pc_success` is the intended comparison metric.
- A separate diagnostic rollout with the new environment (`outputs/rl100/sync165-new-env-analysis-seed800000/summary.json`) found 85/100 and all 15 failures IK-reachable at the 2 mm probe threshold; this is a diagnostic stochastic-DDIM path, not the canonical evaluator result.
