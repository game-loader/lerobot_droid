# Iterative IL6 launch

Recorded on 2026-08-23 (Asia/Shanghai).

## Completed IL5 offline stage

Source run:
`outputs/rl100/iterative-moya-il5-20260822-225800/round_001/offline/attempt_001/run`

The completed AM-Q offline stage used a target of 100 behavior-policy
promotions and evaluated every five promotions with 100 headless CUDA Newton
episodes. The best evaluated checkpoint was `sync_005`, with 74 successful
episodes of 100. The final `sync_100` evaluation measured 66 successes of 100.

The source IL checkpoint for that stage was `step_015000`, which had the
highest IL Newton success at 72% among the 5000-step checkpoints.

## IL6 launch

Run root:
`outputs/rl100/iterative-moya-il6-20260823-101000`

IL6 starts from the completed IL5 500-episode canonical merged dataset and
the IL5 `step_015000` fixed-normalizer checkpoint. The collection source is
the IL5 offline `sync_005` checkpoint (74/100 evaluation success). The new
offline stage is configured with:

- `old_policy_sync_interval=0` (AM-Q controls promotion)
- `old_policy_sync_target=150`
- 100 Newton episodes every five promotions
- 100000 IQL updates and 10000 dynamics updates
- 10 diffusion inference steps and `probability_sigma_min=0.1`

At record time, the initial 100-episode CUDA rollout from `sync_005` was
active with collection seed 106000.

## IL6 progress

The initial rollout and canonical merge completed before IL warm-start. The
merged dataset summary records `episodes_saved: 600`, `success_count: 447`,
and `failure_count: 153`.

The fixed-normalizer IL stage is active from the IL5 `step_015000` checkpoint.
Its first saved checkpoint (`step_005000`) completed a 100-episode headless
CUDA Newton evaluation with `pc_success: 71.0`; no video paths were emitted.
The downstream offline stage has not started yet and remains locked to the
150-promotion target in `round_manifest.json`.
