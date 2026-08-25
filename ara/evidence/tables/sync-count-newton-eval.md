# Sync-Count Newton Evaluation

Run: `outputs/rl100/offline-moya-il-amq-sync50-20260819-143809`

SwanLab: `https://swanlab.cn/@game-loader/moya-rl100/runs/jnl2aa1a`

The actor phase stopped at 50 completed behavior-policy synchronizations. A
loadable LeRobot policy bundle was saved and evaluated for 100 headless Newton
episodes every five synchronizations.

| Old-policy sync count | Actor updates | Successes | Success rate |
|---:|---:|---:|---:|
| 5 | 4,250 | 56 | 56% |
| 10 | 6,050 | 65 | 65% |
| 15 | 8,800 | 59 | 59% |
| 20 | 10,150 | 48 | 48% |
| 25 | 11,900 | 60 | 60% |
| 30 | 13,850 | 57 | 57% |
| 35 | 16,250 | 52 | 52% |
| 40 | 19,550 | 54 | 54% |
| 45 | 22,100 | 52 | 52% |
| 50 | 24,250 | 43 | 43% |

Final checkpoint counters:

- IQL updates: 100,000
- Dynamics updates: 34,250, including the 10,000-update warm-up
- Actor updates: 24,250
- AM-Q promotion attempts: 485
- Promotions / old-policy synchronizations: 50

Sources:

- `outputs/rl100/offline-moya-il-amq-sync50-20260819-143809/metrics.jsonl`
- `outputs/rl100/offline-moya-il-amq-sync50-20260819-143809/eval/sync_*/eval_info.json`
- `outputs/rl100/offline-moya-il-amq-sync50-20260819-143809/checkpoints/final/rl_state.pt`
