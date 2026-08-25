# Iterative IL4 launch

Recorded on 2026-08-21 (Asia/Shanghai).

## Retrospective IL checkpoint evaluation

Source:
`outputs/rl100/iterative-moya-il3-20260820-150000/round_001/il/attempt_001/train/eval`

Each checkpoint used 100 headless Moya Newton episodes, batch size 16,
10 diffusion inference steps, policy device `cuda`, environment device
`cuda:0`, and seed 202000.

| IL step | pc_success |
| ---: | ---: |
| 5000 | 54% |
| 10000 | 55% |
| 15000 | 54% |
| 20000 | 61% |
| 25000 | 62% |
| 30000 | 58% |
| 35000 | 50% |
| 40000 | 50% |
| 45000 | 60% |
| 50000 | 63% |

Selected checkpoint:
`outputs/rl100/iterative-moya-il3-20260820-150000/round_001/il/attempt_001/train/checkpoints/050000/pretrained_model`

Selection summary:
`outputs/rl100/iterative-moya-il3-20260820-150000/round_001/il_eval_retrofit/summary.json`

## Next-cycle launch

Run root:
`outputs/rl100/iterative-moya-il4-20260821-211000`

The source offline run selected `sync_035`, measured at 68% success over
100 Newton episodes. Its production collection saved 100 episodes with
70 successes and 30 failures. The canonical merge contains 400 episodes:
302 successes and 98 failures.

The fixed-normalizer IL stage warm-started from the retrospective best
checkpoint. Its first in-training evaluation completed at step 5000 with
60 successes in 100 episodes. The evaluation saved no video and wrote an
atomic provenance sidecar binding checkpoint SHA256, seed 103000, batch size
16, 10 diffusion inference steps, `cuda` policy device, and `cuda:0` Newton
device.

SwanLab:
https://swanlab.cn/@game-loader/moya-rl100/runs/ago83oi3

Primary artifacts:

- `round_001/rollout/attempt_001/collection/collection_summary.json`
- `round_001/merge/attempt_001/merged/collection_summary.json`
- `round_001/il/attempt_001/train/eval/step_005000/eval_info.json`
- `round_001/il/attempt_001/train/eval/step_005000/eval_provenance.json`
