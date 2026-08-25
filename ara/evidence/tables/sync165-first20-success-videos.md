# IL7 `sync_165` first-20 success videos

## Inputs

| Field | Value | Source |
|---|---:|---|
| Checkpoint | `outputs/rl100/iterative-moya-il7-20260824-000250/round_001/offline/attempt_001/run/checkpoints/sync_165/pretrained_model` | `outputs/eval/moya_rl100_il7_sync165_first20_success_20260824-180454/summary.json:12` |
| Model SHA-256 | `a724b015d533b0b5a28ab757aac8c201cfe80f977ada37ae5ab40ff0aba35ebe` | `summary.json:47` |
| Evaluation seed / batch | `800000` / `16` | `summary.json:53-72` |
| Episode length / FPS | `930` / `60` | `summary.json:15-16` |
| Inference steps | `10` | `summary.json:46` |
| Devices | policy `cuda`; simulation `cuda:0` | `summary.json:49`, `summary.json:1057` |

## Fresh result

| Metric | Result | Source |
|---|---:|---|
| Episodes | `100` | `outputs/eval/.../eval_info.json:7` |
| Successes | `83` | `eval_info.json:8` / `summary.json:43-44` |
| First 20 successful global indices | `0,2,3,4,5,6,7,8,9,10,13,14,15,16,17,19,20,21,22,23` | `summary.json:20-40` |
| Selection matched fresh vector | `true` | `summary.json:20-40`, `summary.json:1035-1059` |

## Video validation

All 20 retained files are in `outputs/eval/moya_rl100_il7_sync165_first20_success_20260824-180454/videos/`.

- Every file has `931` frames, `1280x720`, `60/1`, H.264, `yuv420p`, and duration `15.516667 s` (`ffprobe` verification).
- Every file has a distinct first/middle/final frame digest (`framemd5` verification).
- Every video passed the accepted terminal criterion: `is_success=true`, `true_grasp_ever=true`, `clear_table_ever=true`, final lift at least `0.015 m`, zero table contacts, and positive hand contacts (`summary.json` video records).
- Initial charger positions vary across the retained set: X `0.3567156..0.3789069 m`; Y `-0.0120370..0.0140516 m` (`summary.json` video records).

The first attempt replayed saved actions in a new CUDA process and produced a different success set because Newton GPU contact execution was not bitwise deterministic across processes. The final run avoids this by snapshotting GPU body poses during the policy rollout and rendering only after terminal validation.
