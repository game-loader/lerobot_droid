# Offline RL Restart With Compact Metrics

Date: 2026-08-20

Source policy checkpoint:

`outputs/rl100/iterative-moya-il2-20260819-204553/round_001/il/attempt_001/train/checkpoints/050000/pretrained_model`

Dataset:

`outputs/rl100/iterative-moya-il2-20260819-204553/round_001/merge/attempt_001/merged/dataset`

Implementation:

- `OfflineTrainer(debug=False)` is the default.
- Aggregate actor diagnostics remain available (`ratio_q05/q50/q95/max`, aggregate delta log-probability, KL, replay error).
- Per-denoising `info/actor/denoise_*` and `info/actor/post_update/denoise_*` metrics are emitted only with `--debug`.
- Iterative orchestration exposes the same behavior through `--offline-debug`.

Run:

`outputs/rl100/offline-moya-il2-il050000-amq-sync50-20260820-121500`

SwanLab: https://swanlab.cn/@game-loader/moya-rl100/runs/b6nfshio

Live verification at 2026-08-20:

- IQL updates: 100000
- Actor updates: 1357
- Old-policy synchronizations: 0 (AM-Q gate has not accepted a promotion yet)
- Actor approx KL: 0.0426490
- Ratio q05/q50/q95/max: 0.5718 / 0.9749 / 1.3257 / 3.1299
- `denoise_*` keys in the latest metrics row: 0
- CUDA process remains active.
