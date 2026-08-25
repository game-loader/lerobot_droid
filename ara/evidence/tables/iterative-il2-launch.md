# Iterative IL Round 2 Launch

Recorded 2026-08-19 from the live artifacts under
`outputs/rl100/iterative-moya-il2-20260819-204553`.

## Selected Source Checkpoint

```text
label=sync_010 success_rate=65.0 episodes=100
```

Source: `round_001/round_manifest.json`.

## Production Rollout

```text
episodes_saved=100 success_count=59 failure_count=41 complete=True
```

Source: `round_001/rollout/attempt_002/collection/collection_summary.json`.

## Cumulative Dataset

```text
episodes_saved=200 success_count=153 failure_count=47 complete=True
```

```text
dataset.num_frames=172181 (172K)
dataset.num_episodes=200
```

Sources:

- `round_001/merge/attempt_001/merged/collection_summary.json`
- `round_001/il/attempt_001/il_train.log`

## IL Warm Start

```text
'preserve_pretrained_processor_stats': True
cfg.steps=50000 (50K)
```

The run initialized a fresh optimizer/scheduler from the previous IL policy at
`outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model`
and started SwanLab run `rl100-round-001-il`:

```text
https://swanlab.cn/@game-loader/moya-rl100/runs/3yu1jn9q
```

Source: `round_001/il/attempt_001/il_train.log`.

## Snapshot Compatibility Repair

The first production collection attempt failed before environment rollout:

```text
ValueError: processor action statistics disagree for 'count'
```

Inspection showed the selected periodic snapshot stored the preprocessor
`action.count` as a scalar and the postprocessor value as a length-one tensor,
while the MIN_MAX `action.min` and `action.max` tensors remained identical.
After restricting compatibility equality to statistics consumed by the selected
normalization mode, the real `sync_010` checkpoint loaded on CUDA. A two-episode
Newton collector smoke then published a canonical dataset with one success and
one failure.

Sources:

- `round_001/rollout/attempt_001/collect.log`
- `RL/adapters/checkpoint.py`
- `tests/rl100/test_checkpoint_adapter.py`
- `/tmp/rl100-sync010-collector-smoke/collection_summary.json`
