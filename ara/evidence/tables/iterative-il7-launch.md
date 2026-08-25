# Iterative IL7 launch

Recorded on 2026-08-24 (Asia/Shanghai).

## Source and policy-selection change

The completed IL6 round selected offline checkpoint `sync_120` at 81% Newton
success. IL7 collection uses that checkpoint, while IL warm-start starts from
IL6's final IL checkpoint `step_050000`.

For this round, periodic IL Newton evaluations remain enabled for monitoring,
but offline RL initialization is explicitly pinned to the final IL checkpoint
(`--offline-use-il-final`). The offline stage target is 200 AM-Q behavior-policy
promotions, evaluated every five promotions with 100 headless CUDA Newton
episodes.

## Completed collection and merge

The source checkpoint completed a 100-episode CUDA rollout with 80 successes
and 20 failures. Canonical v3 merge produced 700 episodes with 527 successes
and 173 failures at 60 Hz.

## Active IL stage

Fixed-normalizer IL is running on CUDA with 50,000 optimizer steps and 5,000
step checkpoints. The downstream offline command will receive
`round_001/il/attempt_001/train/checkpoints/050000/pretrained_model`, regardless
of which periodic IL evaluation has the highest `pc_success`.

The orchestrator manifest records `offline_sync_target: 200` and
`offline_use_il_final: true`.

## Implementation validation

The selector change adds an explicit final-checkpoint mode while preserving
the previous best-evaluation mode as the default. Focused RL100 tests pass:
`29 passed`; Ruff check and format checks also pass.
