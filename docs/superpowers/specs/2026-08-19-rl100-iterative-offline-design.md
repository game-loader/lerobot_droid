# RL-100 Iterative Offline Learning Design

## Goal

Add a resumable Moya state-only outer loop that expands a LeRobot v3 dataset with
100 Newton rollouts from the highest measured-success RL checkpoint, retrains
Diffusion Policy by imitation learning, and reruns the existing IQL/dynamics/
AM-Q offline stage.

## Semantics

- All new episodes are retained, including successful and failed episodes.
- Rewards are sparse and terminal: successful termination receives `1`; every
  other frame, including failed terminals, receives `0`.
- The checkpoint with the highest real Newton success rate is used only for
  rollout collection.
- IL is warm-started from the previous round's IL checkpoint, never directly
  from an offline-RL actor checkpoint.
- IL starts with a fresh optimizer, scheduler, step counter, and RNG state.
- The previous IL checkpoint's state/action normalizer is reused unchanged for
  the complete IL run. It is not updated online and is not replaced by merged
  dataset statistics.
- Offline RL starts from the newly trained IL checkpoint and retrains IQL and
  dynamics on the complete expanded dataset.

## Architecture

`RL/datasets/merge_lerobot_v3.py` materializes a new canonical dataset through
`LeRobotDataset.create/add_frame/save_episode/finalize`. It inherits the base
state/action schema, adds canonical `next.reward`, `next.done`, and
`next.truncated` features, rebuilds all indices and statistics, and validates
the result before atomic publication.

`RL/cli/train_iterative_offline.py` owns round manifests and stage boundaries:

```text
select best checkpoint -> collect 100 -> merge dataset -> IL -> offline RL
```

Every stage writes to a unique round directory. A completed artifact with a
valid manifest is reused on restart; incomplete staging directories are never
treated as successful stages. The orchestrator invokes existing CLIs through
the current Python environment so each stage retains its established CUDA,
SwanLab, and checkpoint behavior.

## Failure handling

- Source schema, fps, dimensions, task, or terminal labels that disagree are
  rejected before publication.
- A failed merge leaves only a sibling `.incomplete-*` directory and never a
  final dataset path.
- A failed subprocess stops the round and records its command and log path in
  the manifest.
- Existing source datasets and checkpoints are immutable.

## Verification

Unit tests cover legacy IL terminal-label synthesis, rollout-label validation,
index rebuilding, summary lineage, atomic failure, reload through
`LeRobotV3DecisionDataset`, fixed normalizer preservation, and orchestrator
resume/skip behavior. A real-data merge smoke validates the original 100 plus
new 100 episode case before any long training run.
