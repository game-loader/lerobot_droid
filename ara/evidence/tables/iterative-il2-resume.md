# Iterative IL2 Resume Evidence

Date: 2026-08-20

## Recovery

- The previous IL attempt had a complete `050000` checkpoint (`training_step.json` reports `step: 50000`) but its parent orchestrator record was marked failed because serialized unused `count` tensors changed shape.
- Runtime-normalization fingerprinting now hashes only configured statistics (`min/max`, `mean/std`, or configured quantiles), canonicalized on CPU. Source and output fingerprints both resolve to `f8bded67e480dd4013535f7e92495152b946c0d12ff7fe29524c46bb7a81fde1`.
- The resumable orchestrator claimed the verified output without rerunning rollout, merge, or IL and entered offline RL.

## Current Run

Command output at 2026-08-20T11:41:26+08:00:

```json
{"progress/iql_updates":100000.0,"progress/actor_updates":15699.0,"progress/old_policy_syncs":1.0,"progress/phase_id":1.0,"info/actor/approx_kl":0.03420473262667656,"info/actor/ratio_q05":0.5475590229034424,"info/actor/ratio_q50":0.9597189426422119,"info/actor/ratio_q95":1.350358486175537,"info/actor/ratio_max":1.8577847480773926}
```

The offline process is running on `cuda` with AM-Q enabled, `old-policy-sync-target=50`, and 100-episode Newton evaluation every five accepted synchronizations. SwanLab run: https://swanlab.cn/@game-loader/moya-rl100/runs/i8odqnwx

Latest live sample: 17056 actor updates, 1 accepted behavior synchronization, `approx_kl=0.0372886`, and ratio `q05/q50/q95/max=0.6025/0.9724/1.3797/3.2019`; the process remains active.
