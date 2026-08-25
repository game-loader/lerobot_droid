# RL-100 AM-Q Offline Parity Design

**Status:** approved for implementation in this turn

## Goal

Make the state-based offline diffusion RL path follow RL-100's behavior-policy
iteration semantics: compare old and current likelihoods on the same stored DDIM
transition, keep the behavior snapshot fixed across candidate updates, and use a
learned state dynamics model plus IQL Q values as an AM-Q promotion gate.

## Current defect

The current CLI performs one actor optimizer step per batch and synchronizes the
old policy after every step. Ratio/KL are computed from likelihoods captured
before that optimizer step, so the recorded ratio is structurally one. The
available state dynamics and promotion-gate classes are not wired into the CLI;
the executed path uses only IQL `min(Q1,Q2)-V` advantage.

## Design

### PPO behavior iteration

`OfflineTrainer.train_actor_step` continues to sample one complete stochastic
DDIM trace from the old behavior policy and accumulate gradients across every
denoising step before one optimizer step. After the step it replays the same
trace under the updated current policy under `no_grad` and reports post-update
ratio/KL diagnostics. The trace contract and the independent probability sigma
remain unchanged.

The CLI keeps the old snapshot fixed for a configurable behavior interval. With
AM-Q enabled, promotion is controlled only by the gate; with AM-Q disabled, the
interval provides a deterministic fallback. The current policy is never silently
replaced by the old policy on a rejected gate.

### State AM-Q evaluator

Add a state-only `AMQEvaluator` that:

1. samples candidate and behavior action chunks from the same initial normalized
   state batch with matched seeds;
2. queries IQL `min(Q1,Q2)` on each modeled state/action;
3. advances normalized state history with the state dynamics ensemble mean while
   tracking reward/done probabilities and ensemble disagreement;
4. returns candidate and behavior model-Q means plus validation/disagreement
   diagnostics.

Policy adapters receive denormalized state histories; IQL and dynamics receive
   normalized state histories/actions. Terminal state-delta loss is masked because
   the collection format intentionally repeats the terminal pre-action state.

### Promotion gate

At the configured behavior interval, the evaluator compares candidate and
behavior AM-Q. Promotion requires dynamics validation below its threshold and a
relative candidate improvement of at least 5% of the behavior reference. A
rejected candidate remains current for continued optimization against the old
behavior denominator; a promoted candidate is copied into the old snapshot.
Every gate decision is logged under `info/amq/*`.

### Configuration

Add an `AMQConfig` nested in `RLConfig` and CLI flags for enablement, dynamics
steps, evaluation interval, rollout horizon, ensemble size, relative margin, and
maximum validation loss. Preserve existing `old_policy_sync_interval` as the
non-AM-Q fallback and default AM-Q off for compatibility; formal RL-100 runs
explicitly enable it.

## Testing

- Unit-test post-update ratio/KL using a tiny adapter and assert that a changed
  current policy produces non-unit post-update ratios on the stored trace.
- Unit-test terminal-masked dynamics loss and deterministic AM-Q rollout shapes,
  finite diagnostics, matched candidate/behavior seeds, and promotion/rejection
  boundaries.
- Test CLI propagation of AM-Q flags and rejection of invalid thresholds.
- Run all `tests/rl100`, Ruff, and a CPU AM-Q smoke. Only after these pass run a
  short CUDA offline smoke before resuming a long actor run.
