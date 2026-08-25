# RL-100 AM-Q Offline Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair offline diffusion RL so post-update PPO diagnostics are meaningful and AM-Q gates behavior-policy promotion using state dynamics.

**Architecture:** Keep DDIM traces and dual sigma unchanged. Separate candidate optimizer updates from behavior snapshot promotion, add a state-only AM-Q evaluator over normalized dynamics rollouts, and expose gate metrics through the existing `info/*` namespace. The CLI trains IQL and dynamics before actor updates, then promotes old policy only on a validated AM-Q improvement.

**Tech Stack:** Python 3.12, PyTorch, LeRobot Diffusion Policy adapters, uv, pytest, SwanLab.

---

### Task 1: Add post-update PPO diagnostics

**Files:**
- Modify: `RL/trainers/offline.py:220-326`
- Test: `tests/rl100/test_offline_trainer.py`

- [ ] **Step 1: Write the failing test**

Add a test that constructs identical current/old tiny adapters, runs one actor
step with snapshot synchronization disabled, and asserts the returned
`info/actor/post_update/ratio_q95` is finite and differs from one after the
optimizer changes current parameters, while `info/actor/old_replay_abs_delta_max`
remains below `1e-5`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_offline_trainer.py -k post_update -q`

Expected: FAIL because no post-update diagnostic namespace exists.

- [ ] **Step 3: Implement the smallest update probe**

After `actor_optimizer.step()` and before any snapshot synchronization, replay
the same `trace` under `current_policy` inside `torch.no_grad()`. Reduce the
post-update log probabilities with the existing masks and call
`denoising_ppo_reduced_metrics`. Emit the post-update aggregate and per-denoise
values under `info/actor/post_update/*` and use the post-update aggregate for the
top-level `actor/ratio_mean`, `actor/clip_fraction`, and `actor/approx_kl` fields.
Keep the pre-update loss and per-denoise values under their existing diagnostic
names for compatibility.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_offline_trainer.py -k post_update -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RL/trainers/offline.py tests/rl100/test_offline_trainer.py
git commit -m "fix(rl): report post-update offline PPO ratios"
```

### Task 2: Make dynamics terminal-safe and expose IQL Q evaluation

**Files:**
- Modify: `RL/algorithms/dynamics.py:285-370`
- Modify: `RL/algorithms/iql.py:282-389`
- Test: `tests/rl100/test_dynamics.py`
- Test: `tests/rl100/test_iql.py`

- [ ] **Step 1: Write the failing tests**

Add tests asserting terminal transitions do not contribute to state-delta loss,
while reward and done losses remain finite, and add an IQL helper test asserting
`min_q(observation, action, action_valid)` returns one scalar per batch item and
uses the active action mask.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_dynamics.py tests/rl100/test_iql.py -k 'terminal or min_q' -q`

Expected: FAIL because dynamics loss has no terminal mask and IQL has no public
Q-evaluation helper.

- [ ] **Step 3: Implement terminal masking and `min_q`**

Pass `done` into the dynamics loss calculation, multiply only the state-delta
MSE by `(~done)`, retain reward BCE and done BCE for all rows, and normalize by
the number of active nonterminal rows with an epsilon fallback. Add
`IQL.min_q(observation, action, action_valid)` that uses the online feature
encoder and returns `torch.minimum(q1, q2)` under `no_grad`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_dynamics.py tests/rl100/test_iql.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add RL/algorithms/dynamics.py RL/algorithms/iql.py tests/rl100/test_dynamics.py tests/rl100/test_iql.py
git commit -m "feat(rl): add terminal-safe dynamics and IQL q evaluation"
```

### Task 3: Implement state AM-Q rollout evaluator

**Files:**
- Create: `RL/algorithms/amq.py`
- Modify: `RL/adapters/checkpoint.py:383-410`
- Modify: `RL/policy/diffusion_adapter.py`
- Test: `tests/rl100/test_amq.py`

- [ ] **Step 1: Write failing evaluator tests**

Add a tiny state-only evaluator test that checks candidate/behavior rollout
shapes, matched-seed action generation, finite candidate/behavior AM-Q means,
ensemble disagreement, and rejection of image observations.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_amq.py -q`

Expected: FAIL because the evaluator module does not exist.

- [ ] **Step 3: Implement `AMQEvaluator`**

Implement a state-only evaluator that accepts current/behavior adapters, IQL,
state dynamics, horizon, and seed. For each initial normalized observation:

```python
raw_observation = checkpoint.unnormalize_observation(normalized_observation)
trace = adapter.sample_trace(raw_observation, generator=matched_generator)
action = trace.final_actions[:, execution_slice, :]
q = iql.min_q(normalized_observation, action, action_valid)
prediction = dynamics.predict(normalized_observation, action, action_valid)
state = dynamics.next_state_history(normalized_observation, prediction.mean())
```

Advance the normalized state history with the predicted next state, freeze
terminal rows, accumulate per-step Q and done-masked reward, and report means,
validation loss, disagreement, and rollout horizon. Use identical seeds for
candidate and behavior comparisons. Add public observation denormalization to
`CheckpointAdapter` for this state-only path.

- [ ] **Step 4: Run evaluator tests and verify they pass**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_amq.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add RL/algorithms/amq.py RL/adapters/checkpoint.py RL/policy/diffusion_adapter.py tests/rl100/test_amq.py
git commit -m "feat(rl): add state AM-Q evaluator"
```

### Task 4: Wire behavior promotion and CLI configuration

**Files:**
- Modify: `RL/config.py`
- Modify: `RL/cli/train_offline.py`
- Modify: `RL/trainers/offline.py`
- Modify: `RL/README.md`
- Test: `tests/rl100/test_tracking.py`
- Test: `tests/rl100/test_offline_trainer.py`
- Test: `tests/rl100/test_types_and_config.py`

- [ ] **Step 1: Write failing configuration and gate tests**

Add parser/config tests for AM-Q enablement, dynamics steps, rollout horizon,
evaluation interval, relative margin, maximum validation loss, and ensemble
size. Add a trainer test asserting a rejected gate leaves old policy unchanged
and a promoted gate copies current policy into old policy.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/rl100/test_tracking.py tests/rl100/test_offline_trainer.py tests/rl100/test_types_and_config.py -k amq -q`

Expected: FAIL because CLI and trainer have no AM-Q configuration path.

- [ ] **Step 3: Implement config and orchestration**

Add `AMQConfig` to `RLConfig`. In the offline CLI, build the state dynamics
ensemble from the same state encoder, train it for the configured number of
steps before actor updates, and pass it plus `PolicyPromotionGate` and
`AMQEvaluator` into the trainer. At the configured interval, evaluate candidate
and behavior on the same held-out batch; log `info/amq/*`; promote only when the
gate accepts. Keep `old_policy_sync_interval` as the disabled-AM-Q fallback.
Do not enable an image AM-Q path.

- [ ] **Step 4: Update documentation and tests**

Document the full paper-aligned launch command, explain that `actor/loss` may be
near zero after normalized advantages, and describe post-update ratio/KL and
AM-Q promotion metrics. Run the focused tests.

- [ ] **Step 5: Commit**

```bash
git add RL/config.py RL/cli/train_offline.py RL/trainers/offline.py RL/README.md tests/rl100
git commit -m "feat(rl): wire AM-Q behavior promotion into offline training"
```

### Task 5: Full verification and short CUDA smoke

**Files:**
- Test: `tests/rl100`

- [ ] **Step 1: Run all RL tests**

Run: `UV_CACHE_DIR=.uv-cache HF_HOME=.hf-datasets-cache/hf HF_DATASETS_CACHE=.hf-datasets-cache/datasets uv run pytest tests/rl100 -q`

Expected: all tests pass with only documented CUDA skips.

- [ ] **Step 2: Run static checks**

Run: `UV_CACHE_DIR=.uv-cache uv run ruff check RL tests/rl100`

Expected: no errors.

- [ ] **Step 3: Run a short CUDA AM-Q smoke**

Run the offline CLI from the original IL checkpoint with `--iql-steps 10`,
`--dynamics-steps 10`, `--actor-steps 3`, `--amq-enabled`, and a fresh output
directory. Assert the metrics contain post-update non-unit ratio diagnostics,
finite `info/amq/*`, and at least one gate decision.

- [ ] **Step 4: Inspect the smoke artifact**

Reload the final checkpoint, verify its manifest/config includes AM-Q settings,
and confirm no training process remains before any long run.
