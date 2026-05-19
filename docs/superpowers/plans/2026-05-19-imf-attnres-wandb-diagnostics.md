# iMF-AttnRes WandB Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log iMF velocity, spike/non-spike, AttnRes depth-attention, and gradient norm diagnostics to WandB during IMF-AttnRes training.

**Architecture:** Add scalar diagnostics where the needed tensors already exist: loss-time metrics in `IMFAttnResModel.compute_loss()`, depth-attention summaries from cached `AttnResOperator` weights, and gradient norms in `lerobot_train.update_policy()` after backward and before optimizer step. Reuse the existing `output_dict` path into WandB. Diagnostics are opt-in with `policy.enable_imf_diagnostics=true`; spike threshold is configurable with `policy.imf_diagnostics_spike_loss_threshold` and defaults to `0.2`. Training uses pseudo-Huber velocity loss by default (`policy.loss_type=pseudo_huber`, `policy.pseudo_huber_delta=1.0`) with `policy.loss_type=mse` retained for ablations.

**Tech Stack:** Python 3.12, PyTorch, pytest, LeRobot policy/training utilities, WandB scalar logger.

---

### Task 1: Policy loss and AttnRes diagnostics

**Files:**
- Modify: `tests/policies/imf_attnres/test_imf_attnres_policy.py`
- Modify: `src/lerobot/policies/imf_attnres/attnres_transformer_components.py`
- Modify: `src/lerobot/policies/imf_attnres/modeling_imf_attnres.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert `policy.forward()` returns an `output_dict` with scalar iMF diagnostics, spike/non-spike bucket counts based on `loss > 0.2`, and AttnRes summaries.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run pytest tests/policies/imf_attnres/test_imf_attnres_policy.py::test_imf_attnres_forward_returns_wandb_diagnostics -q`
Expected: FAIL because diagnostics keys are missing.

- [ ] **Step 3: Implement diagnostics**

Cache detached AttnRes weights on each `AttnResOperator`. Add helper methods on `IMFAttnResModel` to compute norm stats, bucket stats, and AttnRes entropy/max summaries. Make `IMFAttnResPolicy.forward()` return `(loss, diagnostics)`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `uv run pytest tests/policies/imf_attnres/test_imf_attnres_policy.py::test_imf_attnres_forward_returns_wandb_diagnostics -q`
Expected: PASS.

### Task 2: Training gradient diagnostics

**Files:**
- Modify: `tests/policies/imf_attnres/test_imf_attnres_policy.py`
- Modify: `src/lerobot/scripts/lerobot_train.py`

- [ ] **Step 1: Write failing tests**

Add a test that runs `update_policy()` with a tiny accelerator shim and asserts `grad_norm/total`, `grad_norm/attnres`, and `grad_norm/main_dit` appear in `output_dict`.

- [ ] **Step 2: Run test to verify RED**

Run: `uv run pytest tests/policies/imf_attnres/test_imf_attnres_policy.py::test_imf_attnres_update_policy_logs_gradient_norm_buckets -q`
Expected: FAIL because gradient norm keys are missing.

- [ ] **Step 3: Implement gradient norm bucket logging**

Add helper functions in `lerobot_train.py` to compute L2 norms from available gradients and identify AttnRes parameters by name. Add the scalar values to `output_dict` before optimizer step.

- [ ] **Step 4: Run test to verify GREEN**

Run: `uv run pytest tests/policies/imf_attnres/test_imf_attnres_policy.py::test_imf_attnres_update_policy_logs_gradient_norm_buckets -q`
Expected: PASS.

### Task 3: Regression verification

**Files:**
- Test: `tests/policies/imf_attnres/test_imf_attnres_policy.py`

- [ ] **Step 1: Run targeted IMF-AttnRes tests**

Run: `uv run pytest tests/policies/imf_attnres/test_imf_attnres_policy.py -q`
Expected: PASS.

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- src/lerobot/policies/imf_attnres src/lerobot/scripts/lerobot_train.py tests/policies/imf_attnres/test_imf_attnres_policy.py docs/superpowers`
Expected: Diff only contains diagnostics, tests, and docs.
