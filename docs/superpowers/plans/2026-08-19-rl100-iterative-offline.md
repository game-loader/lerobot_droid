# RL-100 Iterative Offline Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a resumable outer loop that rolls out the best RL checkpoint, appends 100 episodes to a canonical LeRobot v3 dataset, warm-starts IL with the previous IL model and fixed normalizer, then reruns offline RL.

**Architecture:** A pure v3 materializer handles schema reconciliation and atomic dataset publication. A CLI orchestrator coordinates collection, IL, and offline RL as independently restartable subprocess stages with a JSON manifest. Existing collector and offline trainer remain the algorithmic sources of truth.

**Tech Stack:** Python 3.12, PyTorch, LeRobot v3, `uv`, Newton CUDA, SwanLab, pytest.

---

### Task 1: Canonical v3 merge utility

**Files:**
- Create: `RL/datasets/__init__.py`
- Create: `RL/datasets/merge_lerobot_v3.py`
- Test: `tests/rl100/test_lerobot_v3_merge.py`

- [ ] Write tests for base IL episodes without RL fields, rollout episodes with canonical fields, schema mismatch, index rebuilding, and atomic failure.
- [ ] Implement source loading, terminal-label synthesis from collection summaries, strict feature compatibility checks, and canonical writer materialization.
- [ ] Validate reloaded output and generate lineage/episode summary atomically.
- [ ] Run focused merge tests and a real two-source metadata smoke.

### Task 2: Fixed-normalizer IL entry point

**Files:**
- Modify: `src/lerobot/scripts/lerobot_train.py`
- Modify: `src/lerobot/configs/train.py`
- Create: `RL/cli/train_il_warmstart.py`
- Test: `tests/rl100/test_il_warmstart.py`

- [ ] Add an explicit config/CLI switch that prevents dataset stats from overriding pretrained processor stats.
- [ ] Keep pretrained policy weights while creating fresh optimizer, scheduler, step, and RNG state.
- [ ] Add a small wrapper that resolves the final standard LeRobot checkpoint and records the source normalizer fingerprint.
- [ ] Test both fixed-stat and default-stat paths without starting CUDA training.

### Task 3: Resumable outer-loop orchestrator

**Files:**
- Create: `RL/cli/train_iterative_offline.py`
- Modify: `RL/README.md`
- Test: `tests/rl100/test_iterative_offline.py`

- [ ] Define round configuration, manifest schema, artifact completeness checks, and best-checkpoint selection from Newton eval JSON.
- [ ] Implement collection, merge, IL, and offline-RL subprocess stages with deterministic paths and captured logs.
- [ ] Implement restart behavior that skips only validated completed stages and refuses ambiguous partial outputs.
- [ ] Pass the newly trained IL checkpoint, merged dataset root, repo id, and summary to offline RL.
- [ ] Add dry-run and smoke options for CI/local validation.

### Task 4: Integration verification

**Files:**
- Modify: `RL/README.md`
- Test: `tests/rl100/test_iterative_offline.py`

- [ ] Run all `tests/rl100` tests, Ruff, mypy, and `git diff --check`.
- [ ] Run the real-data 200-episode merge smoke and reload it through the RL adapter.
- [ ] Run a two-step IL and one-step offline-RL orchestration smoke with subprocesses mocked or explicitly smoke-bounded.
- [ ] Record manifests, hashes, and validation results for the first real round.
