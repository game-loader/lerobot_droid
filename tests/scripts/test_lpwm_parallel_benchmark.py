import copy
import hashlib

import numpy as np
import pytest

from scripts.lpwm_full.benchmark_parallel import (
    PARITY_KEYS,
    TASK_IDS,
    AuditEnv,
    compare_results,
    observation_digest,
    select_tasks,
)
from scripts.lpwm_full.evaluate import SUITES


def catalog():
    return {
        "tasks": [
            {"suite": s, "task_id": i, "global_task_id": g}
            for g, (s, i) in enumerate((s, i) for s, n in SUITES.items() for i in range(n))
        ]
    }


def test_fixed_panel_covers_all_suites_without_duplicates():
    tasks = select_tasks(catalog(), TASK_IDS)
    assert len(tasks) == 16 and len({t["global_task_id"] for t in tasks}) == 16
    assert {t["suite"] for t in tasks} == set(SUITES)
    assert sum(t["suite"] == "libero_90" for t in tasks) == 8


@pytest.mark.parametrize("ids", [[0, 0], [-1], [130]])
def test_reject_bad_task_selection(ids):
    with pytest.raises(ValueError):
        select_tasks(catalog(), ids)


def result():
    row = dict.fromkeys(PARITY_KEYS, 0)
    row.update(
        seed=4242,
        init_state_index=3,
        episode_index=0,
        success=True,
        control_steps=14,
        terminated=True,
        truncated=False,
        reached_step_limit=False,
        clipped_action_components=0,
        executed_action_components=98,
        global_task_id=120,
        action_sha256="a",
        observation_sha256="b",
    )
    return {"episodes": [row], "end_to_end_seconds": 12.0, "rollout_seconds": 10.0}


def test_matched_results_speedup_and_parity():
    a = result()
    b = copy.deepcopy(a)
    b.update(end_to_end_seconds=6.0, rollout_seconds=2.5)
    comparison = compare_results(a, b)
    assert comparison["bitwise_trajectory_parity"] and comparison["outcome_parity"]
    assert comparison["end_to_end_speedup"] == 2 and comparison["rollout_speedup"] == 4


@pytest.mark.parametrize("field", PARITY_KEYS)
def test_all_seed_outcome_and_length_fields_checked(field):
    a = result()
    b = copy.deepcopy(a)
    b["episodes"][0][field] += 1
    if field == "episode_index":
        with pytest.raises(ValueError, match="coverage"):
            compare_results(a, b)
        return
    comparison = compare_results(a, b)
    assert not comparison["outcome_parity"] and comparison["outcome_mismatches"][0]["fields"] == [field]


@pytest.mark.parametrize("field", ["action_sha256", "observation_sha256"])
def test_trace_mismatch_not_called_bitwise_equivalent(field):
    a = result()
    b = copy.deepcopy(a)
    b["episodes"][0][field] = "changed"
    comparison = compare_results(a, b)
    assert comparison["outcome_parity"] and not comparison["bitwise_trajectory_parity"]


def test_different_coverage_and_duplicate_rows_fail():
    a = result()
    b = copy.deepcopy(a)
    b["episodes"][0]["global_task_id"] = 121
    with pytest.raises(ValueError, match="coverage"):
        compare_results(a, b)
    b = copy.deepcopy(a)
    b["episodes"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        compare_results(a, b)


def test_observation_hash_order_independent_and_pixel_sensitive():
    a = {"pixels": np.zeros((2, 2, 3), dtype=np.uint8), "state": np.zeros(8, np.float32)}
    h1 = hashlib.sha256()
    observation_digest(h1, a)
    h2 = hashlib.sha256()
    observation_digest(h2, dict(reversed(list(a.items()))))
    assert h1.hexdigest() == h2.hexdigest()
    a["pixels"][0, 0, 0] = 1
    h3 = hashlib.sha256()
    observation_digest(h3, a)
    assert h3.hexdigest() != h1.hexdigest()


def test_audit_forwards_init_state_and_does_not_mutate_actions():
    class Dummy:
        init_state_id = 0

        def reset(self, **kwargs):
            return {"pixels": np.zeros((2, 2, 3), np.uint8)}, kwargs

        def step(self, action):
            return {"pixels": np.ones((2, 2, 3), np.uint8)}, 0, False, False, {}

    env = Dummy()
    audit = AuditEnv(env)
    audit.init_state_id = 7
    assert env.init_state_id == audit.init_state_id == 7
    audit.reset(seed=1)
    action = np.zeros(7, np.float32)
    audit.step(action)
    action[0] = 1
    assert audit.actions[0][0] == 0
    audit.reset(seed=1)
    assert audit.actions == []


def test_expanded_panel_fully_supplies_32_workers_and_retains_previous_panel():
    from collections import Counter

    from scripts.lpwm_full.benchmark_parallel import expanded_task_ids, validate_worker_plan

    ids = expanded_task_ids()
    assert len(ids) == len(set(ids)) == 64
    assert set(TASK_IDS) <= set(ids)
    assert ids == expanded_task_ids() and ids != expanded_task_ids(24)
    assert Counter(t["suite"] for t in select_tasks(catalog(), ids)) == {
        "libero_spatial": 8,
        "libero_object": 8,
        "libero_goal": 8,
        "libero_90": 32,
        "libero_10": 8,
    }
    validate_worker_plan(ids, [8, 16, 32, 8])


@pytest.mark.parametrize("order", [[32], [8, 32, 8]])
def test_old_panel_cannot_fake_32_way_concurrency(order):
    from scripts.lpwm_full.benchmark_parallel import validate_worker_plan

    with pytest.raises(ValueError, match="as many distinct task jobs"):
        validate_worker_plan(TASK_IDS, order)


def test_cli_allows_matched_parallel_baseline_with_old_defaults_preserved(tmp_path):
    from scripts.lpwm_full.benchmark_parallel import parse_args

    base = ["--checkpoint", str(tmp_path / "ck"), "--output", str(tmp_path / "out"), "--gpu-uuid", "GPU-test"]
    old = parse_args(base)
    assert old.worker_order == [1, 4, 8, 1] and old.task_ids == list(TASK_IDS)
    new = parse_args(base + ["--expanded-panel", "--worker-order", "8", "16", "32", "8"])
    assert new.worker_order == [8, 16, 32, 8] and len(new.task_ids) == 64
    with pytest.raises(SystemExit):
        parse_args(base + ["--worker-order", "32"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--worker-order", "64"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--expanded-panel", "--task-ids", "0"])
