"""CPU-only guards for isolated episode scheduling; no robot/env/GPU launch."""

import pytest

from scripts.lpwm_full import benchmark_episodes as bench, benchmark_parallel as base


def test_episode_queue_is_exact_task_major_cartesian_product():
    tasks = [{"global_task_id": task_id} for task_id in base.TASK_IDS]
    jobs = bench.make_jobs(tasks, 10)
    keys = [(job["task"]["global_task_id"], job["episode_index"]) for job in jobs]
    assert len(keys) == len(set(keys)) == 160
    assert keys == [(task_id, episode) for task_id in base.TASK_IDS for episode in range(10)]
    assert jobs[0]["task"] is tasks[0]


@pytest.mark.parametrize("episodes", [-1, 0, 11])
def test_never_resample_or_exceed_original_plan(episodes):
    with pytest.raises(ValueError, match="ten-episode"):
        bench.make_jobs([{"global_task_id": 0}], episodes)


@pytest.mark.parametrize("tasks", [[], [{"global_task_id": 0}, {"global_task_id": 0}]])
def test_duplicate_or_empty_tasks_rejected(tasks):
    with pytest.raises(ValueError, match="unique"):
        bench.make_jobs(tasks, 10)


def test_phase_plan_is_reproducible_and_has_bracketed_controls():
    plan = bench.phase_plan(20260924)
    assert plan == bench.phase_plan(20260924)
    assert any(plan != bench.phase_plan(seed) for seed in range(10))
    assert plan[:2] == [("task", 8), ("episode", 8)]
    assert plan[-2:] == [("episode", 8), ("task", 8)]
    assert set(plan[2:-2]) == {("episode", 10), ("episode", 16), ("episode", 32)}


def test_same_task_overlap_is_measured_not_inferred_from_job_count():
    rows = [
        {"global_task_id": 0, "rollout_start_unix": 0, "rollout_finished_unix": 3},
        {"global_task_id": 0, "rollout_start_unix": 1, "rollout_finished_unix": 2},
        {"global_task_id": 1, "rollout_start_unix": 2, "rollout_finished_unix": 5},
    ]
    assert bench.peak_concurrency(rows) == {
        "peak_active_episodes": 2,
        "peak_same_task_episodes": {"0": 2, "1": 1},
    }
    assert bench.peak_concurrency([]) == {"peak_active_episodes": 0, "peak_same_task_episodes": {}}


def test_single_task_env_cache_closes_old_env(monkeypatch):
    class Native:
        closed = 0

        def close(self):
            self.closed += 1

    class Wrapper:
        env = Native()

    wrapper = Wrapper()
    monkeypatch.setattr(bench, "_ENV", wrapper)
    monkeypatch.setattr(bench, "_ENV_TASK_ID", 1)
    monkeypatch.setattr(bench, "_ENV_META", {"plans": []})
    bench.close_environment()
    assert wrapper.env.closed == 1
    assert bench._ENV is bench._ENV_TASK_ID is bench._ENV_META is None
    bench.close_environment()
    assert wrapper.env.closed == 1


def test_cli_keeps_all_ten_episode_plans():
    args = bench.parse_args(["--checkpoint", "/ckpt", "--output", "/out", "--gpu-uuid", "GPU-test"])
    assert args.episodes_per_task == 10
    assert args.task_ids == list(base.TASK_IDS)


def test_cli_refuses_underfilled_episode_queue():
    with pytest.raises(SystemExit):
        bench.parse_args(
            [
                "--checkpoint",
                "/ckpt",
                "--output",
                "/out",
                "--gpu-uuid",
                "GPU-test",
                "--episodes-per-task",
                "1",
            ]
        )


def test_summary_omits_large_arrays_and_assignments():
    assert bench.summarize(
        {"episodes": [], "worker_initialization": [], "episode_jobs_by_worker": {}, "workers": 32}
    ) == {"workers": 32}


def test_environment_reuse_does_not_change_ten_episode_plan(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.lpwm_ab import evaluate as native

    created, plans = [], []

    class FakeEnv:
        _init_states = list(range(50))
        closed = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def close(self):
            self.closed = True

    suite = SimpleNamespace(get_task=lambda _: SimpleNamespace(name="task", language="language"))
    monkeypatch.setitem(
        sys.modules,
        "lerobot.envs.libero",
        SimpleNamespace(TASK_SUITE_MAX_STEPS={"suite": 400}, LiberoEnv=FakeEnv, _get_suite=lambda _: suite),
    )
    monkeypatch.setattr(
        base, "_BUNDLE", {"mapping": {"agentview_image": "agent", "robot0_eye_in_hand_image": "wrist"}}
    )
    monkeypatch.setattr(native, "episode_plan", lambda *args: plans.append(args) or list(range(10)))
    monkeypatch.setattr(bench, "_ENV", None)
    monkeypatch.setattr(bench, "_ENV_TASK_ID", None)
    monkeypatch.setattr(bench, "_ENV_META", None)
    task = {"global_task_id": 1, "suite": "suite", "task_id": 0, "name": "task", "language": "language"}
    first, metadata, made = bench.environment_for(task)
    assert made and metadata["plans"] == list(range(10))
    assert created[0].kwargs["hard_reset"] is True
    assert created[0].kwargs["n_envs"] == 1
    assert created[0].kwargs["num_steps_wait"] == 10
    second, metadata2, made2 = bench.environment_for(task)
    assert second is first and metadata2 is metadata and not made2
    assert plans == [(50, 10, 42, "validation", 1)]
    third, _, made3 = bench.environment_for(dict(task, global_task_id=2))
    assert third is not first and made3 and created[0].closed
    assert plans[-1] == (50, 10, 42, "validation", 2)
    bench.close_environment()
    assert created[-1].closed


def test_reset_ordinals_replay_skipped_seeds_states_not_actions(monkeypatch):
    from scripts.lpwm_ab import evaluate as native

    seeds, resets = [], []
    monkeypatch.setattr(native, "seed_rollout", seeds.append)
    monkeypatch.setattr(bench, "_ENV_NEXT_EPISODE", 0)
    plans = [{"seed": 10 + i, "init_state_index": 30 + i} for i in range(10)]

    class Env:
        init_state_id = None

        def reset(self, seed):
            resets.append((seed, self.init_state_id))

    env = Env()
    count, seconds = bench.prepare_episode(env, {"plans": plans}, 3)
    assert count == 3 and seconds >= 0
    assert seeds == [10, 11, 12]
    assert resets == [(10, 30), (11, 31), (12, 32)]
    assert bench._ENV_NEXT_EPISODE == 3
    assert bench.prepare_episode(env, {"plans": plans}, 3)[0] == 0
    with pytest.raises(ValueError, match="increase"):
        bench.prepare_episode(env, {"plans": plans}, 2)
