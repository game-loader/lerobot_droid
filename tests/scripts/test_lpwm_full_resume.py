"""CPU-only continuation/telemetry/task-dispatch contracts; no real services."""

import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.lpwm_full import evaluate, resume
from scripts.lpwm_monitor.resume_train import RestoredBatchSampler


def test_readback_verifies_all_140_metrics_in_bounded_chunks_after_retry(monkeypatch):
    expected = {f"metric/{i}": i / 140 for i in range(140)}
    calls, sleeps = [], []

    class Remote:
        state = "FINISHED"

        def metrics(self, keys, sample):
            calls.append(keys)
            return {"list": [{"key": key, "metrics": [{"index": 1, "data": expected[key]}]} for key in keys]}

    class API:
        attempts = 0

        def run(self, _):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("must not print provider strings")
            return Remote()

    monkeypatch.setattr(evaluate.time, "sleep", sleeps.append)
    result = evaluate.verify_online_resilient(API(), "private/run", expected, attempts=2, delay=1)
    assert result == {"attempts": 2, "metrics_verified": 140}
    assert list(map(len, calls)) == [40, 40, 40, 20]
    assert sleeps == [1]


def test_readback_cannot_accept_missing_metrics_or_expose_provider_text(capsys):
    remote = SimpleNamespace(state="FINISHED", metrics=lambda **_: {"list": []})
    api = SimpleNamespace(run=lambda _: remote)
    with pytest.raises(RuntimeError, match="last error type: ValueError"):
        evaluate.verify_online_resilient(api, "private/run", {"metric": 1}, attempts=1)
    assert "error_type" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["RUNNING", "CRASHED"])
def test_readback_requires_finished_remote_run(state):
    with pytest.raises(RuntimeError, match="not verified"):
        evaluate.verify_online_resilient(
            SimpleNamespace(run=lambda _: SimpleNamespace(state=state)), "r", {"m": 1}, attempts=1
        )


def test_restore_rng_restores_all_cpu_streams(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    generator = torch.Generator().manual_seed(42)
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "dataloader_rng": generator.get_state(),
    }
    reference = random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator)
    resume.restore_rng(state, generator)
    actual = random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator)
    assert reference[:2] == actual[:2]
    assert torch.equal(reference[2], actual[2]) and torch.equal(reference[3], actual[3])


def test_restore_rng_rejects_changed_cuda_device_count(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [torch.get_rng_state()],
        "dataloader_rng": torch.get_rng_state(),
    }
    with pytest.raises(ValueError, match="device count"):
        resume.restore_rng(state, torch.Generator())


@pytest.mark.parametrize("size,batch,start", [(10001, 32, 100), (10001, 32, 400), (10001, 8, 1500)])
def test_reconstructed_sampler_preserves_next_batches_across_original_epochs(size, batch, start):
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(
        torch.arange(size),
        batch_size=batch,
        shuffle=True,
        generator=generator,
        num_workers=1,
        persistent_workers=True,
        drop_last=True,
    )
    iterator = iter(loader)
    try:
        for _ in range(start):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(loader)
                next(iterator)
        sampler = RestoredBatchSampler(size, batch, start, start + 5, generator.get_state())
        assert list(sampler) == [next(iterator).tolist() for _ in range(5)]
    finally:
        del iterator, loader


def test_task_worker_runs_ten_ordered_episodes_in_one_hard_reset_environment(monkeypatch):
    made, observed = [], []

    class Env:
        _init_states = list(range(50))
        closed = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            made.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "lerobot.envs.libero",
        SimpleNamespace(TASK_SUITE_MAX_STEPS={"suite": 520}, LiberoEnv=Env),
    )

    def rollout(env, policy, plan, mapping, stats, language, device, max_steps, image_size):
        observed.append((id(env), plan["episode_index"], max_steps, image_size))
        return {
            **plan,
            "success": False,
            "control_steps": 1,
            "executed_action_components": 7,
            "clipped_action_components": 0,
        }

    monkeypatch.setattr(evaluate.native, "rollout_episode", rollout)
    bundle = {
        "policy": object(),
        "mapping": {"agent": "image"},
        "state_stats": {},
        "language_cache": SimpleNamespace(select=lambda *args: ({}, {})),
    }
    task = SimpleNamespace(name="task", language="instruction")
    item = {"suite": "suite", "task_id": 0, "global_task_id": 30}
    result = evaluate.evaluate_task(
        item,
        bundle,
        {"suite": SimpleNamespace(get_task=lambda _: task)},
        False,
        "validation",
        torch.device("cpu"),
    )
    assert len(made) == 1 and made[0].closed
    assert made[0].kwargs["hard_reset"] and made[0].kwargs["n_envs"] == 1
    assert [row[1] for row in observed] == list(range(10))
    assert len({row[0] for row in observed}) == 1
    assert all(row[2:] == (520, 128) for row in observed)
    assert result["num_episodes"] == 10 and result["successes"] == 0


def test_task_dispatch_rejects_unapproved_global_or_worker_count():
    args = SimpleNamespace(workers=10)
    with pytest.raises(ValueError, match="authorized task8"):
        list(evaluate.task_results(args, [], {}, {}, torch.device("cpu")))


def test_resume_validates_model_and_optimizer_without_resetting_counters(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from scripts.lpwm_ab import train_b_sweep as trainer

    checkpoint = tmp_path / "old/checkpoints/step_005000"
    checkpoint.mkdir(parents=True)
    old = checkpoint.parent.parent
    output = tmp_path / "continued"
    output.mkdir()
    args = SimpleNamespace(
        schedule_steps=80000,
        steps=5002,
        workers=1,
        batch_size=32,
        grad_accumulation=1,
        output=output,
        resume_eval_result=tmp_path / "evaluation.json",
        eval_workers=8,
    )
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=trainer.cosine_lr(5000, total_steps=80000))
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_weights = {k: value.detach().clone() for k, value in policy.state_dict().items()}
    for value in optimizer.state.values():
        value["step"].fill_(5000)
    generator = torch.Generator().manual_seed(42)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    torch.randperm(200000, generator=generator)
    state = {
        "step": 5000,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "dataloader_rng": generator.get_state(),
        "schedule": {"warmup_steps": 500, "total_steps": 80000, "lr": 1e-4, "min_lr": 1e-5},
    }
    torch.save(state, checkpoint / "optimizer.pt")
    save_file(expected_weights, str(checkpoint / "model.safetensors"))
    (old / "metrics.jsonl").write_text(json.dumps({"step": 5000, "validation/fm_loss": 0.25}) + "\n")
    (old / "swanlab_run.json").write_text('{"id":"parent"}')
    completion = {"step": 5000, "sha256": {}}
    (checkpoint / "complete.json").write_text(json.dumps(completion))
    monkeypatch.setattr(
        resume,
        "verify_resume_inputs",
        lambda *a: (
            checkpoint,
            completion,
            {"initial_weights_sha256": "initial"},
            {"eval/num_episodes": 1300},
        ),
    )
    monkeypatch.setattr(trainer, "verify_checkpoint", lambda *a: completion)
    fresh = torch.nn.Linear(2, 1)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
    continued = resume.restore_training(
        args, fresh, fresh_optimizer, torch.Generator(), {}, {}, "initial", 200000
    )
    assert continued["step"] == 5000 and continued["best_saved_loss"] == 0.25
    assert all(torch.equal(value, fresh.state_dict()[key]) for key, value in expected_weights.items())
    assert all(int(value["step"]) == 5000 for value in fresh_optimizer.state.values())
    assert continued["receipt"]["bitwise_data_order_resume"] is True
    assert (output / "checkpoints/step_005000").is_symlink()
    assert continued["receipt"]["next_lr"] == trainer.cosine_lr(5001, total_steps=80000)


def test_native_shared_dlp_aliases_use_strict_model_aware_loading(tmp_path):
    from safetensors.torch import save_model

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.visual_alias = self.encoder

    source, target = Shared(), Shared()
    path = tmp_path / "shared.safetensors"
    save_model(source, str(path))
    resume.load_model(target, str(path), strict=True, device="cpu")
    assert target.encoder is target.visual_alias
    assert torch.equal(target.encoder.weight, source.encoder.weight)
    bad = torch.nn.Linear(2, 3)
    with pytest.raises(RuntimeError):
        resume.load_model(bad, str(path), strict=True, device="cpu")
