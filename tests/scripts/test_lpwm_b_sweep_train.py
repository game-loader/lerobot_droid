"""CPU contract tests only: no GPU, credentials, telemetry, or real simulator calls."""

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "lpwm_b_sweep_train", Path(__file__).parents[2] / "scripts/lpwm_ab/train_b_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


@pytest.fixture
def cli(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "image_size": 128,
                "language": {"task_ids": list(range(10)), "hidden_dim": 6},
                "tasks": {str(i): f"task {i}" for i in range(10)},
                "cameras": ["observation.images.image", "observation.images.image2"],
            }
        )
    )
    np.save(data / "language_embeddings.npy", np.ones((10, 3, 6), dtype=np.float32))
    np.save(data / "language_masks.npy", np.ones((10, 3), dtype=bool))
    runner = tmp_path / "runner.py"
    runner.write_text("# test-only runner; never invoked by these mocked hook tests\n")
    credential = tmp_path / "credential"
    credential.write_text("test-only-not-a-secret")
    return [
        "--data",
        str(data),
        "--output",
        str(tmp_path / "run"),
        "--eval-runner",
        str(runner),
        "--eval-python",
        sys.executable,
        "--credential-file",
        str(credential),
        "--project",
        "test-project",
        "--run-name",
        "w003",
    ]


def evaluation_result(step=5000, tasks=range(10), episodes=10):
    rows = [
        {
            "task_id": task,
            "num_episodes": episodes,
            "successes": episodes,
            "success_rate": 1.0,
            "pc_success": 100.0,
        }
        for task in tasks
    ]
    count = len(rows) * episodes
    return {
        "status": "complete",
        "checkpoint": {"step": step},
        "per_task": rows,
        "num_episodes": count,
        "successes": count,
        "success_rate": 1.0,
        "pc_success": 100.0,
    }


def write_verified_upload(output, *, preflight=False):
    metadata = {
        "status": "complete",
        "verified_upload": True,
        "formal_result_eligible": not preflight,
        "result_sha256": sweep.file_sha256(output),
    }
    output.with_suffix(".swanlab.json").write_text(json.dumps(metadata))
    return metadata


class ToyPolicy(torch.nn.Module):
    """Explicit test double, never a replacement for the native production policy."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.config = SimpleNamespace(image_features={"observation.images.image": None})

    def save_pretrained(self, destination):
        (destination / "config.json").write_text('{"type": "test-double"}')
        (destination / "model.safetensors").write_bytes(self.weight.detach().numpy().tobytes())


@pytest.fixture
def bundle(cli):
    args = sweep.parse_args(cli)
    policy = ToyPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    policy.weight.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    split = {"sha256": "test-split", "train_episode_ids": [0], "validation_episode_ids": [1]}
    stats = {"mean": [0.0] * 8, "std": [1.0] * 8, "source": "train episodes only"}
    generator = torch.Generator().manual_seed(42)

    def save(step=5000):
        return sweep.save_checkpoint(
            policy, optimizer, step, args.output, args, split, stats, generator, "test-initialization"
        )

    return args, save


def test_cosine_endpoints_and_fixed_30k_horizon():
    assert sweep.cosine_lr(0) == 0
    assert sweep.cosine_lr(1) == pytest.approx(1e-4 / 500)
    assert sweep.cosine_lr(500) == pytest.approx(1e-4)
    assert sweep.cosine_lr(30_000) == pytest.approx(1e-5)
    assert sweep.cosine_lr(15_250) == pytest.approx(5.5e-5)
    assert all(sweep.cosine_lr(i) > sweep.cosine_lr(i + 1) for i in range(500, 30_000))
    for step in (-1, 30_001):
        with pytest.raises(ValueError):
            sweep.cosine_lr(step)
    with pytest.raises(ValueError):
        sweep.cosine_lr(1, warmup_steps=500, total_steps=2)


def test_exact_orchestrator_cli_and_architecture(cli):
    args = sweep.parse_args(
        cli
        + [
            "--reconstruction-weight",
            "0.5",
            "--dynamics-weight",
            "1.5",
            "--prior-weight",
            "0.001",
            "--world-weight",
            "0.03",
            "--world-ramp-steps",
            "1000",
            "--lr",
            "1e-4",
            "--min-lr",
            "1e-5",
            "--warmup-steps",
            "500",
            "--batch-size",
            "8",
            "--grad-accumulation",
            "4",
            "--steps",
            "30000",
            "--seed",
            "42",
            "--workers",
            "0",
            "--gradient-log-every",
            "500",
        ]
    )
    assert (args.variant, args.seed, args.rec_weight, args.dyn_weight) == ("B", 42, 0.5, 1.5)
    assert args.swanlab_mode == "online" and not args.disable_eval
    base = sweep.training_helpers()
    config = sweep.make_config(base, json.loads((args.data / "manifest.json").read_text()), args)
    expected = {
        "variant": "B",
        "hidden_dim": 256,
        "world_hidden_dim": 256,
        "scene_n_layers": 2,
        "expert_n_layers": 4,
        "world_n_layers": 4,
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "num_inference_steps": 10,
        "action_token_repeat": 3,
        "world_ramp_steps": 1000,
        "world_warmup_steps": 0,
        "reconstruction_weight": 0.5,
        "dynamics_weight": 1.5,
        "prior_weight": 0.001,
        "world_weight": 0.03,
        "freeze_encoder": False,
        "dropout": 0.0,
    }
    for key, value in expected.items():
        assert getattr(config, key) == value


@pytest.mark.parametrize(
    "flags",
    [
        ["--steps", "2"],
        ["--steps", "1000"],
        ["--resume"],
        ["--resume", "checkpoint"],
        ["--disable-eval"],
        ["--swanlab-mode", "disabled"],
        ["--world-weight", "nan"],
        ["--world-weight", "-1"],
        ["--lr", "0.1"],
        ["--min-lr", "0.0"],
        ["--seed", "43"],
        ["--batch-size", "4"],
        ["--grad-accumulation", "2"],
        ["--warmup-steps", "2"],
        ["--world-ramp-steps", "10"],
        ["--eval-task-ids", "0"],
        ["--eval-episodes-per-task", "1", "--eval-max-steps", "8"],
        ["--eval-max-steps", "8"],
        ["--preflight", "--eval-task-ids", "0", "0"],
    ],
)
def test_production_gates_and_recipe_are_not_silently_relaxed(cli, flags):
    with pytest.raises(SystemExit):
        sweep.parse_args(cli + flags)


def test_explicit_preflight_can_disable_eval_without_credentials(tmp_path):
    args = sweep.parse_args(
        [
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "run"),
            "--steps",
            "2",
            "--preflight",
            "--disable-eval",
            "--swanlab-mode",
            "disabled",
        ]
    )
    assert args.steps == 2 and args.preflight and args.disable_eval


def test_production_requires_eval_runner_and_credentials(tmp_path):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "run")])


def test_six_permanent_complete_checkpoints_and_alias_only_retention(bundle, monkeypatch):
    args, save = bundle
    original = sweep.atomic_json
    completed = []

    def assert_marker_last(path, value):
        if path.name == "complete.json":
            assert not path.exists()
            assert {p.name for p in path.parent.iterdir()} >= sweep.REQUIRED_ARTIFACTS
            assert set(value["sha256"]) == {p.name for p in path.parent.iterdir()}
            completed.append(value["step"])
        original(path, value)

    monkeypatch.setattr(sweep, "atomic_json", assert_marker_last)
    hashes = {}
    for step in range(5000, 30001, 5000):
        checkpoint = save(step)
        marker = sweep.verify_checkpoint(checkpoint, step)
        assert marker["status"] == "complete"
        hashes[step] = marker["sha256"]
        sweep.checkpoint_alias(checkpoint, "latest")
        if step == 10_000:
            sweep.checkpoint_alias(checkpoint, "best_loss")
    sweep.checkpoint_alias(checkpoint, "final")
    root = args.output / "checkpoints"
    assert completed == list(range(5000, 30001, 5000))
    assert len([p for p in root.iterdir() if not p.is_symlink()]) == 6
    for step, expected in hashes.items():
        assert sweep.verify_checkpoint(root / f"step_{step:06d}", step)["sha256"] == expected
    for name, step in (("latest", 30000), ("best_loss", 10000), ("final", 30000)):
        assert (root / name).is_symlink()
        assert (root / name).resolve() == root / f"step_{step:06d}"
    with pytest.raises(FileExistsError):
        save(5000)
    with pytest.raises(FileExistsError, match="immutable"):
        sweep.checkpoint_alias(root / "step_005000", "final")
    state = torch.load(checkpoint / "optimizer.pt", weights_only=False, map_location="cpu")
    assert state["step"] == 30000 and state["optimizer"]["state"]
    assert {"python_rng", "numpy_rng", "torch_rng", "cuda_rng", "dataloader_rng", "schedule"} <= state.keys()
    assert "credential" not in (checkpoint / "experiment.json").read_text()


def test_partial_export_never_complete_and_never_overwritten(bundle, monkeypatch):
    args, save = bundle

    def partial(self, destination):
        (destination / "model.safetensors").write_bytes(b"partial")
        raise OSError("test interrupted exporter")

    monkeypatch.setattr(ToyPolicy, "save_pretrained", partial)
    with pytest.raises(OSError, match="interrupted"):
        save()
    checkpoint = args.output / "checkpoints/step_005000"
    assert not (checkpoint / "complete.json").exists()
    assert not (checkpoint.parent / "latest").exists()
    with pytest.raises(FileExistsError):
        save()


def test_corruption_and_real_directory_alias_fail_closed(bundle):
    args, save = bundle
    checkpoint = save()
    alias = args.output / "checkpoints/latest"
    alias.mkdir()
    with pytest.raises(FileExistsError):
        sweep.checkpoint_alias(checkpoint, "latest")
    (checkpoint / "optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        sweep.verify_checkpoint(checkpoint, 5000)


def test_output_claim_rejects_competing_or_restarted_trainers(tmp_path):
    path = tmp_path / "run"
    sweep.claim_output(path)
    with pytest.raises(FileExistsError, match="resume"):
        sweep.claim_output(path)


@pytest.mark.parametrize(
    "corruption", ["status", "count", "step", "rate", "tasks", "duplicate", "task_count", "aggregate"]
)
def test_rollout_gate_rejects_invalid_results(corruption):
    result = evaluation_result()
    if corruption == "status":
        result["status"] = "running"
    elif corruption == "count":
        result["num_episodes"] = 99
    elif corruption == "step":
        result["checkpoint"]["step"] = 10000
    elif corruption == "rate":
        result["success_rate"] = float("nan")
    elif corruption == "tasks":
        result["per_task"].pop()
    elif corruption == "duplicate":
        result["per_task"][1]["task_id"] = 0
    elif corruption == "task_count":
        result["per_task"][0]["num_episodes"] = 1
    else:
        result.update(successes=50, success_rate=0.5, pc_success=50.0)
    with pytest.raises(ValueError):
        sweep.validate_eval_result(result, 5000)


def test_formal_and_one_episode_preflight_results_are_separate(cli):
    formal = sweep.validate_eval_result(evaluation_result(), 5000)
    assert formal["rollout/num_episodes"] == 100
    assert formal["rollout/task_09/success_rate"] == 1
    args = sweep.parse_args(
        cli
        + [
            "--preflight",
            "--steps",
            "2",
            "--eval-task-ids",
            "0",
            "--eval-episodes-per-task",
            "1",
            "--eval-max-steps",
            "8",
            "--eval-max-steps",
            "8",
        ]
    )
    result = evaluation_result(2, tasks=[0], episodes=1)
    with pytest.raises(ValueError):
        sweep.validate_eval_result(result, 2)
    with pytest.raises(ValueError, match="preflight"):
        sweep.validate_eval_result(result, 2, task_ids=[0], episodes_per_task=1)
    logged = sweep.validate_eval_result(
        result,
        2,
        task_ids=args.eval_task_ids,
        episodes_per_task=args.eval_episodes_per_task,
        preflight=args.preflight,
    )
    assert logged["preflight_rollout/num_episodes"] == 1
    assert not any(key.startswith("rollout/") for key in logged)


def test_blocking_eval_child_contract_inherits_cuda_and_captures_output(bundle, monkeypatch):
    args, save = bundle
    checkpoint = save()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device-token")
    calls = []

    def run(command, **kwargs):
        import os

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-device-token"
        assert "env" not in kwargs and "shell" not in kwargs
        assert kwargs["stderr"] == sweep.subprocess.STDOUT
        assert kwargs["check"] is False
        assert command[:2] == [str(args.eval_python), str(args.eval_runner)]
        flags = dict(zip(command[2::2], command[3::2], strict=True))
        assert flags == {
            "--checkpoint": str(checkpoint.resolve()),
            "--output": str((args.output / "eval/step_005000.json").resolve()),
            "--credential-file": str(args.credential_file.resolve()),
            "--project": "test-project",
            "--run-name": "w003-eval-step_005000",
            "--episodes-per-task": "10",
            "--seed": "42",
            "--seed-namespace": "validation",
        }
        kwargs["stdout"].write("test child stdout and stderr\n")
        Path(flags["--output"]).write_text(json.dumps(evaluation_result()))
        write_verified_upload(Path(flags["--output"]))
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    logged = sweep.run_checkpoint_eval(args, checkpoint, 5000)
    assert logged["rollout/success_rate"] == 1.0 and len(calls) == 1
    assert "child stdout" in (args.output / "eval/step_005000.log").read_text()
    with pytest.raises(FileExistsError):
        sweep.run_checkpoint_eval(args, checkpoint, 5000)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure", ["returncode", "missing", "invalid_json", "incomplete", "wrong_model", "mutate"]
)
def test_child_eval_failures_propagate(bundle, monkeypatch, failure):
    args, save = bundle
    checkpoint = save()

    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        result = evaluation_result()
        if failure == "returncode":
            return SimpleNamespace(returncode=2)
        if failure == "missing":
            return SimpleNamespace(returncode=0)
        if failure == "invalid_json":
            output.write_text("{")
            return SimpleNamespace(returncode=0)
        if failure == "incomplete":
            result["status"] = "failed"
        if failure == "wrong_model":
            result["checkpoint"]["model_sha256"] = "wrong"
        if failure == "mutate":
            (checkpoint / "model.safetensors").write_bytes(b"mutated")
        output.write_text(json.dumps(result))
        write_verified_upload(output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError)):
        sweep.run_checkpoint_eval(args, checkpoint, 5000)


class DiagnosticWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 1, bias=False)
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("forward_counter", torch.zeros(()))

    def world_loss(self, images, actions, current_step=None):
        assert not actions.requires_grad
        self.forward_counter.add_(1)
        random.random()
        np.random.random()
        torch.rand(1)
        return 3 * self.encoder(images).sum(), {}


class DiagnosticPolicy(torch.nn.Module):
    def __init__(self, weight=0.3):
        super().__init__()
        self.world_model = DiagnosticWorld()
        self.weight = weight

    def flow_matching_loss(self, batch):
        self.world_model.forward_counter.add_(1)
        random.random()
        np.random.random()
        torch.rand(1)
        return self.world_model.encoder(batch["world.images"]).sum(), {}

    def _world_scale(self, step):
        return self.weight * min(1.0, (step + 1) / 1000)


def test_encoder_diagnostics_isolate_rng_buffers_grad_and_use_ramped_world_scale():
    policy = DiagnosticPolicy().train()
    batch = {"world.images": torch.ones(3, 2), "world.actions": torch.ones(3, 7, requires_grad=True)}
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 19.0)
    before = copy.deepcopy(policy.state_dict())
    old_grads = [p.grad.clone() for p in policy.parameters()]
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    expected_draw = (random.random(), np.random.random(), torch.rand(1))
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    result = sweep.encoder_gradient_diagnostics(policy, batch, 499)
    assert result["gradient/world_to_fm_ratio"] == pytest.approx(3 * 0.3 * 0.5)
    assert result["gradient/cosine"] == pytest.approx(1.0)
    assert result["gradient/fm_encoder_norm"] == pytest.approx(18**0.5)
    assert random.random() == expected_draw[0] and np.random.random() == expected_draw[1]
    torch.testing.assert_close(torch.rand(1), expected_draw[2])
    assert policy.training and policy.world_model.training
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for parameter, grad in zip(policy.parameters(), old_grads, strict=True):
        torch.testing.assert_close(parameter.grad, grad)
    assert batch["world.actions"].grad is None


def test_world_zero_diagnostic_has_finite_zero_ratio_and_cosine():
    policy = DiagnosticPolicy(weight=0)
    result = sweep.encoder_gradient_diagnostics(
        policy, {"world.images": torch.ones(3, 2), "world.actions": torch.ones(3, 7)}, 999
    )
    assert result["gradient/world_to_fm_ratio"] == 0
    assert result["gradient/cosine"] == 0
    assert all(parameter.grad is None for parameter in policy.parameters())


@pytest.mark.parametrize("batch,accumulation", [(8, 4), (16, 2), (32, 1)])
@pytest.mark.parametrize("failure", [None, "evaluation", "eval_upload", "online_flush"])
def test_two_update_preflight_exercises_diagnostic_eval_and_final_completion(
    cli, monkeypatch, failure, batch, accumulation
):
    """Real data/loop/export code, explicit tiny policy and child/telemetry doubles."""
    import weakref

    args = sweep.parse_args(
        cli
        + [
            "--preflight",
            "--batch-size",
            str(batch),
            "--grad-accumulation",
            str(accumulation),
            "--steps",
            "2",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--gradient-log-every",
            "500",
            "--validation-batches",
            "1",
            "--eval-task-ids",
            "0",
            "--eval-episodes-per-task",
            "1",
            "--eval-max-steps",
            "8",
            "--eval-max-steps",
            "8",
        ]
    )
    root = args.data
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["episodes"] = [
        {"episode_index": i, "start": i * 32, "end": (i + 1) * 32, "task_index": 0} for i in range(4)
    ]
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name, array in {
        "states": np.ones((128, 8), dtype=np.float32),
        "actions": np.ones((128, 7), dtype=np.float32),
        "images": np.zeros((128, 2, 3, 4, 4), dtype=np.uint8),
        "task_index": np.zeros(128, dtype=np.int64),
    }.items():
        np.save(root / f"{name}.npy", array)
    base = sweep.training_helpers()
    events, losses = [], []

    class TrainingPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.world_model = DiagnosticWorld()

        def forward(self, batch, current_step):
            events.append(("microbatch", current_step))
            assert self.training
            assert batch["action"].shape == (args.batch_size, 16, 7)
            assert batch["world.actions"].shape == (args.batch_size, 2, 7)
            loss = self.world_model.encoder.weight.square().sum()
            losses.append(weakref.ref(loss))
            return loss, {"fm_loss": loss.detach()}

        def get_optim_params(self):
            return self.parameters()

        def save_pretrained(self, destination):
            self.config.save_pretrained(destination)
            (destination / "model.safetensors").write_bytes(b"explicit-test-double")

    monkeypatch.setattr(base, "LPWMFMPolicy", TrainingPolicy)
    monkeypatch.setattr(base, "validate", lambda *a: {"validation/fm_loss": 0.1})

    def diagnostic(policy, batch, step):
        assert policy.training
        events.append(("diagnostic", step))
        return {"gradient/world_to_fm_ratio": 0.01}

    monkeypatch.setattr(sweep, "encoder_gradient_diagnostics", diagnostic)

    def child(command, **kwargs):
        assert all(loss() is None for loss in losses), "Training graphs survived to the eval boundary"
        assert sum(event[0] == "microbatch" for event in events) == 2 * args.grad_accumulation
        assert "--preflight" in command
        assert command[command.index("--max-steps") + 1] == "8"
        assert command[-2:] == ["--task-ids", "0"]
        assert command[command.index("--episodes-per-task") + 1] == "1"
        assert (args.output / "checkpoints/step_000002/complete.json").is_file()
        assert json.loads((args.output / "status.json").read_text())["status"] == "evaluating"
        events.append(("eval", 2))
        if failure == "evaluation":
            return SimpleNamespace(returncode=9)
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(evaluation_result(2, tasks=[0], episodes=1)))
        if failure != "eval_upload":
            write_verified_upload(output, preflight=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", child)

    def login(**kwargs):
        assert kwargs["save"] is False

    def log(values, step):
        if "preflight_rollout/num_episodes" in values:
            assert values["preflight_rollout/num_episodes"] == 1
            events.append(("rollout_logged", step))

    def finish():
        assert json.loads((args.output / "status.json").read_text())["status"] != "completed"
        events.append(("online_flush", 2))
        if failure == "online_flush":
            raise RuntimeError("Test-only telemetry flush failure")

    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(
            login=login,
            init=lambda **kwargs: SimpleNamespace(id="test", url="test-only"),
            log=log,
            finish=finish,
        ),
    )
    monkeypatch.setattr(sweep, "parse_args", lambda argv: args)
    if failure:
        with pytest.raises(RuntimeError):
            sweep.main([])
    else:
        sweep.main([])
    status = json.loads((args.output / "status.json").read_text())
    assert status["step"] == 2
    assert status["status"] == ("failed" if failure else "completed")
    assert ("diagnostic", 1) in events
    assert events.index(("diagnostic", 2)) < events.index(("eval", 2)) < events.index(("online_flush", 2))
    if failure in ("evaluation", "eval_upload"):
        assert not (args.output / "checkpoints/final").exists()
        assert ("rollout_logged", 2) not in events
    else:
        assert (args.output / "checkpoints/final").is_symlink()
        assert (
            events.index(("eval", 2))
            < events.index(("rollout_logged", 2))
            < events.index(("online_flush", 2))
        )
    saved = (args.output / "experiment.json").read_text()
    assert "credential" not in saved and "test-only-not-a-secret" not in saved


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing",
        "invalid_json",
        "running",
        "unverified",
        "integer_true",
        "eligibility",
        "wrong_hash",
        "uploaded_alias_only",
    ],
)
def test_upload_gate_requires_exact_server_verified_result(tmp_path, preflight, failure):
    output = tmp_path / "step_005000.json"
    output.write_text(json.dumps(evaluation_result()))
    if failure == "missing":
        with pytest.raises(RuntimeError, match="sidecar"):
            sweep.verify_eval_upload(output, preflight=preflight)
        return
    metadata = write_verified_upload(output, preflight=preflight)
    sidecar = output.with_suffix(".swanlab.json")
    if failure is None:
        sweep.verify_eval_upload(output, preflight=preflight)
        return
    if failure == "running":
        metadata["status"] = "running"
    elif failure == "unverified":
        metadata["verified_upload"] = False
    elif failure == "integer_true":
        metadata["verified_upload"] = 1
    elif failure == "eligibility":
        metadata["formal_result_eligible"] = preflight
    elif failure == "wrong_hash":
        metadata["result_sha256"] = "different-result"
    elif failure == "uploaded_alias_only":
        metadata.pop("verified_upload")
        metadata["uploaded"] = True
    sidecar.write_text("{" if failure == "invalid_json" else json.dumps(metadata))
    with pytest.raises((RuntimeError, ValueError)):
        sweep.verify_eval_upload(output, preflight=preflight)


@pytest.mark.parametrize("batch,accumulation", [(8, 4), (16, 2), (32, 1)])
def test_batch_tuning_preserves_effective_batch(cli, batch, accumulation):
    args = sweep.parse_args(cli + ["--batch-size", str(batch), "--grad-accumulation", str(accumulation)])
    assert args.batch_size * args.grad_accumulation == 32


@pytest.mark.parametrize("batch,accumulation", [(8, 1), (16, 4), (32, 2)])
def test_batch_tuning_rejects_effective_batch_change(cli, batch, accumulation):
    with pytest.raises(SystemExit):
        sweep.parse_args(cli + ["--batch-size", str(batch), "--grad-accumulation", str(accumulation)])


def test_full_scope_can_explicitly_authorize_80k_without_relaxing_old_sweep(cli):
    with pytest.raises(SystemExit):
        sweep.parse_args(cli + ["--steps", "80000", "--schedule-steps", "80000"])
    args = sweep.parse_args(
        cli + ["--steps", "80000", "--schedule-steps", "80000"], allowed_schedule_steps=(30000, 80000)
    )
    assert args.steps == args.schedule_steps == 80000
    assert sweep.cosine_lr(500, total_steps=80000) == pytest.approx(1e-4)
    assert sweep.cosine_lr(80000, total_steps=80000) == pytest.approx(1e-5)
    assert sweep.cosine_lr(30000, total_steps=80000) > 1e-5
    with pytest.raises(SystemExit):
        sweep.parse_args(cli + ["--steps", "80000"], allowed_schedule_steps=(30000, 80000))


def test_80k_checkpoint_preserves_schedule_endpoint(bundle):
    args, save = bundle
    args.schedule_steps = args.steps = 80000
    path = save(5000)
    state = torch.load(path / "optimizer.pt", weights_only=False)
    assert state["schedule"]["total_steps"] == 80000
