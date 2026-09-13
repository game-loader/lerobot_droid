"""Real CPU IMF training verifies schedule propagation and teacher update cadence."""

import pytest
import torch

pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
pytest.importorskip("accelerate", exc_type=ModuleNotFoundError)
pytest.importorskip("diffusers", exc_type=ModuleNotFoundError)

from lerobot.policies.imf_attnres.configuration_imf_attnres import IMFAttnResConfig
from lerobot.policies.imf_attnres.modeling_imf_attnres import IMFAttnResPolicy
from lerobot.scripts import lerobot_train
from tests.training.test_ema import make_dummy_dataset, make_train_config


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_imf_train_schedule_teacher_updates_and_checkpoint(tmp_path, monkeypatch, accumulation_steps):
    root = make_dummy_dataset(tmp_path)
    cfg = make_train_config(root, tmp_path / "output", steps=4, ema_enable=False)
    cfg.policy = IMFAttnResConfig(
        device="cpu",
        push_to_hub=False,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        n_layer=1,
        n_emb=32,
        n_head=4,
        n_kv_head=4,
        spatial_softmax_num_keypoints=4,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        drop_n_last_frames=0,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        enable_semigroup_consistency=True,
        semigroup_start_step=0,
        semigroup_warmup_steps=2,
        semigroup_loss_weight=0.4,
    )
    cfg.optimizer = cfg.policy.get_optimizer_preset()
    cfg.scheduler = cfg.policy.get_scheduler_preset()
    cfg.accelerator.gradient_accumulation.steps = accumulation_steps
    steps, hook_steps, weights, teacher_updates = [], [], [], []
    policies = []
    forward = IMFAttnResPolicy.forward
    update = IMFAttnResPolicy.update
    factory = lerobot_train.make_policy

    def tracked_forward(self, batch, current_step=None):
        steps.append(current_step)
        loss, diagnostics = forward(self, batch, current_step=current_step)
        assert torch.isfinite(loss)
        weights.append((diagnostics or {}).get("imf_diagnostics/semigroup/weight", 0.0))
        return loss, diagnostics

    def tracked_update(self):
        teacher_updates.append(len(steps))
        return update(self)

    def make_policy(*args, **kwargs):
        policy = factory(*args, **kwargs)
        policy.register_forward_pre_hook(
            lambda module, args, kwargs: hook_steps.append(kwargs.get("current_step")), with_kwargs=True
        )
        policies.append(policy)
        return policy

    monkeypatch.setattr(IMFAttnResPolicy, "forward", tracked_forward)
    monkeypatch.setattr(IMFAttnResPolicy, "update", tracked_update)
    monkeypatch.setattr(lerobot_train, "make_policy", make_policy)
    lerobot_train.train(cfg)

    assert steps == [0, 1, 2, 3]
    assert hook_steps == steps
    assert weights == pytest.approx([0.0, 0.2, 0.4, 0.4])
    assert teacher_updates == list(range(accumulation_steps, 5, accumulation_steps))

    checkpoint = cfg.output_dir / "checkpoints" / "000004" / "pretrained_model"
    assert (checkpoint / "policy_preprocessor.json").is_file()
    assert (checkpoint / "policy_postprocessor.json").is_file()
    restored = IMFAttnResPolicy.from_pretrained(checkpoint, local_files_only=True)
    assert restored.config.enable_semigroup_consistency
    torch.testing.assert_close(restored.state_dict(), policies[0].state_dict(), rtol=0, atol=0)
    teacher = restored.model.semigroup_teacher_head
    assert teacher is not None
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
