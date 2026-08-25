# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from diffusers import DDIMScheduler

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from RL.adapters.checkpoint import CheckpointAdapter
from RL.algorithms.ppo import denoising_ppo_loss
from RL.config import TraceConfig
from RL.policy.ddim import stochastic_ddim_step
from RL.policy.diffusion_adapter import DiffusionRLAdapter, _stack_policy_images
from RL.types import ObservationBatch


@pytest.fixture(scope="module")
def tiny_diffusion_adapter(tmp_path_factory: pytest.TempPathFactory) -> DiffusionRLAdapter:
    root = tmp_path_factory.mktemp("tiny_rl_diffusion_checkpoint")
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        device="cpu",
        pretrained_backbone_weights=None,
        down_dims=(8,),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        num_train_timesteps=8,
        num_inference_steps=4,
    )
    policy = DiffusionPolicy(config)
    policy.save_pretrained(Path(root))
    stats = {
        "observation.state": {
            "min": torch.full((3,), -1.0),
            "max": torch.full((3,), 1.0),
        },
        "action": {
            "min": torch.tensor([-2.0, -4.0]),
            "max": torch.tensor([2.0, 4.0]),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(root)
    postprocessor.save_pretrained(root)
    checkpoint = CheckpointAdapter.load(root, device="cpu")
    return DiffusionRLAdapter(
        checkpoint,
        TraceConfig(num_inference_steps=4, eta=1.0, sigma_min=0.01, sigma_max=0.1),
    )


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_stochastic_ddim_replay_matches_sample_and_backpropagates(
    prediction_type: str,
) -> None:
    scheduler = DDIMScheduler(
        num_train_timesteps=8,
        prediction_type=prediction_type,
        clip_sample=True,
        clip_sample_range=0.5,
    )
    scheduler.set_timesteps(4)
    timestep = int(scheduler.timesteps[0])
    previous_timestep = int(scheduler.timesteps[1])
    sample = torch.linspace(-0.5, 0.5, 16).reshape(2, 4, 2)
    model_output = torch.full_like(sample, 0.2)
    sampled = stochastic_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        previous_timestep=previous_timestep,
        sample=sample,
        eta=1.0,
        sigma_min=0.01,
        sigma_max=0.1,
        generator=torch.Generator().manual_seed(7),
    )
    replay_model_output = model_output.clone().requires_grad_()
    replayed = stochastic_ddim_step(
        scheduler=scheduler,
        model_output=replay_model_output,
        timestep=timestep,
        previous_timestep=previous_timestep,
        sample=sample,
        eta=1.0,
        sigma_min=0.01,
        sigma_max=0.1,
        previous_sample=sampled.previous_sample,
    )

    torch.testing.assert_close(replayed.previous_sample, sampled.previous_sample)
    torch.testing.assert_close(replayed.mean, sampled.mean)
    torch.testing.assert_close(replayed.log_prob, sampled.log_prob)
    assert 0.01 - 1e-7 <= replayed.std.item() <= 0.1 + 1e-7
    assert torch.isfinite(replayed.log_prob).all()
    replayed.log_prob.sum().backward()
    assert replay_model_output.grad is not None
    assert torch.isfinite(replay_model_output.grad).all()


def test_stochastic_ddim_uses_configured_clipping_at_final_transition() -> None:
    scheduler = DDIMScheduler(
        num_train_timesteps=8,
        prediction_type="sample",
        clip_sample=True,
        clip_sample_range=0.25,
    )
    scheduler.set_timesteps(4)
    sample = torch.zeros(1, 2, 1)
    output = stochastic_ddim_step(
        scheduler=scheduler,
        model_output=torch.full_like(sample, 10.0),
        timestep=int(scheduler.timesteps[-1]),
        previous_timestep=None,
        sample=sample,
        eta=0.0,
        sigma_min=0.01,
        sigma_max=0.1,
        previous_sample=torch.zeros_like(sample),
    )

    torch.testing.assert_close(output.mean, torch.full_like(sample, 0.25))
    assert output.std.item() == pytest.approx(0.01)


def test_stochastic_ddim_separates_sampling_and_probability_sigma() -> None:
    scheduler = DDIMScheduler(
        num_train_timesteps=8,
        prediction_type="sample",
        clip_sample=False,
    )
    scheduler.set_timesteps(4)
    sample = torch.zeros(1, 2, 1)
    expected_generator = torch.Generator().manual_seed(29)
    expected_noise = torch.randn(sample.shape, generator=expected_generator)

    output = stochastic_ddim_step(
        scheduler=scheduler,
        model_output=torch.zeros_like(sample),
        timestep=int(scheduler.timesteps[-1]),
        previous_timestep=None,
        sample=sample,
        eta=1.0,
        sigma_min=0.0067,
        sigma_max=0.1,
        probability_sigma_min=0.1,
        generator=torch.Generator().manual_seed(29),
    )

    assert output.std.item() == pytest.approx(0.0067)
    assert output.probability_std.item() == pytest.approx(0.1)
    torch.testing.assert_close(
        output.previous_sample,
        output.mean + output.std * expected_noise,
    )
    expected_log_prob = (
        -0.5 * ((output.previous_sample - output.mean) / output.probability_std).square()
        - output.probability_std.log()
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    )
    torch.testing.assert_close(output.log_prob, expected_log_prob)


def test_stochastic_ddim_rejects_invalid_schedule_pair() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=8)

    with pytest.raises(ValueError, match="previous_timestep"):
        stochastic_ddim_step(
            scheduler=scheduler,
            model_output=torch.zeros(1, 2, 1),
            timestep=2,
            previous_timestep=3,
            sample=torch.zeros(1, 2, 1),
            eta=1.0,
            sigma_min=0.01,
            sigma_max=0.1,
        )


def test_stochastic_ddim_rejects_nonfinite_raw_sigma() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=8)
    scheduler.set_timesteps(4)
    with pytest.raises(ValueError, match="raw_std"):
        stochastic_ddim_step(
            scheduler=scheduler,
            model_output=torch.zeros(1, 2, 1),
            timestep=int(scheduler.timesteps[0]),
            previous_timestep=int(scheduler.timesteps[1]),
            sample=torch.zeros(1, 2, 1),
            eta=1e308,
            sigma_min=0.01,
            sigma_max=0.1,
            previous_sample=torch.zeros(1, 2, 1),
        )


def test_trace_replay_has_unit_ratio_and_unet_gradients(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    observation = ObservationBatch(
        {"observation.state": torch.zeros(2, 2, 3)}
    )
    trace = tiny_diffusion_adapter.sample_trace(
        observation, generator=torch.Generator().manual_seed(11)
    )
    replayed = tiny_diffusion_adapter.recompute_log_prob(observation, trace)

    assert trace.latents.shape == (4, 2, 4, 2)
    assert trace.next_latents.shape == trace.latents.shape
    assert trace.old_log_prob.shape == trace.latents.shape
    assert trace.timesteps.shape == (4,)
    assert trace.final_actions.shape == (2, 4, 2)
    assert not trace.latents.requires_grad
    assert not trace.old_log_prob.requires_grad
    torch.testing.assert_close(replayed, trace.old_log_prob, rtol=1e-5, atol=1e-5)

    replay_info = tiny_diffusion_adapter.verify_trace_replay(observation, trace)
    assert replay_info["old_replay_abs_delta_max"] == pytest.approx(0.0, abs=1e-6)
    assert replay_info["old_replay_abs_delta_mean"] == pytest.approx(0.0, abs=1e-7)

    new_executable = tiny_diffusion_adapter.executable_log_prob(replayed)
    old_executable = tiny_diffusion_adapter.executable_log_prob(trace.old_log_prob)
    loss, metrics = denoising_ppo_loss(
        new_executable,
        old_executable,
        torch.ones(2),
        step_mask=torch.ones(2, 2, dtype=torch.bool),
        action_dim_mask=tiny_diffusion_adapter.checkpoint.active_action_mask,
        clip_ratio=0.2,
    )
    tiny_diffusion_adapter.policy.zero_grad(set_to_none=True)
    loss.backward()

    assert metrics["ratio_mean"] == pytest.approx(1.0, rel=1e-5)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tiny_diffusion_adapter.policy.diffusion.unet.parameters()
    )


def test_adapter_reports_effective_and_raw_sigma_for_each_denoising_step(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    diagnostics = tiny_diffusion_adapter.denoising_step_diagnostics()

    assert len(diagnostics) == len(tiny_diffusion_adapter.timesteps)
    assert diagnostics[0]["step"] == 0.0
    assert diagnostics[-1]["previous_timestep"] == -1.0
    assert diagnostics[-1]["sigma_raw"] == pytest.approx(0.0)
    assert diagnostics[-1]["sigma_effective"] == pytest.approx(0.01)
    assert diagnostics[-1]["sigma_inverse_square"] == pytest.approx(10000.0)
    assert diagnostics[-1]["sigma_sample_effective"] == pytest.approx(0.01)
    assert diagnostics[-1]["sigma_probability"] == pytest.approx(0.1)
    assert diagnostics[-1]["sigma_probability_inverse_square"] == pytest.approx(100.0)
    assert diagnostics[-1]["sigma_clamped_to_min"] == 1.0
    assert all(
        0.01 <= diagnostics[index]["sigma_effective"] <= 0.1
        for index in range(len(diagnostics))
    )


def test_probability_sigma_does_not_change_sampled_trace(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    sampling_floor = DiffusionRLAdapter(
        tiny_diffusion_adapter.checkpoint,
        TraceConfig(
            num_inference_steps=4,
            eta=1.0,
            sigma_min=0.01,
            sigma_max=0.1,
            probability_sigma_min=0.01,
        ),
    )
    probability_floor = DiffusionRLAdapter(
        tiny_diffusion_adapter.checkpoint,
        TraceConfig(
            num_inference_steps=4,
            eta=1.0,
            sigma_min=0.01,
            sigma_max=0.1,
            probability_sigma_min=0.1,
        ),
    )
    observation = ObservationBatch({"observation.state": torch.zeros(2, 2, 3)})

    sampled = sampling_floor.sample_trace(
        observation, generator=sampling_floor.make_generator(31)
    )
    widened = probability_floor.sample_trace(
        observation, generator=probability_floor.make_generator(31)
    )

    torch.testing.assert_close(widened.latents, sampled.latents, rtol=0, atol=0)
    torch.testing.assert_close(widened.next_latents, sampled.next_latents, rtol=0, atol=0)
    torch.testing.assert_close(widened.final_actions, sampled.final_actions, rtol=0, atol=0)
    assert not torch.equal(widened.old_log_prob, sampled.old_log_prob)


def test_adapter_rejects_mismatched_transition_contract(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    other = DiffusionRLAdapter(
        tiny_diffusion_adapter.checkpoint,
        TraceConfig(num_inference_steps=4, eta=0.5, sigma_min=0.01, sigma_max=0.1),
    )

    with pytest.raises(ValueError, match="transition contract"):
        tiny_diffusion_adapter.assert_transition_compatible(other)


def test_adapter_rejects_trace_with_inconsistent_final_action(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    observation = ObservationBatch({"observation.state": torch.zeros(1, 2, 3)})
    trace = tiny_diffusion_adapter.sample_trace(
        observation, generator=torch.Generator().manual_seed(23)
    )
    invalid = replace(
        trace,
        final_actions=trace.final_actions.clone().add(1.0),
    )

    with pytest.raises(ValueError, match="final_actions"):
        tiny_diffusion_adapter.recompute_log_prob(observation, invalid)


def test_execution_slice_starts_after_observation_prefix(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    full = torch.arange(4 * 1 * 4 * 2, dtype=torch.float32).reshape(4, 1, 4, 2)
    sliced = tiny_diffusion_adapter.executable_log_prob(full)

    torch.testing.assert_close(sliced, full[:, :, 1:3, :])

    observation = ObservationBatch(
        {"observation.state": torch.zeros(1, 2, 3)}
    )
    trace = tiny_diffusion_adapter.sample_trace(
        observation, generator=torch.Generator().manual_seed(13)
    )
    actions = tiny_diffusion_adapter.executable_actions(trace)
    expected = tiny_diffusion_adapter.checkpoint.unnormalize_action(
        trace.final_actions[:, 1:3, :]
    )
    assert actions.shape == (1, 2, 2)
    torch.testing.assert_close(actions, expected)


def test_stepwise_replay_backpropagates_without_retaining_previous_graphs(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    observation = ObservationBatch(
        {"observation.state": torch.zeros(2, 2, 3)}
    )
    trace = tiny_diffusion_adapter.sample_trace(
        observation, generator=tiny_diffusion_adapter.make_generator(17)
    )
    tiny_diffusion_adapter.policy.zero_grad(set_to_none=True)
    count = 0

    for index, replayed_step in enumerate(
        tiny_diffusion_adapter.iter_recomputed_log_prob(observation, trace)
    ):
        loss, _ = denoising_ppo_loss(
            tiny_diffusion_adapter.executable_log_prob(replayed_step),
            tiny_diffusion_adapter.executable_log_prob(trace.old_log_prob[index : index + 1]),
            torch.ones(2),
            step_mask=torch.ones(2, 2, dtype=torch.bool),
            action_dim_mask=tiny_diffusion_adapter.checkpoint.active_action_mask,
            clip_ratio=0.2,
        )
        loss.backward()
        count += 1

    assert count == 4
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tiny_diffusion_adapter.policy.diffusion.unet.parameters()
    )


def test_single_step_image_stack_matches_lerobot_shape() -> None:
    batch = {
        "observation.image.front": torch.zeros(2, 3, 8, 8),
        "observation.image.wrist": torch.ones(2, 3, 8, 8),
    }

    stacked = _stack_policy_images(
        batch,
        image_keys=("observation.image.front", "observation.image.wrist"),
        n_obs_steps=1,
    )

    assert stacked.shape == (2, 1, 2, 3, 8, 8)
    torch.testing.assert_close(stacked[:, 0, 0], batch["observation.image.front"])
    torch.testing.assert_close(stacked[:, 0, 1], batch["observation.image.wrist"])


def test_adapter_creates_generator_on_policy_device(
    tiny_diffusion_adapter: DiffusionRLAdapter,
) -> None:
    generator = tiny_diffusion_adapter.make_generator(19)

    assert generator.device.type == next(tiny_diffusion_adapter.policy.parameters()).device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stochastic_ddim_rejects_cpu_generator_for_cuda_sample() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=8)
    sample = torch.zeros(1, 2, 1, device="cuda")

    with pytest.raises(ValueError, match="generator device"):
        stochastic_ddim_step(
            scheduler=scheduler,
            model_output=torch.zeros_like(sample),
            timestep=2,
            previous_timestep=1,
            sample=sample,
            eta=1.0,
            sigma_min=0.01,
            sigma_max=0.1,
            generator=torch.Generator().manual_seed(3),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_stochastic_ddim_rejects_generator_on_wrong_cuda_index() -> None:
    scheduler = DDIMScheduler(num_train_timesteps=8)
    sample = torch.zeros(1, 2, 1, device="cuda:0")
    with pytest.raises(ValueError, match="generator device"):
        stochastic_ddim_step(
            scheduler=scheduler,
            model_output=torch.zeros_like(sample),
            timestep=2,
            previous_timestep=1,
            sample=sample,
            eta=1.0,
            sigma_min=0.01,
            sigma_max=0.1,
            generator=torch.Generator(device="cuda:1").manual_seed(3),
        )


def test_rl_scheduler_is_private(tiny_diffusion_adapter: DiffusionRLAdapter) -> None:
    assert tiny_diffusion_adapter.scheduler is not tiny_diffusion_adapter.policy.diffusion.noise_scheduler
    assert isinstance(tiny_diffusion_adapter.scheduler, DDIMScheduler)
    assert tiny_diffusion_adapter.policy.config.noise_scheduler_type == "DDPM"
