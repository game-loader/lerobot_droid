"""Frozen FastWAM proposals with one video prefill per observation, then Q-weighted selection."""

from collections import deque

import torch

from .checkpoint import load_checkpoint


class FastWAMQPlanner:
    """Input: raw batched LeRobot observations. Output: normalized BC-space action chunks.

    Apply the original FastWAM postprocessor exactly once before sending actions to the environment.
    Neither action normalization nor gripper toggles are applied before Q scoring.
    """

    def __init__(self, fastwam_policy, q_function, preprocessor):
        """Construct the component from its configuration and supplied dependencies."""
        self.fastwam = fastwam_policy.requires_grad_(False).eval()
        self.q = q_function.eval()
        self.config = q_function.config
        self.preprocessor = preprocessor
        self._queue = deque()

    @classmethod
    def from_checkpoints(cls, fastwam_checkpoint, q_checkpoint, device="cuda"):
        """Load frozen BC and inference-only Q, returning the original BC postprocessor."""
        from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

        from .integration import load_processors

        config = FastWAMConfig.from_pretrained(fastwam_checkpoint)
        config.device = device
        fastwam = FastWAMPolicy.from_pretrained(fastwam_checkpoint, config=config, strict=True).to(device)
        pre, post = load_processors(config, fastwam_checkpoint, device)
        q, _ = load_checkpoint(q_checkpoint, device=device, with_target=False)
        return cls(fastwam, q, pre), post

    def reset(self):
        """Clear queued actions at an episode boundary."""
        self._queue.clear()
        self.fastwam.reset()

    @torch.no_grad()
    def sample_candidates(self, processed_batch, generator=None):
        """Draw independent FastWAM chunks while reusing each observation video cache."""
        from lerobot.policies.fastwam.modeling_fastwam import _batch_to_infer_kwargs, _slice_infer_kwargs

        arguments = _batch_to_infer_kwargs(processed_batch, self.fastwam.config)
        batch_size = arguments["input_image"].shape[0]
        return torch.stack(
            [
                self._sample_observation(
                    _slice_infer_kwargs(arguments, index=i, batch_size=batch_size), generator
                )
                for i in range(batch_size)
            ]
        )

    def _sample_observation(self, args, generator):
        # Adapted from this repo's FastWAM.infer_action, reusing its video cache across all N draws.
        model = self.fastwam.model
        image, _, _ = model._normalize_infer_input_image(args["input_image"])
        image = image.to(device=model.device, dtype=model.torch_dtype)
        latents = model._encode_input_image_latents_tensor(input_image=image, tiled=args["tiled"])
        context, context_mask = model._prepare_infer_context(
            None if args["context"] is not None else args["prompt"],
            args["context"],
            args["context_mask"],
            model._normalize_infer_proprio(args["proprio"]),
        )
        video = model.video_expert.pre_dit(
            x=latents,
            timestep=torch.zeros(1, device=model.device, dtype=latents.dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        seq_len = video["tokens"].shape[1]
        mask = model._build_mot_attention_mask(
            video_seq_len=seq_len,
            action_seq_len=self.config.chunk_size,
            video_tokens_per_frame=int(video["meta"]["tokens_per_frame"]),
            device=model.device,
        )
        cache = model.mot.prefill_video_cache(
            video_tokens=video["tokens"],
            video_freqs=video["freqs"],
            video_t_mod=video["t_mod"],
            video_context_payload={"context": video["context"], "mask": video["context_mask"]},
            video_attention_mask=mask[:seq_len, :seq_len],
        )
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.config.fastwam_inference_steps,
            device=model.device,
            dtype=model.torch_dtype,
            shift_override=args["sigma_shift"],
        )
        candidates = []
        # Independent noise on every call; do not reuse FastWAM's fixed inference_seed=42.
        noise_device = model.device if generator is None else generator.device
        noise = torch.randn(
            self.config.num_candidates,
            self.config.chunk_size,
            self.config.action_dim,
            device=noise_device,
            generator=generator,
        ).to(device=model.device, dtype=model.torch_dtype)
        for chunk in noise.split(self.config.candidate_batch_size):
            n = len(chunk)
            expanded_cache = [{k: v.expand(n, *v.shape[1:]) for k, v in layer.items()} for layer in cache]
            for timestep, delta in zip(timesteps, deltas, strict=True):
                prediction = model._predict_action_noise_with_cache(
                    latents_action=chunk,
                    timestep_action=timestep.expand(n),
                    context=context.expand(n, -1, -1),
                    context_mask=context_mask.expand(n, -1),
                    video_kv_cache=expanded_cache,
                    attention_mask=mask,
                    video_seq_len=seq_len,
                )
                chunk = model.infer_action_scheduler.step(prediction, delta, chunk)
            candidates.append(chunk.float())
        return torch.cat(candidates)

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, generator=None):
        """Combine FastWAM proposals using the critic on raw batched observations."""
        self.q.eval()
        candidates = self.sample_candidates(self.preprocessor(dict(batch)), generator)
        device = next(self.q.parameters()).device
        images = torch.stack([batch[k] for k in self.config.camera_keys], dim=1).to(device)
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks] * len(images)
        candidates = candidates.to(device)
        with torch.autocast(
            device.type,
            dtype=getattr(torch, self.config.amp_dtype),
            enabled=self.config.amp_dtype != "float32",
        ):
            values = self.q.score_candidates(images, candidates, tasks)
        return weighted_actions(candidates, values, self.config.q_temperature, self.config.n_elites)

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        """Return the next normalized action, replanning when the execution prefix ends."""
        if not self._queue:
            chunks = self.predict_action_chunk(batch, **kwargs)[:, : self.config.execution_steps]
            self._queue.extend(chunks.transpose(0, 1))
        return self._queue.popleft()


def weighted_actions(candidates, values, temperature=1.0, n_elites=0):
    """Take a softmax-Q weighted mean, optionally restricting to the top candidates."""
    if n_elites:
        values, indices = values.topk(min(n_elites, values.shape[1]), dim=1)
        candidates = candidates.gather(1, indices[:, :, None, None].expand(-1, -1, *candidates.shape[2:]))
    weights = (values.float() / temperature).softmax(1)
    return (weights[:, :, None, None] * candidates).sum(1)
