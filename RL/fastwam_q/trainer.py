"""Q-only AdamW/AMP training and self-improvement; FastWAM is never an optimizer parameter."""

from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .data import sample_mixed


class FastWAMQTrainer:
    """Update only Q parameters using AMP, AdamW, chunk TD, and an EMA target."""

    def __init__(self, q_function, normalizer=None, device="cuda"):
        """Construct the component from its configuration and supplied dependencies."""
        self.q = q_function.to(device)
        self.config, self.normalizer = q_function.config, normalizer
        self.device = torch.device(device)
        dino = list(self.q.online.image_encoder.backbone.parameters())
        dino_ids = {id(p) for p in dino}
        head = [p for p in self.q.online.parameters() if p.requires_grad and id(p) not in dino_ids]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": head, "lr": self.config.learning_rate},
                {"params": [p for p in dino if p.requires_grad], "lr": self.config.dino_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
            foreach=False,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda" and self.config.amp_dtype == "float16"
        )
        self.step = 0

    def prepare_batch(self, batch):
        """Move tensors to the Q device and apply the original BC action normalization."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if self.normalizer is not None:
            for key in ("action_chunk", "next_action_chunk"):
                batch[key] = self.normalizer(batch[key])
        return batch

    def update(self, batches):
        """One optimizer step, optionally accumulating a list of microbatches."""
        if isinstance(batches, dict):
            batches = [batches]
        self.q.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for batch in batches:
            batch = self.prepare_batch(batch)
            with torch.autocast(
                self.device.type,
                dtype=getattr(torch, self.config.amp_dtype),
                enabled=self.config.amp_dtype != "float32",
            ):
                loss, values = self.q.td_loss(batch)
            self.scaler.scale(loss / len(batches)).backward()
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0) + float(value) / len(batches)
        self.scaler.unscale_(self.optimizer)
        metrics["grad_norm"] = float(
            torch.nn.utils.clip_grad_norm_(self.q.online.parameters(), self.config.grad_clip_norm)
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= old_scale:
            self.q.polyak_update()
            self.step += 1
        self.optimizer.zero_grad(set_to_none=True)
        metrics["step"] = self.step
        return metrics

    def train_replay(self, demos, online=None, steps=200, generator=None):
        """Run Q-only updates on demonstrations and optional online replay."""
        metrics = []
        for _ in range(steps):
            batches = [
                sample_mixed(demos, online, self.config.batch_size, generator)
                for _ in range(self.config.gradient_accumulation_steps)
            ]
            metrics.append(self.update(batches))
        return metrics

    def self_improve(self, planner, demos, replay, collect, *, iterations=5, updates=200, output=None):
        """collect(planner, iteration) yields episode dictionaries; caller owns resets/reward labels.

        Nothing starts a robot or simulator automatically. The same planner sees updated Q weights.
        """
        planner.q = self.q
        planner.config = self.config
        for iteration in range(iterations):
            self.q.eval()
            for episode in collect(planner, iteration):
                replay.add_episode(episode)
            metrics = self.train_replay(demos, replay, updates)
            if output is not None:
                self.save(Path(output) / f"iteration_{iteration + 1:03d}")
                torch.save(replay.episodes, Path(output) / f"iteration_{iteration + 1:03d}" / "replay.pt")
            yield {"iteration": iteration + 1, "replay_chunks": len(replay), **metrics[-1]}

    def save(self, path):
        """Save Q weights, optimizer, target, tokenizer, and action normalization."""
        save_checkpoint(path, self.q, self.normalizer, trainer=self)

    def restore(self, path):
        """Restore optimizer, AMP scaler, update counter, and PyTorch random states."""
        state = torch.load(Path(path) / "training.pt", map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
