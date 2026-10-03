"""Small, deterministic controls and pre-update guards for critic stability experiments."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StabilityCase:
    """A single treatment; unlisted settings stay at the source checkpoint's values."""

    name: str
    head_lr: float = 3e-4
    dino_lr: float = 9e-5
    precision: str = "bfloat16"
    warmup_steps: int = 0
    update_dino: bool = True


CASES = {
    case.name: case
    for case in (
        StabilityCase("baseline"),
        StabilityCase("dino_lr_div10", dino_lr=9e-6),
        StabilityCase("head_lr_div10", head_lr=3e-5),
        StabilityCase("both_lr_div10", head_lr=3e-5, dino_lr=9e-6),
        StabilityCase("both_lr_div10_warmup", head_lr=3e-5, dino_lr=9e-6, warmup_steps=300),
        StabilityCase("critic_fp32", precision="float32"),
        StabilityCase("no_dino_updates", update_dino=False),
    )
}


def learning_rates(case, update):
    """Use a linear restart warmup, then constant rates; do not conflate it with decay."""
    scale = min(1.0, update / case.warmup_steps) if case.warmup_steps else 1.0
    return case.head_lr * scale, case.dino_lr * scale


@dataclass
class GradientGuard:
    """Stop BEFORE Adam/EMA on a nonfinite, catastrophic, or persistently extreme gradient."""

    hard_limit: float = 1e6
    sustained_limit: float = 1e3
    patience: int = 20
    consecutive: int = 0

    def check(self, norm, loss):
        """Return a stop reason, or None to permit the optimizer update."""
        if not math.isfinite(norm) or not math.isfinite(loss):
            return "nonfinite_loss_or_gradient"
        if norm >= self.hard_limit:
            return "gradient_at_or_above_1e6"
        self.consecutive = self.consecutive + 1 if norm > self.sustained_limit else 0
        if self.consecutive >= self.patience:
            return "gradient_above_1e3_for_20_consecutive_updates"
        return None


def gradient_norm(parameters):
    """Return the global L2 norm for a parameter group without synchronizing each tensor."""
    norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    return torch.stack(norms).norm() if norms else torch.tensor(0.0)


def monitored_update(trainer, batch, case, update, guard, seed):
    """Same BF16/FP32 AdamW and EMA update as the original trainer, with extra measurements."""
    torch.manual_seed(seed + update)
    torch.cuda.manual_seed_all(seed + update)
    trainer.q.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    for group, lr in zip(trainer.optimizer.param_groups, learning_rates(case, update), strict=True):
        group["lr"] = lr
    batch = trainer.prepare_batch(batch)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=case.precision == "bfloat16"):
        loss, values = trainer.q.td_loss(batch)
    loss.backward()
    head, dino = trainer.optimizer.param_groups
    head_norm = float(gradient_norm(head["params"]))
    dino_norm = float(gradient_norm(dino["params"]))
    norm = math.hypot(head_norm, dino_norm)
    metrics = {k: float(v) for k, v in values.items()}
    metrics.update(
        update=update,
        attempted_step=trainer.step + 1,
        grad_norm=norm,
        grad_norm_head=head_norm,
        grad_norm_dino=dino_norm,
        dino_gradient_energy_fraction=dino_norm**2 / max(norm**2, 1e-30),
        clipped=float(norm > trainer.config.grad_clip_norm),
        head_lr=head["lr"],
        dino_lr=dino["lr"],
        action_clipped_fraction=float((batch["action_chunk"].abs() >= 5).float().mean()),
    )
    reason = guard.check(norm, metrics["loss"])
    metrics["guard_consecutive"] = guard.consecutive
    if reason:
        metrics.update(step=trainer.step, optimizer_applied=False)
        return metrics, reason
    torch.nn.utils.clip_grad_norm_(trainer.q.online.parameters(), trainer.config.grad_clip_norm)
    metrics["grad_norm_after_clip"] = float(gradient_norm(trainer.q.online.parameters()))
    metrics["clip_coefficient"] = min(1.0, trainer.config.grad_clip_norm / (norm + 1e-6))
    trainer.optimizer.step()
    trainer.q.polyak_update()
    trainer.step += 1
    trainer.optimizer.zero_grad(set_to_none=True)
    metrics.update(step=trainer.step, optimizer_applied=True)
    return metrics, None
