"""Atomic checkpoints: accepted policy for deployment, candidate/optimizers for resume."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import torch

CHECKPOINT_KIND = "smolvla_offline_rl_v2_tokens"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, trainer, pre, post, metadata, counters):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".incomplete-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        deployment = staging / "pretrained_model"
        trainer.behavior.policy.save_pretrained(deployment)
        pre.save_pretrained(deployment)
        post.save_pretrained(deployment)
        # Only candidate actor tensors differ from accepted weights. Frozen SmolVLM is stored once.
        candidate = {
            name: p.detach().cpu() for name, p in trainer.current.policy.named_parameters() if p.requires_grad
        }
        torch.save(
            {
                "candidate": candidate,
                "actor_optimizer": trainer.actor_optimizer.state_dict(),
                "iql": trainer.iql.state_dict(),
                "q_optimizer": trainer.iql.q_optimizer.state_dict(),
                "v_optimizer": trainer.iql.v_optimizer.state_dict(),
                "dynamics": trainer.dynamics.state_dict(),
                "dynamics_optimizer": trainer.dynamics.optimizer.state_dict(),
                "actor_updates": trainer.actor_updates,
                "promotions": trainer.promotions,
                "counters": counters,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            staging / "training_state.pt",
        )
        (staging / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        hashes = {str(p.relative_to(staging)): file_hash(p) for p in staging.rglob("*") if p.is_file()}
        (staging / "manifest.json").write_text(
            json.dumps({"kind": CHECKPOINT_KIND, "sha256": hashes}, indent=2) + "\n"
        )
        staging.rename(path)
    except BaseException:
        shutil.rmtree(staging)
        raise


def validate_checkpoint(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") == "smolvla_offline_rl_v1":
        raise ValueError("Old v1 MLP/KV dynamics RL states are incompatible; restart token heads from IL")
    if manifest.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Not a SmolVLA offline RL checkpoint")
    required = {
        "run.json",
        "training_state.pt",
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
    }
    if not required.issubset(manifest.get("sha256", {})):
        raise ValueError("Checkpoint manifest is missing mandatory files")
    for name, expected in manifest["sha256"].items():
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file() or file_hash(file) != expected:
            raise ValueError(f"Checkpoint integrity check failed: {name}")
    return json.loads((path / "run.json").read_text())


def restore_checkpoint(path, trainer):
    validate_checkpoint(path)
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    current = {name: p for name, p in trainer.current.policy.named_parameters() if p.requires_grad}
    if current.keys() != state["candidate"].keys():
        raise ValueError("Candidate parameter contract changed")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state["candidate"][name])
    trainer.actor_optimizer.load_state_dict(state["actor_optimizer"])
    trainer.iql.load_state_dict(state["iql"], strict=True)
    trainer.iql.q_optimizer.load_state_dict(state["q_optimizer"])
    trainer.iql.v_optimizer.load_state_dict(state["v_optimizer"])
    trainer.dynamics.load_state_dict(state["dynamics"], strict=True)
    trainer.dynamics.optimizer.load_state_dict(state["dynamics_optimizer"])
    trainer.actor_updates, trainer.promotions = state["actor_updates"], state["promotions"]
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["counters"]
