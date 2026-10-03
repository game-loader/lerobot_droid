"""Self-contained Q checkpoints: network configs, weights, tokenizer, normalization, optimizer."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

from lerobot.utils.import_utils import _transformers_available, require_package

from .config import FastWAMQConfig
from .data import ActionNormalizer
from .model import FastWAMQFunction, T5TextEncoder

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


def save_checkpoint(path, q, normalizer=None, trainer=None):
    """Persist a Q checkpoint with optional training state and BC normalization."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(asdict(q.config), indent=2))
    q.online.image_encoder.backbone.config.save_pretrained(path / "vision_config")
    if q.text_encoder is not None:
        q.text_encoder.model.config.save_pretrained(path / "text_config")
        q.text_encoder.tokenizer.save_pretrained(path / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in q.state_dict().items()},
        path / "model.safetensors",
    )
    if normalizer is not None:
        torch.save(normalizer.state_dict(), path / "normalizer.pt")
    if trainer is not None:
        state = {
            "optimizer": trainer.optimizer.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "step": trainer.step,
            "torch_rng": torch.get_rng_state(),
        }
        if trainer.device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, path / "training.pt")


def load_checkpoint(path, *, device="cpu", with_target=True):
    """Reconstruct a saved Q locally, optionally excluding its training-only target."""
    require_package("transformers", extra="fastwam")
    path = Path(path)
    config = FastWAMQConfig(**json.loads((path / "config.json").read_text()))
    # Rebuild from saved configs, not pretrained downloads; all weights are in model.safetensors.
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(path / "vision_config"))
    text = None
    if config.use_text:
        text = T5TextEncoder(
            config,
            model=T5EncoderModel(AutoConfig.from_pretrained(path / "text_config")),
            tokenizer=AutoTokenizer.from_pretrained(path / "tokenizer"),
        )
    q = FastWAMQFunction(config, dino_backbone=backbone, text_encoder=text, with_target=with_target)
    weights = load_file(path / "model.safetensors")
    if not with_target:
        weights = {k: v for k, v in weights.items() if not k.startswith("target.")}
    q.load_state_dict(weights)
    q.to(device).eval()
    normalizer = None
    if (path / "normalizer.pt").exists():
        normalizer = ActionNormalizer.from_state_dict(torch.load(path / "normalizer.pt", weights_only=True))
    return q, normalizer
