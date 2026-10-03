"""Prepare pretrained encoders, accepting a verified native DINOv3 conversion from ModelScope."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    """Prepare model files or leave an explicit zero-step blocked receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    receipts, errors = {}, {}
    for repo, name in (
        ("google/t5-v1_1-base", "t5-v1_1-base"),
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov3-vitl16-pretrain-lvd1689m"),
    ):
        try:
            model_dir = args.root / "models" / name
            conversion = model_dir / "conversion.json"
            if name == "dinov3-vitl16-pretrain-lvd1689m" and conversion.is_file():
                receipt = json.loads(conversion.read_text())
                if receipt["status"] == "passed" and (model_dir / "model.safetensors").is_file():
                    receipts[repo] = {**receipt, "path": str(model_dir)}
                    continue
            info = api.model_info(repo)
            files = {f.rfilename for f in info.siblings}
            weights = (
                "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "pytorch_model*.bin"
            )
            snapshot_download(
                repo,
                revision=info.sha,
                local_dir=args.root / "models" / name,
                allow_patterns=["*.json", "*.model", weights],
                max_workers=4,
            )
            receipts[repo] = {"revision": info.sha, "path": str(args.root / "models" / name)}
        except Exception as exc:
            errors[repo] = {"type": type(exc).__name__, "message": str(exc)}
    (args.root / "assets/model_sources.json").write_text(json.dumps(receipts, indent=2))
    status = {
        "status": "blocked_pretrained_access" if errors else "models_ready",
        "step": 0,
        "target_steps": 45000,
        "checkpoint_interval": 5000,
        "errors": errors,
        "models": receipts,
        "video_cache_rebuilt": False,
    }
    (args.root / "run/progress.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
