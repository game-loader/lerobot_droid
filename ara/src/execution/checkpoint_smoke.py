import argparse
import json
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig


parser = argparse.ArgumentParser()
parser.add_argument("policy", choices=("dp3", "smolvla"))
parser.add_argument("--result", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(17)
base = Path("/home/droid/project/lerobot_droid/outputs")
if args.policy == "dp3":
    from lerobot.policies.dp3.configuration_dp3 import DP3Config
    from lerobot.policies.dp3.modeling_dp3 import DP3Policy

    path = base / "franka_duo_dp3_action20_pc_only_dit_il_50k/train/checkpoints/050000/pretrained_model"
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    config.device, config.use_amp, config.num_inference_steps = "cpu", False, 1
    policy = DP3Policy.from_pretrained(path, config=config, strict=True, local_files_only=True).eval()
    batch = {
        "observation.state": torch.zeros(1, 34),
        "observation.point_cloud": torch.linspace(-0.1, 0.1, 2048 * 3).reshape(1, 2048, 3),
    }
    noise = torch.randn(1, config.horizon, 20, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        output = policy.select_action(batch, noise=noise)
else:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = base / "train/franka_duo_smolvla_vlm_only_512_h64_a32_50k/checkpoints/050000/pretrained_model"
    config = SmolVLAConfig.from_pretrained(path, local_files_only=True)
    config.device, config.use_amp, config.num_steps = "cpu", False, 1
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True, local_files_only=True).eval()
    batch = {key: torch.zeros(1, *feature.shape) for key, feature in config.input_features.items()}
    batch["observation.language.tokens"] = torch.ones(1, 8, dtype=torch.long)
    batch["observation.language.attention_mask"] = torch.ones(1, 8, dtype=torch.bool)
    with torch.inference_mode():
        output = policy.predict_action_chunk(batch)

assert torch.isfinite(output).all()
torch.save(output.cpu(), args.result)
print(json.dumps({"policy": args.policy, "strict_load": True, "shape": list(output.shape), "finite": True,
                  "parameters": sum(p.numel() for p in policy.parameters()), "result": str(args.result)}))
