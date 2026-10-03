"""Standalone *simulator* evaluation of frozen LPWM-FM checkpoints on LIBERO Spatial.

No dataset actions/targets, offline loss, or task-ID embeddings are used. Install
and configure LIBERO/EGL before invoking this script; imports that allocate a
simulator are delayed until main(). Helper tests can run without native SDKs.

Example:
    python scripts/lpwm_ab/evaluate.py --checkpoint frozen/step_2000 \
        --output results/step_2000.json --device cuda --episodes-per-task 10

Use identical --seed and --seed-namespace for paired A/B validation. Final testing
uses --seed-namespace final (disjoint init-state pool AND independent random seeds).
"""

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SUITE = "libero_spatial"
LANGUAGE = "observation.language.embedding"
LANGUAGE_MASK = "observation.language.attention_mask"
STATE = "observation.state"
REQUIRED_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "experiment.json",
    "state_normalization.json",
    "language_metadata.json",
    "language_embeddings.npy",
    "language_masks.npy",
)
# Explicit semantic aliases only: never assume the first configured view is agentview.
CAMERA_ALIASES = {
    "image": "agentview_image",
    "agent": "agentview_image",
    "agentview": "agentview_image",
    "agentview_image": "agentview_image",
    "image2": "robot0_eye_in_hand_image",
    "wrist": "robot0_eye_in_hand_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand_image",
}


def normalize_task_description(text: str) -> str:
    """Exact natural-language matching after Unicode/case/whitespace/punctuation normalization."""
    if not isinstance(text, str):
        raise TypeError("Task description must be a natural-language string.")
    result = re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    if not result:
        raise ValueError("Task description is empty.")
    return result


class LanguageCache:
    """Resolve text to frozen cached rows; numeric suite task IDs are never a fallback."""

    def __init__(self, metadata: dict, embeddings: np.ndarray, masks: np.ndarray, language_dim: int):
        task_ids = metadata["language"]["task_ids"]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Duplicate task IDs in language cache metadata.")
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.masks = np.asarray(masks)
        if self.embeddings.ndim != 3 or self.embeddings.shape[0] != len(task_ids):
            raise ValueError("Language embeddings must be [num_cached_tasks,L,D].")
        if self.embeddings.shape[2] != language_dim or self.masks.shape != self.embeddings.shape[:2]:
            raise ValueError("Language width/mask shape does not match checkpoint configuration.")
        if not np.isfinite(self.embeddings).all() or not np.isin(self.masks, [0, 1]).all():
            raise ValueError("Language embeddings/masks must be finite with binary masks.")
        self.masks = self.masks.astype(bool)
        if not self.masks.any(axis=1).all():
            raise ValueError("Each cached task requires at least one unmasked language token.")
        self.lookup: dict[str, tuple[int, int, str]] = {}
        for row, task_id in enumerate(task_ids):
            description = metadata["tasks"][str(task_id)]
            normalized = normalize_task_description(description)
            if normalized in self.lookup:
                raise ValueError(f"Ambiguous duplicate normalized task description: {description!r}")
            self.lookup[normalized] = (row, int(task_id), description)

    def select(self, task_description: str, device: torch.device | str) -> tuple[dict, dict]:
        normalized = normalize_task_description(task_description)
        if normalized not in self.lookup:
            raise ValueError(f"No cached language embedding for task description {task_description!r}.")
        row, task_id, description = self.lookup[normalized]
        tensors = {
            LANGUAGE: torch.from_numpy(self.embeddings[row].copy()).unsqueeze(0).to(device),
            LANGUAGE_MASK: torch.from_numpy(self.masks[row].copy()).unsqueeze(0).to(device),
        }
        return tensors, {"cache_row": row, "cache_task_id": task_id, "cache_description": description}


def camera_mapping(camera_keys: list[str]) -> dict[str, str]:
    """Return raw simulator camera -> full checkpoint feature, preserving checkpoint view order."""
    mapping = {}
    for key in camera_keys:
        if not key.startswith("observation.images."):
            raise ValueError(f"Not an image feature: {key}")
        alias = key.removeprefix("observation.images.")
        if alias not in CAMERA_ALIASES:
            raise ValueError(f"Unknown camera semantics for {key}; refusing an order-based guess.")
        raw = CAMERA_ALIASES[alias]
        if raw in mapping:
            raise ValueError(f"Multiple checkpoint image keys map to the same camera {raw}.")
        mapping[raw] = key
    if set(mapping) != {"agentview_image", "robot0_eye_in_hand_image"}:
        raise ValueError("These A/B checkpoints require exactly agent and wrist cameras.")
    return mapping


def rotate_resize_rgb(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """One 180-degree raw-camera rotation, then training-cache PIL RGB bilinear resize."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Simulator image must be HWC RGB uint8.")
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    resized = (
        Image.fromarray(rotated).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    )
    return np.asarray(resized, dtype=np.uint8).copy()


def canonical_state(robot_state: dict) -> np.ndarray:
    """Exact canonical8 order and xyzw quaternion conversion from LiberoProcessorStep."""
    pos = torch.as_tensor(np.asarray(robot_state["eef"]["pos"], dtype=np.float32))
    quat = torch.as_tensor(np.asarray(robot_state["eef"]["quat"], dtype=np.float32))
    gripper = torch.as_tensor(np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32))
    if pos.shape != (3,) or quat.shape != (4,) or gripper.shape != (2,):
        raise ValueError("Expected EEF position3, quaternion4 (xyzw), gripper qpos2.")
    if not all(torch.isfinite(value).all() for value in (pos, quat, gripper)):
        raise ValueError("Simulator robot state contains nonfinite values.")
    w = quat[3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    axis_angle = torch.zeros(3, dtype=torch.float32)
    if denominator > 1e-10:
        axis_angle = quat[:3] / denominator * (2.0 * torch.acos(w))
    return torch.cat((pos, axis_angle, gripper)).numpy()


def state_statistics(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    mean, std = (np.asarray(stats[key], dtype=np.float32) for key in ("mean", "std"))
    if mean.shape != (8,) or std.shape != (8,):
        raise ValueError("Saved state normalization must contain eight means/stds.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Saved state means/stds must be finite and stds positive.")
    # Do NOT recompute stats, clamp them again, or add a new normalization epsilon.
    return mean, std


def observation_batch(
    observation: dict,
    mapping: dict[str, str],
    stats: tuple[np.ndarray, np.ndarray],
    language: dict[str, torch.Tensor],
    device: torch.device | str,
    image_size: int = 128,
) -> dict[str, torch.Tensor]:
    mean, std = stats
    state = (canonical_state(observation["robot_state"]) - mean) / std
    batch = {STATE: torch.from_numpy(state.copy()).unsqueeze(0).to(device), **language}
    for raw_camera, feature in mapping.items():
        # Our LiberoEnv camera_name_mapping yields full checkpoint feature keys.
        # Raw camera names are accepted for helper callers, never inferred by dictionary order.
        pixels = observation["pixels"]
        key = feature if feature in pixels else raw_camera
        rgb = rotate_resize_rgb(pixels[key], image_size)
        batch[feature] = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
        )
    return batch


def clip_native_action(action: torch.Tensor | np.ndarray) -> tuple[np.ndarray, int]:
    if isinstance(action, torch.Tensor):
        action = action.detach().float().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.shape == (1, 7):
        action = action[0]
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Policy must return one finite native LIBERO action with shape [1,7] or [7].")
    count = int(np.count_nonzero((action < -1.0) | (action > 1.0)))
    return np.clip(action, -1.0, 1.0), count  # no gripper inversion/scaling


def protocol_seed(seed: int, namespace: str, task_id: int, episode_index: int) -> int:
    if namespace not in {"validation", "final"}:
        raise ValueError("Seed namespace must be validation or final.")
    text = f"lpwm-libero-rollout-v1:{namespace}:{seed}:{task_id}:{episode_index}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def episode_plan(
    total_states: int,
    episodes: int,
    seed: int,
    namespace: str,
    task_id: int,
    init_state_offset: int = 0,
) -> list[dict[str, int]]:
    """Deterministic no-replacement selection from disjoint validation/final state pools."""
    if init_state_offset < 0:
        raise ValueError("init-state-offset must be nonnegative.")
    if total_states < 2 or episodes < 1:
        raise ValueError("Need at least two init states and one episode.")
    midpoint = total_states // 2
    pool = np.arange(0, midpoint) if namespace == "validation" else np.arange(midpoint, total_states)
    rng = np.random.default_rng(protocol_seed(seed, namespace, task_id, -1))
    if init_state_offset + episodes > len(pool):
        raise ValueError(
            f"Requested {episodes} episodes with offset {init_state_offset}, "
            f"but {namespace} init-state pool has only {len(pool)}."
        )
    return [
        {
            "episode_index": i,
            "init_state_index": int(state),
            "seed": protocol_seed(seed, namespace, task_id, init_state_offset + i),
        }
        for i, state in enumerate(rng.permutation(pool)[init_state_offset : init_state_offset + episodes])
    ]


def seed_rollout(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def rollout_episode(
    env,
    policy,
    plan: dict[str, int],
    mapping: dict[str, str],
    stats: tuple[np.ndarray, np.ndarray],
    language: dict[str, torch.Tensor],
    device: torch.device | str,
    max_steps: int,
    image_size: int = 128,
    video_writer=None,
) -> dict:
    """Run actual control steps; only the simulator's is_success determines success."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    seed_rollout(plan["seed"])
    policy.reset()  # drop history/action chunks from the preceding episode
    env.init_state_id = plan["init_state_index"]  # wrapper increments after reset; we set it explicitly
    clipped, executed, success = 0, 0, False
    terminated = truncated = False
    start = time.monotonic()
    try:
        observation, _ = env.reset(seed=plan["seed"])
        for _ in range(max_steps):
            batch = observation_batch(observation, mapping, stats, language, device, image_size)
            if video_writer is not None:
                agent_key = mapping["agentview_image"]
                rgb = (batch[agent_key][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                video_writer.append_data(rgb)
            # This is select_action, not predict_action_chunk each step: policy owns its execution queue.
            action, count = clip_native_action(policy.select_action(batch))
            observation, _, terminated, truncated, info = env.step(action)
            executed += 1
            clipped += count
            if "is_success" not in info:
                raise KeyError("LIBERO step did not provide authoritative is_success.")
            success = bool(info["is_success"])
            if success or terminated or truncated:
                break
        if video_writer is not None:
            key = mapping["agentview_image"]
            pixels = observation["pixels"]
            video_writer.append_data(
                rotate_resize_rgb(pixels.get(key, pixels.get("agentview_image")), image_size)
            )
        return {
            **plan,
            "success": success,
            "control_steps": executed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reached_step_limit": executed >= max_steps and not (success or terminated or truncated),
            "clipped_action_components": clipped,
            "executed_action_components": executed * 7,
            "action_clipping_fraction": clipped / (executed * 7) if executed else 0.0,
            "duration_seconds": time.monotonic() - start,
        }
    finally:
        policy.reset()  # also clear queues if invalid actions or simulator exceptions abort a rollout


def summarize_episodes(episodes: list[dict]) -> dict:
    """Every episode contributes exactly one trial, regardless of its chunk/control-step count."""
    count = len(episodes)
    if not count:
        raise ValueError("Cannot report simulator success with an empty episode denominator.")
    successes = sum(int(row["success"]) for row in episodes)
    components = sum(row["executed_action_components"] for row in episodes)
    clipped = sum(row["clipped_action_components"] for row in episodes)
    return {
        "successes": successes,
        "num_episodes": count,
        "success_rate": successes / count,
        "pc_success": 100.0 * successes / count,
        "control_steps": sum(row["control_steps"] for row in episodes),
        "clipped_action_components": clipped,
        "executed_action_components": components,
        "action_clipping_fraction": clipped / components if components else 0.0,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Replace the JSON only after the entire new result is written and flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(checkpoint: Path, device: torch.device) -> dict[str, Any]:
    """Strictly load local exported weights; no resume optimizer or network downloads."""
    from lerobot.configs import NormalizationMode
    from lerobot.policies.lpwm_fm.configuration_lpwm_fm import LPWMFMConfig
    from lerobot.policies.lpwm_fm.modeling_lpwm_fm import LPWMFMPolicy

    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("--checkpoint must be the frozen exported checkpoint directory.")
    hashes = {name: file_sha256(checkpoint / name) for name in REQUIRED_ARTIFACTS}
    experiment = json.loads((checkpoint / "experiment.json").read_text())
    if not isinstance(experiment.get("step"), int) or experiment["step"] < 0:
        raise ValueError("experiment.json must identify a nonnegative checkpoint step.")
    if experiment.get("action_normalization") != "identity":
        raise ValueError("Only exported native-action (identity) checkpoints are supported.")
    config = LPWMFMConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, LPWMFMConfig) or config.action_dim != 7 or config.state_dim != 8:
        raise ValueError("Expected a LIBERO7/canonical8 LPWM-FM checkpoint.")
    if (
        config.normalization_mapping["STATE"] != NormalizationMode.MEAN_STD
        or config.normalization_mapping["VISUAL"] != NormalizationMode.IDENTITY
    ):
        raise ValueError("Expected saved STATE mean/std and VISUAL identity normalization.")
    if config.normalization_mapping["ACTION"] != NormalizationMode.IDENTITY:
        raise ValueError("Checkpoint config disagrees with native-action normalization metadata.")
    keys = list(config.image_features)
    if keys != experiment["camera_keys"]:
        raise ValueError("Checkpoint camera ordering disagrees with its saved experiment metadata.")
    mapping = camera_mapping(keys)
    stats = state_statistics(json.loads((checkpoint / "state_normalization.json").read_text()))
    metadata = json.loads((checkpoint / "language_metadata.json").read_text())
    cache = LanguageCache(
        metadata,
        np.load(checkpoint / "language_embeddings.npy", allow_pickle=False),
        np.load(checkpoint / "language_masks.npy", allow_pickle=False),
        config.language_dim,
    )
    config.device = str(device)
    policy = (
        LPWMFMPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    if hashes != {name: file_sha256(checkpoint / name) for name in REQUIRED_ARTIFACTS}:
        raise RuntimeError("Checkpoint changed while loading; evaluate an immutable watcher snapshot.")
    metadata_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "policy": policy,
        "mapping": mapping,
        "state_stats": stats,
        "language_cache": cache,
        "checkpoint": {
            "path": str(checkpoint),
            "step": experiment["step"],
            "variant": experiment.get("variant"),
            "training_seed": experiment.get("seed"),
            "split_sha256": experiment.get("split_sha256"),
            "model_sha256": hashes["model.safetensors"],
            "artifact_sha256": hashes,
            "bundle_sha256": metadata_hash,
        },
    }


class OptionalVideoWriter:
    """Best-effort video only: unavailable codecs must not erase actual simulator results."""

    def __init__(self, path: Path):
        self.path = path
        self.error = None
        self.writer = None
        try:
            self.writer = VideoWriter(path)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def append_data(self, rgb: np.ndarray) -> None:
        if self.writer is not None and self.error is None:
            try:
                self.writer.append_data(rgb)
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"


class VideoWriter:
    """Optional PyAV H.264 writer: native video dependency is imported only when requested."""

    def __init__(self, path: Path, fps: int = 20):
        from lerobot.utils.import_utils import _av_available, require_package

        if not _av_available:
            require_package("av", extra="video")
        import av

        self.av = av
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.pix_fmt = "yuv420p"
        except Exception:
            self.container.close()
            raise
        self.started = False

    def append_data(self, rgb: np.ndarray) -> None:
        if not self.started:
            self.stream.height, self.stream.width = rgb.shape[:2]
            self.started = True
        frame = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        try:
            if self.started:
                for packet in self.stream.encode():
                    self.container.mux(packet)
        finally:
            self.container.close()


def simulator_api():
    """Preflight headless config before importing LIBERO (upstream may otherwise call input())."""
    config_file = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"Preconfigure LIBERO at {config_file} before evaluation; interactive setup is not permitted."
        )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv, _get_suite

    return LiberoEnv, _get_suite(SUITE), TASK_SUITE_MAX_STEPS[SUITE]


def run_evaluation(args: argparse.Namespace) -> dict:
    """Load one frozen policy and execute a fixed paired simulator protocol across selected tasks."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly use --device cpu if intended.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must select cpu or cuda[:index].")
    if args.init_state_offset < 0:
        raise ValueError("init-state-offset must be nonnegative.")
    if args.episodes_per_task < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("Episode count and optional max-steps must be positive.")
    # Headless config checks happen before policy allocation; simulator remains lazily constructed at reset.
    env_class, suite, suite_steps = simulator_api()
    max_steps = suite_steps if args.max_steps is None else min(args.max_steps, suite_steps)
    task_ids = list(range(len(suite.tasks))) if args.task_ids is None else args.task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique.")
    if any(task < 0 or task >= len(suite.tasks) for task in task_ids):
        raise ValueError(f"Task IDs must lie in [0,{len(suite.tasks) - 1}].")
    bundle = load_checkpoint(args.checkpoint, device)
    policy, mapping, stats = bundle["policy"], bundle["mapping"], bundle["state_stats"]
    # Resolve every requested task BEFORE running any episodes; no partial result for missing task language.
    language_rows = {
        task: bundle["language_cache"].select(suite.get_task(task).language, device) for task in task_ids
    }
    protocol = {
        "name": "lpwm-libero-spatial-rollout-v1",
        "suite": SUITE,
        "seed": args.seed,
        "seed_namespace": args.seed_namespace,
        "task_ids": task_ids,
        "episodes_per_task": args.episodes_per_task,
        "init_state_offset": args.init_state_offset,
        "init_state_offset_semantics": "skip offset entries of the seeded permutation within namespace pool",
        "max_control_steps": max_steps,
        "local_suite_max_steps": suite_steps,
        "horizon_source": "src/lerobot/envs/libero.py:TASK_SUITE_MAX_STEPS[libero_spatial]",
        "requested_max_steps": args.max_steps,
        "control_freq_hz": 20,
        "settle_steps": 10,
        "settle_action": [0, 0, 0, 0, 0, 0, -1],
        "settle_steps_count_toward_horizon": False,
        "init_states": "seeded permutation without replacement; validation first half, final second half",
        "seed_derivation": "sha256(lpwm-libero-rollout-v1:namespace:seed:task_id:episode_index), low32 little-endian",
        "paired_across_variants": True,
        "control_mode": "relative",
        "hard_reset": True,
        "camera_mapping": mapping,
        "raw_render_size": [256, 256],
        "policy_image_size": policy.config.image_size,
        "image_preprocessing": "raw camera HWC uint8 -> rotate180 ONCE -> PIL RGB bilinear resize -> CHW float /255",
        "state_preprocessing": "eef_pos3 + xyzw_quat_to_axisangle3 + gripper_qpos2; (state-saved_mean)/saved_std",
        "language_selection": "exact normalized natural-language description; no task-ID fallback",
        "action_execution": "native LIBERO7, finite validation, clip [-1,1], no gripper inversion",
        "horizon": policy.config.horizon,
        "n_obs_steps": policy.config.n_obs_steps,
        "n_action_steps": policy.config.n_action_steps,
        "flow_inference_steps": policy.config.num_inference_steps,
        "success_definition": "env.step info.is_success; one Bernoulli trial per completed episode",
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "video": bool(args.video),
        "video_policy": "first episode/task; best effort, errors reported per episode",
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": bundle["checkpoint"],
        "protocol": protocol,
        "per_task": [],
    }
    all_episodes = []
    start = time.monotonic()
    for task_id in task_ids:
        env = env_class(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=SUITE,
            episode_length=max_steps,
            camera_name=list(mapping),
            camera_name_mapping=mapping,
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            init_states=True,
            n_envs=1,
            num_steps_wait=10,
            control_freq=20,
            control_mode="relative",
            hard_reset=True,
        )
        language, language_info = language_rows[task_id]
        rows = []
        try:
            total_states = len(env._init_states)
            plan = episode_plan(
                total_states,
                args.episodes_per_task,
                args.seed,
                args.seed_namespace,
                task_id,
                init_state_offset=args.init_state_offset,
            )
            for episode in plan:
                writer, video_path = None, None
                try:
                    if args.video and episode["episode_index"] == 0:
                        video_path = (
                            args.output.parent
                            / f"{args.output.stem}_videos"
                            / f"task_{task_id:02d}_episode_000.mp4"
                        )
                        writer = OptionalVideoWriter(video_path)
                    row = rollout_episode(
                        env,
                        policy,
                        episode,
                        mapping,
                        stats,
                        language,
                        device,
                        max_steps,
                        image_size=policy.config.image_size,
                        video_writer=writer,
                    )
                finally:
                    if writer is not None:
                        writer.close()
                if writer is not None:
                    if writer.error is None:
                        row["video"] = str(video_path)
                    else:
                        row["video_error"] = writer.error
                rows.append(row)
                print(json.dumps({"task_id": task_id, **row}), flush=True)
            task_result = {
                "task_id": task_id,
                "task_description": env.task_description,
                "language": language_info,
                "available_init_states": total_states,
                **summarize_episodes(rows),
                "episodes": rows,
            }
            result["per_task"].append(task_result)
            all_episodes.extend(rows)
        finally:
            env.close()
    result.update(summarize_episodes(all_episodes))
    result["duration_seconds"] = time.monotonic() - start
    result["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    atomic_json(args.output, result)
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Immutable exported checkpoint directory"
    )
    parser.add_argument("--output", type=Path, required=True, help="Atomic result JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument(
        "--task-ids", type=int, nargs="+", help="Spatial task IDs, space separated; default all"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Optional safety cap; cannot extend the local suite horizon"
    )
    parser.add_argument(
        "--init-state-offset",
        type=int,
        default=0,
        help="Skip this many entries of each task namespace init-state permutation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-namespace", choices=("validation", "final"), default="validation")
    parser.add_argument("--video", action="store_true", help="Save the first trajectory of each task as MP4")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    result = run_evaluation(parse_args(argv))
    print(
        json.dumps({key: result[key] for key in ("successes", "num_episodes", "success_rate", "pc_success")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
