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

"""Stateless SmolVLA WebSocket chunk inference for trusted local networks.

Run with ``python -m lerobot.scripts.lerobot_smolvla_server --checkpoint PATH``.
See examples/tutorial/smolvla/README_server.md for the wire contract.
"""

import argparse
import asyncio
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 2160
DEFAULT_TASK = "pick cup and bowl"
WS_PROTOCOL = "smolvla.msgpack.v1"


class InvalidRequestError(ValueError):
    """The request cannot be used as an observation."""


class InferenceBusyError(RuntimeError):
    """A previous observation is still being processed."""


def decode_image(encoded: Any) -> torch.Tensor:
    """Decode an RGB JPEG/PNG without resizing; the policy owns resize/pad."""
    if not isinstance(encoded, bytes) or not encoded:
        raise InvalidRequestError("Each image must be nonempty JPEG/PNG bytes (MessagePack bin)")
    if len(encoded) > MAX_IMAGE_BYTES:
        raise InvalidRequestError("Encoded image exceeds the 4 MiB limit")
    try:
        with Image.open(io.BytesIO(encoded), formats=("JPEG", "PNG")) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS or getattr(img, "n_frames", 1) != 1:
                raise InvalidRequestError("Image is too large or animated")
            if img.mode != "RGB":
                raise InvalidRequestError("Images must have three RGB channels (not grayscale, RGBA or BGR)")
            pixels = np.array(img, dtype=np.uint8, copy=True)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, InvalidRequestError):
            raise
        raise InvalidRequestError("Corrupt JPEG/PNG image") from exc
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float().div_(255)


def decode_observation(
    payload: Any, camera_sources: dict[str, str], state_dim: int, default_task: str
) -> tuple[dict[str, Any], str | None]:
    from lerobot.utils.constants import OBS_STATE

    if not isinstance(payload, dict) or set(payload) - {"images", "state", "task", "request_id"}:
        raise InvalidRequestError("Expected an object with images, state, optional task and request_id")
    state = payload.get("state")
    if not isinstance(state, list) or len(state) != state_dim:
        raise InvalidRequestError(f"state must be a flat array of {state_dim} numbers")
    for value in state:
        if type(value) not in (int, float) or not -float(np.finfo(np.float32).max) <= value <= float(
            np.finfo(np.float32).max
        ):
            raise InvalidRequestError("state values must be finite float32 numbers")
    task = payload.get("task", default_task)
    if not isinstance(task, str) or not task.strip() or len(task) > 1024:
        raise InvalidRequestError("task must be a nonempty string of at most 1024 characters")
    request_id = payload.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 128):
        raise InvalidRequestError("request_id must be a string of at most 128 characters")
    images = payload.get("images")
    if not isinstance(images, dict) or set(images) != set(camera_sources):
        raise InvalidRequestError(f"images must contain exactly these cameras: {', '.join(camera_sources)}")
    observation = {OBS_STATE: torch.tensor(state, dtype=torch.float32), "task": task}
    for name, source in camera_sources.items():
        observation[source] = decode_image(images[name])
    return observation, request_id


def _stats_dimension(step: Any, key: str) -> int:
    """Fail closed on missing normalization stats, including stale feature metadata."""
    from lerobot.configs.types import NormalizationMode

    feature = step.features[key]
    if step.norm_map[feature.type] != NormalizationMode.MEAN_STD:
        raise ValueError(f"This server requires saved MEAN_STD normalization for {key}")
    stats = (step.stats or {}).get(key, {})
    mean, std = np.asarray(stats.get("mean")), np.asarray(stats.get("std"))
    if mean.ndim != 1 or not mean.size or std.shape != mean.shape:
        raise ValueError(f"Missing or incompatible mean/std statistics for {key}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError(f"Invalid mean/std statistics for {key}")
    return int(mean.size)


class SmolVLAInference:
    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, checkpoint: str, task: str):
        from lerobot.processor import (
            NormalizerProcessorStep,
            RenameObservationsProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.default_task = task
        self.lock = threading.Lock()
        config = policy.config
        if config.n_obs_steps != 1 or config.adapt_to_pi_aloha or config.rtc_config is not None:
            raise ValueError("This stateless server requires single-observation, non-ALOHA, non-RTC SmolVLA")
        normalizers = [s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)]
        unnormalizers = [s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep)]
        if len(normalizers) != 1 or len(unnormalizers) != 1:
            raise ValueError("Checkpoint must contain exactly one normalizer and one unnormalizer")
        self.state_dim = _stats_dimension(normalizers[0], OBS_STATE)
        self.action_dim = _stats_dimension(unnormalizers[0], ACTION)
        self.chunk_size = config.chunk_size
        if self.state_dim > config.max_state_dim or self.action_dim != config.action_feature.shape[0]:
            raise ValueError("Checkpoint statistics do not match model state/action capacity")

        rename_map = {}
        for step in preprocessor.steps:
            if isinstance(step, RenameObservationsProcessorStep):
                rename_map.update(step.rename_map)
        self.camera_sources = {}
        camera_mapping = {}
        for target in config.image_features:
            sources = [source for source, dest in rename_map.items() if dest == target]
            if len(sources) > 1:
                raise ValueError(f"Ambiguous camera mapping for {target}")
            source = sources[0] if sources else target
            name = source.removeprefix(f"{OBS_IMAGES}.")
            self.camera_sources[name] = source
            camera_mapping[name] = target
        if not self.camera_sources:
            raise ValueError("Checkpoint has no image features")
        warnings = []
        configured_dim = config.input_features[OBS_STATE].shape[0]
        if configured_dim != self.state_dim:
            warnings.append(
                f"Saved state feature says {configured_dim}D; using trained mean/std dimension {self.state_dim}D"
            )
            LOGGER.warning(warnings[-1])
        self.metadata = {
            "policy_type": "smolvla",
            "checkpoint": checkpoint,
            "device": str(config.device),
            "state_dim": self.state_dim,
            "configured_state_dim": configured_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "n_action_steps": config.n_action_steps,
            "resize_imgs_with_padding": config.resize_imgs_with_padding,
            "cameras": camera_mapping,
            "default_task": task,
            "action_normalized": False,
            "stateless": True,
            "transport": "websocket",
            "protocol": WS_PROTOCOL,
            "warnings": warnings,
        }

    @classmethod
    def load(cls, checkpoint: Path, device: str, task: str) -> "SmolVLAInference":
        # Keep optional SmolVLA dependencies out of transport-only imports.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        checkpoint = checkpoint.resolve(strict=True)
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if not isinstance(config, SmolVLAConfig):
            raise ValueError("Checkpoint must be a SmolVLA policy")
        config.device = device
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).eval()
        pre, post = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        return cls(policy, pre, post, str(checkpoint), task)

    def infer(self, payload: Any) -> dict[str, Any]:
        if not self.lock.acquire(blocking=False):
            raise InferenceBusyError("Inference is busy; send the latest observation when ready")
        try:
            started = time.perf_counter()
            observation, request_id = decode_observation(
                payload, self.camera_sources, self.state_dim, self.default_task
            )
            with torch.inference_mode():
                self.policy.reset()
                self.preprocessor.reset()
                self.postprocessor.reset()
                batch = self.preprocessor(observation)
                chunk = self.policy.predict_action_chunk(batch)
                actions = self.postprocessor(chunk).detach().to(device="cpu", dtype=torch.float32)
                if tuple(actions.shape) != (1, self.chunk_size, self.action_dim):
                    raise RuntimeError(f"Unexpected action chunk shape: {tuple(actions.shape)}")
                if not torch.isfinite(actions).all().item():
                    raise RuntimeError("Policy returned nonfinite actions")
            return {
                "request_id": request_id,
                "actions": actions[0].tolist(),
                "chunk_size": self.chunk_size,
                "action_dim": self.action_dim,
                "inference_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        finally:
            self.lock.release()


def create_app(engine: SmolVLAInference):
    """Keep socket I/O responsive while one worker owns GPU inference."""
    import msgpack
    from aiohttp import WSMsgType, web

    sockets = set()
    inflight: asyncio.Task | None = None
    stopping = False

    async def send(ws, payload):
        if not ws.closed:
            try:
                await ws.send_bytes(msgpack.packb(payload, use_bin_type=True, use_single_float=True))
            except ConnectionError:
                LOGGER.info("Client disconnected before response")

    def error(code, message, request_id=None):
        return {"request_id": request_id, "error": {"code": code, "message": message}}

    async def run_inference(ws, payload, request_id):
        try:
            result = await asyncio.to_thread(engine.infer, payload)
        except InvalidRequestError as exc:
            result = error("invalid_request", str(exc), request_id)
        except InferenceBusyError as exc:
            result = error("busy", str(exc), request_id)
        except Exception:
            LOGGER.exception("Inference request failed")
            result = error("inference_failed", "Inspect the server log", request_id)
        await send(ws, result)

    async def websocket(request):
        nonlocal inflight
        if stopping or len(sockets) >= 8:
            raise web.HTTPServiceUnavailable(text="Maximum WebSocket connections reached")
        # Reserve a connection before the upgrade yields to another handler.
        ws = web.WebSocketResponse(
            protocols=(WS_PROTOCOL,), max_msg_size=MAX_BODY_BYTES, heartbeat=20, compress=False
        )
        sockets.add(ws)
        try:
            await ws.prepare(request)
            async for message in ws:
                if message.type != WSMsgType.BINARY:
                    if message.type == WSMsgType.TEXT:
                        await send(ws, error("invalid_request", "Send a binary MessagePack message"))
                    continue
                try:
                    payload = msgpack.unpackb(
                        message.data,
                        raw=False,
                        strict_map_key=True,
                        max_bin_len=MAX_IMAGE_BYTES,
                        max_str_len=4096,
                        max_array_len=4096,
                        max_map_len=32,
                        max_ext_len=0,
                    )
                except (msgpack.UnpackException, ValueError, TypeError):
                    await send(ws, error("invalid_request", "Invalid or oversized MessagePack value"))
                    continue
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                if not isinstance(request_id, str) or len(request_id) > 128:
                    request_id = None
                # Read the next frame during inference so pipelined stale observations are rejected.
                if stopping or (inflight is not None and not inflight.done()):
                    await send(ws, error("busy", "Send the latest observation when ready", request_id))
                    continue
                inflight = asyncio.create_task(run_inference(ws, payload, request_id))
        finally:
            sockets.discard(ws)
        return ws

    async def health(request):
        return web.json_response({"status": "ok", "busy": inflight is not None and not inflight.done()})

    async def info(request):
        return web.json_response(engine.metadata)

    async def shutdown(app):
        nonlocal stopping
        stopping = True
        for ws in list(sockets):
            await ws.close(code=1001, message=b"Server shutdown")
        if inflight is not None:
            await inflight

    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.add_routes([web.get("/infer", websocket), web.get("/health", health), web.get("/info", info)])
    app.on_shutdown.append(shutdown)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="Use cached HF backbone/tokenizer only")
    args = parser.parse_args()
    if (
        args.torch_threads < 1
        or not 1 <= args.port <= 65535
        or not args.task.strip()
        or len(args.task) > 1024
    ):
        parser.error("Invalid port, torch-threads or task")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from lerobot.utils.import_utils import require_package

    require_package("aiohttp", extra="smolvla-server")
    require_package("msgpack", extra="smolvla-server")
    from aiohttp import web

    torch.set_num_threads(args.torch_threads)
    engine = SmolVLAInference.load(args.checkpoint, args.device, args.task)
    LOGGER.info("Starting ws://%s:%s/infer; trusted networks only, no authentication", args.host, args.port)
    web.run_app(create_app(engine), host=args.host, port=args.port, handler_cancellation=False)


if __name__ == "__main__":
    main()
