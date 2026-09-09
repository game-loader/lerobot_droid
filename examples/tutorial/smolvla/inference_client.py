"""Persistent binary WebSocket client; camera-file input needs only aiohttp/msgpack."""

import argparse
import asyncio
import io
import json
import time
from pathlib import Path


class SmolVLAClient:
    def __init__(self, url="ws://127.0.0.1:8081/infer", timeout=60):
        self.url = url
        self.timeout = timeout
        self.session = None
        self.ws = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        from aiohttp import ClientSession

        self.session = ClientSession()
        try:
            self.ws = await self.session.ws_connect(
                self.url,
                protocols=("smolvla.msgpack.v1",),
                heartbeat=20,
                compress=0,
                max_msg_size=16 * 1024 * 1024,
            )
        except BaseException:
            await self.session.close()
            raise
        return self

    async def __aexit__(self, *args):
        try:
            if self.ws is not None:
                await self.ws.close()
        finally:
            if self.session is not None:
                await self.session.close()

    async def infer(self, state: list[float], images: dict[str, bytes], task="pick cup and bowl") -> dict:
        """Reuse this connection. Images are JPEG/PNG bytes, not raw OpenCV BGR buffers."""
        import msgpack
        from aiohttp import WSMsgType

        if self.ws is None or self.ws.closed:
            raise RuntimeError("Open the client with 'async with SmolVLAClient(...)'")
        if self._lock.locked():
            raise RuntimeError("Wait for the pending inference before sending a new observation")
        async with self._lock:
            request_id = str(time.time_ns())
            payload = {"state": state, "images": images, "task": task, "request_id": request_id}
            try:
                async with asyncio.timeout(self.timeout):
                    await self.ws.send_bytes(msgpack.packb(payload, use_bin_type=True, use_single_float=True))
                    message = await self.ws.receive()
                if message.type != WSMsgType.BINARY:
                    raise RuntimeError(f"WebSocket closed or returned an unexpected frame: {message.type}")
                result = msgpack.unpackb(message.data, raw=False)
                if result.get("request_id") != request_id:
                    raise RuntimeError("Mismatched response request_id; reconnect before retrying")
            except BaseException:
                # A late reply after a timeout/cancellation must not become the next request's action.
                await self.ws.close()
                raise
            if "error" in result:
                raise RuntimeError(f"Server error: {result['error']}")
            return result


def dataset_observation(root: Path, index: int) -> tuple[list[float], dict[str, bytes]]:
    """Offline smoke input only; live deployments should use synchronized sensors."""
    import numpy as np
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/franka_duo_rgb20d", root=root, video_backend="pyav")
    sample = dataset[index]
    images = {}
    for name in ("head", "wrist_left", "wrist_right"):
        chw = sample[f"observation.images.{name}"]
        pixels = chw.permute(1, 2, 0).mul(255).round().clamp(0, 255).numpy().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        images[name] = buffer.getvalue()
    return sample["observation.state"].tolist(), images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8081/infer")
    parser.add_argument("--task", default="pick cup and bowl")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--state-json", type=Path, help="JSON file containing the flat 20D state array")
    parser.add_argument("--head", type=Path)
    parser.add_argument("--wrist-left", type=Path)
    parser.add_argument("--wrist-right", type=Path)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Requests on one connection, for latency checks"
    )
    args = parser.parse_args()
    if args.dataset_root:
        if any((args.state_json, args.head, args.wrist_left, args.wrist_right)):
            parser.error("Choose either dataset input or camera/state files")
        state, images = dataset_observation(args.dataset_root, args.index)
    else:
        if not all((args.state_json, args.head, args.wrist_left, args.wrist_right)):
            parser.error("Provide --dataset-root or all four --state-json/--head/--wrist-left/--wrist-right")
        state = json.loads(args.state_json.read_text())
        images = {name: getattr(args, name).read_bytes() for name in ("head", "wrist_left", "wrist_right")}
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    async def run():
        async with SmolVLAClient(args.url) as client:
            for _ in range(args.repeat):
                started = time.perf_counter()
                result = await client.infer(state, images, args.task)
                result["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 3)
                print(json.dumps(result, allow_nan=False))

    asyncio.run(run())


if __name__ == "__main__":
    main()
