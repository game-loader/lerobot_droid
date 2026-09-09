# SmolVLA Action Chunk Server

Stateless inference over a persistent binary WebSocket, using a local checkpoint
and its saved processors. Transport is MessagePack; image values are binary
JPEG/PNG bytes with no base64. This avoids base64's roughly 33% size overhead
and reuses the connection; it does not reduce the model's GPU compute time.
No robot connection or motor commands are made by this server.

## Start

Run from the repository root, on the host with NVIDIA driver access:

```bash
uv run --no-sync python -m lerobot.scripts.lerobot_smolvla_server \
  --checkpoint outputs/train/franka_duo_smolvla_512_40k/checkpoints/040000/pretrained_model \
  --device cuda --host 0.0.0.0 --port 8081 --offline
```

Requires `lerobot[smolvla-server]`, including aiohttp and msgpack. To add these
dependencies without removing other installed packages:

```bash
uv sync --locked --inexact --extra training --extra smolvla-server
```

`--offline` requires the checkpoint's SmolVLM backbone and tokenizer already in
the Hugging Face cache. Omit it to allow downloading those dependencies.

The default binding is **0.0.0.0:8081**, accepting external IPv4 connections.
Clients use `ws://<server-LAN-or-tailnet-IP>:8081/infer`, not `0.0.0.0` as the
destination. Allow TCP port 8081 only from trusted clients in the host firewall.
Use `--host 127.0.0.1` for loopback-only access. This lightweight server has
**no authentication or TLS**; do not expose it to the internet. It limits each
request to 16 MiB, each encoded image to 4 MiB, decoded images to 4096*2160
pixels, and WebSocket connections to eight. Oversized messages close with code
1009. Heartbeats detect disconnected peers; WebSocket compression is disabled
because JPEG/PNG are already compressed.

## API

- `GET /health`: returns `{"status": "ok", "busy": false}` after model loading.
- `GET /info`: checkpoint, dimensions, camera mapping, image resize, default task.
- `ws://<host>:8081/infer`: persistent connection for observations/action chunks.

Connect with optional subprotocol `smolvla.msgpack.v1`. Each WebSocket binary
message is exactly one MessagePack map. Pack with `use_bin_type=True`; unpack
with `raw=False`. Responses use the same binary encoding, with float32 actions.
No text JSON, base64, multipart, or pickle input. No unsolicited welcome message.

Example Python payload before MessagePack encoding:

```python
{
  "request_id": "frame-123",
  "state": [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 1, 1],
  "images": {
    "head": head_png_bytes,
    "wrist_left": wrist_left_png_bytes,
    "wrist_right": wrist_right_png_bytes
  },
  "task": "pick cup and bowl"
}
```

`request_id` and `task` are optional. The numeric values above only demonstrate
the schema, not a safe robot pose. Use measured synchronized state and images.
Images must be RGB (not grayscale or RGBA). Original camera resolutions are
accepted. Do not pre-normalize state or
images: the server decodes to CHW float32 `[0,1]`, loads the checkpoint's rename
and normalization/tokenization pipeline, and the policy performs aspect-ratio
preserving resize/pad to **512x512**, then scales pixels to `[-1,1]`.
OpenCV frames are BGR; convert them to RGB before encoding with PIL.

Response fields:

| Field | Meaning |
| --- | --- |
| `request_id` | Echoed identifier, or null |
| `actions` | Float array `[32][20]`, chronological, **unnormalized** |
| `chunk_size` | 32 for this checkpoint |
| `action_dim` | 20 for this checkpoint |
| `inference_ms` | Image decode + preprocessing + GPU inference + CPU postprocessing; excludes network/MessagePack serialization |

Each request calls `predict_action_chunk` once. No action queue is reused across
requests or clients. Sampling is stochastic, so identical inputs can produce
different chunks. Send one observation and await its response before sending the
next. Concurrent/pipelined inference returns a binary error instead of queueing
old observations (including from the same client):

```python
{"request_id": "frame-124", "error": {"code": "busy", "message": "Send the latest observation when ready"}}
```

Other error codes are `invalid_request` and `inference_failed`. A malformed
message does not poison the connection. Errors echo a valid request ID when it
can be decoded; malformed MessagePack uses null. Connection-limit errors use
HTTP 503 before upgrading to WebSocket. Health/info remain available during
GPU inference. There is no HTTP POST inference endpoint.

## This Checkpoint's State and Action Contract

Both state and action use this order (indices are zero-based, slices half-open):

| Slice | Meaning |
| --- | --- |
| `0:3` | Left EE xyz in the dataset's midpoint base frame |
| `3:9` | Left rotation6d: first two rotation-matrix rows, row-major |
| `9:12` | Right EE xyz in the same midpoint base frame |
| `12:18` | Right rotation6d with the same convention |
| `18` | Left gripper, 0=closed, 1=open |
| `19` | Right gripper, 0=closed, 1=open |

These are Cartesian absolute targets, **not joint angles or delta actions**.
The dataset labels each action with the next valid synchronized frame target
after 30 Hz resampling. See
`datasets/franka_duo_lerobot_rgb20d_v1/franka_duo_extras/derived_manifest.json`
for the exact arm-base to midpoint transforms. The caller owns synchronization,
control timing, rot6d-to-rotation conversion, frame transforms, IK if needed,
gripper thresholding/clipping, workspace/velocity/collision checks and emergency
stop. The raw predictions are not guaranteed to be valid rotations or gripper
values in `[0,1]`; the server intentionally does not change learned outputs.

The final checkpoint has **chunk_size=32 and n_action_steps=32**, not a 64-step
prediction horizon. Its state feature metadata still says 6D, inherited from
the base model; trained state mean/std statistics are 20D. The server validates
against those saved statistics, reports the discrepancy through `/info`, and
does not modify the checkpoint or truncate the state.

## Client

Health and checkpoint metadata:

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/info
```

Offline smoke test with a real synchronized dataset frame:

```bash
uv run --no-sync python examples/tutorial/smolvla/inference_client.py \
  --dataset-root datasets/franka_duo_lerobot_rgb20d_v1 --index 0
```

Client using three image files and a JSON state array (no LeRobot needed):

```bash
uv run --no-sync python examples/tutorial/smolvla/inference_client.py \
  --url ws://192.168.50.70:8081/infer \
  --state-json state.json --head head.png \
  --wrist-left wrist_left.png --wrist-right wrist_right.png
```

For a Python sensor loop, keep the client open across observations:

```python
from examples.tutorial.smolvla.inference_client import SmolVLAClient

async with SmolVLAClient("ws://192.168.50.70:8081/infer") as client:
    result = await client.infer(state, {
        "head": head_png_bytes,
        "wrist_left": wrist_left_png_bytes,
        "wrist_right": wrist_right_png_bytes,
    })
    action_chunk = result["actions"]  # [32][20], already unnormalized
    # Reuse client for the next synchronized observation after handling this result.
```

Use `--repeat 5` for repeated offline requests on one connection. CLI output
adds `round_trip_ms`, including MessagePack and socket overhead (but excluding
initial connection establishment and image encoding).

The client closes its socket on timeout/cancellation or mismatched request ID
so a late action cannot be mistaken for the next request's result. Reconnect
explicitly and obtain a new synchronized observation; it does not automatically
retry potentially stale sensor data. Remote camera-file clients require only
aiohttp/msgpack, not PyTorch/LeRobot.

Focused CPU/mock-model tests, including real WebSocket connections:

```bash
uv run --no-sync python -m unittest discover -s tests/scripts -p test_smolvla_server.py -v
```
