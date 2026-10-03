import asyncio
import copy
import importlib.util
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

try:
    import msgpack
    from aiohttp import WSMsgType, WSServerHandshakeError
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:
    msgpack = None

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.scripts.lerobot_smolvla_server import (
    InferenceBusyError,
    InvalidRequestError,
    SmolVLAInference,
    create_app,
    decode_image,
    decode_observation,
)

CAMERAS = {name: f"observation.images.{name}" for name in ("head", "wrist_left", "wrist_right")}


def encoded_image(color=(255, 0, 0), mode="RGB", fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, (12, 8), color).save(buffer, format=fmt)
    return buffer.getvalue()


def payload():
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    return {
        "state": list(range(20)),
        "images": {name: encoded_image(color) for name, color in zip(CAMERAS, colors, strict=True)},
    }


class FakePolicy:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.resets = 0
        self.batch = None

    def reset(self):
        self.resets += 1

    def eval(self):
        return self

    def predict_action_chunk(self, batch):
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        self.batch = batch
        return torch.full((1, 32, 20), float(self.calls))


def engine():
    state_feature = PolicyFeature(FeatureType.STATE, (6,))  # Stale base-model metadata is intentional.
    action_feature = PolicyFeature(FeatureType.ACTION, (20,))
    image_features = {
        f"observation.images.camera{i}": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for i in (1, 2, 3)
    }
    features = {"observation.state": state_feature, **image_features, "action": action_feature}
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
    }
    stats = {
        "observation.state": {"mean": np.arange(20), "std": np.full(20, 2)},
        "action": {"mean": np.arange(20), "std": np.full(20, 3)},
    }
    pre = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(
                rename_map=dict(zip(CAMERAS.values(), image_features, strict=True))
            ),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats),
        ]
    )
    post = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(features={"action": action_feature}, norm_map=norm_map, stats=stats),
            DeviceProcessorStep(device="cpu"),
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    config = SimpleNamespace(
        input_features=features,
        image_features=image_features,
        action_feature=action_feature,
        n_obs_steps=1,
        adapt_to_pi_aloha=False,
        rtc_config=None,
        max_state_dim=32,
        chunk_size=32,
        n_action_steps=32,
        resize_imgs_with_padding=(512, 512),
        device="cpu",
    )
    return SmolVLAInference(FakePolicy(config), pre, post, "test-checkpoint", "pick cup and bowl")


class ObservationTests(unittest.TestCase):
    def test_rgb_layout_and_no_client_resize(self):
        image = decode_image(encoded_image())
        self.assertEqual(tuple(image.shape), (3, 8, 12))
        self.assertEqual(image.dtype, torch.float32)
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(tuple(decode_image(encoded_image(fmt="JPEG")).shape), (3, 8, 12))

    def test_bad_images(self):
        for value in (None, "bad!", "", encoded_image(0, "L"), encoded_image((0, 0, 0, 0), "RGBA")):
            with self.subTest(value=str(value)[:20]), self.assertRaises(InvalidRequestError):
                decode_image(value)
        with (
            patch("lerobot.scripts.lerobot_smolvla_server.MAX_IMAGE_PIXELS", 2),
            self.assertRaises(InvalidRequestError),
        ):
            decode_image(encoded_image())
        with (
            patch("lerobot.scripts.lerobot_smolvla_server.MAX_IMAGE_BYTES", 2),
            self.assertRaises(InvalidRequestError),
        ):
            decode_image(encoded_image())

    def test_strict_state_and_schema(self):
        for state in (
            [0] * 6,
            [[0]] * 20,
            [True] * 20,
            ["0"] * 20,
            [None] * 20,
            [float("nan")] * 20,
            [float("inf")] * 20,
            [1e100] * 20,
            [10**400] * 20,
        ):
            with self.subTest(state=str(state)[:30]), self.assertRaises(InvalidRequestError):
                decode_observation({**payload(), "state": state}, CAMERAS, 20, "task")
        for field, value in (
            ("task", ""),
            ("task", None),
            ("task", "x" * 1025),
            ("request_id", 5),
            ("images", {}),
            ("images", {**payload()["images"], "extra": "x"}),
        ):
            with self.subTest(field=field), self.assertRaises(InvalidRequestError):
                decode_observation({**payload(), field: value}, CAMERAS, 20, "task")
        for invalid in (None, [], {**payload(), "unexpected": 1}):
            with self.assertRaises(InvalidRequestError):
                decode_observation(invalid, CAMERAS, 20, "task")


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine()

    def test_full_unnormalized_chunk_and_fresh_request(self):
        observation = {**payload(), "request_id": "frame-0"}
        before = copy.deepcopy(observation)
        result = self.engine.infer(observation)
        self.assertEqual(observation, before)
        self.assertEqual(result["request_id"], "frame-0")
        self.assertEqual(np.shape(result["actions"]), (32, 20))
        np.testing.assert_allclose(result["actions"], np.tile(np.arange(20) + 3, (32, 1)))
        self.assertEqual(self.engine.policy.batch["task"], ["pick cup and bowl"])
        torch.testing.assert_close(self.engine.policy.batch["observation.state"], torch.zeros(1, 20))
        self.assertNotIn("observation.images.head", self.engine.policy.batch)
        for i in (1, 2, 3):
            torch.testing.assert_close(
                self.engine.policy.batch[f"observation.images.camera{i}"][0, :, 0, 0], torch.eye(3)[i - 1]
            )
        self.assertEqual(tuple(self.engine.policy.batch["observation.images.camera1"].shape), (1, 3, 8, 12))
        second = self.engine.infer({**payload(), "task": "custom task"})
        np.testing.assert_allclose(second["actions"], np.tile(np.arange(20) + 6, (32, 1)))
        self.assertEqual(self.engine.policy.batch["task"], ["custom task"])
        self.assertEqual(self.engine.policy.calls, 2)
        self.assertEqual(self.engine.policy.resets, 2)
        self.assertEqual(self.engine.metadata["state_dim"], 20)
        self.assertEqual(self.engine.metadata["configured_state_dim"], 6)

    def test_busy_and_lock_recovery(self):
        with self.engine.lock, self.assertRaises(InferenceBusyError):
            self.engine.infer(payload())
        with self.assertRaises(InvalidRequestError):
            self.engine.infer({})
        self.assertFalse(self.engine.lock.locked())
        self.engine.infer(payload())

    def test_invalid_output_is_rejected(self):
        for result in (torch.zeros(1, 1, 20), torch.full((1, 32, 20), float("nan"))):
            with (
                patch.object(self.engine.policy, "predict_action_chunk", return_value=result),
                self.assertRaises(RuntimeError),
            ):
                self.engine.infer(payload())
            self.assertFalse(self.engine.lock.locked())

    def test_missing_stats_fail_closed(self):
        self.engine.preprocessor.steps[-1].stats = {}
        with self.assertRaises(ValueError):
            SmolVLAInference(
                self.engine.policy, self.engine.preprocessor, self.engine.postprocessor, "x", "task"
            )

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "SmolVLA optional dependencies required")
    def test_saved_config_loading_and_device_overrides(self):
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        config = SmolVLAConfig(
            device="cpu",
            input_features={
                k: v for k, v in self.engine.policy.config.input_features.items() if k != "action"
            },
            output_features={"action": self.engine.policy.config.action_feature},
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            with (
                patch(
                    "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained",
                    return_value=self.engine.policy,
                ) as weights,
                patch(
                    "lerobot.policies.factory.make_pre_post_processors",
                    return_value=(self.engine.preprocessor, self.engine.postprocessor),
                ) as processors,
            ):
                loaded = SmolVLAInference.load(Path(directory), "cpu", "pick cup and bowl")
            self.assertIsInstance(weights.call_args.kwargs["config"], SmolVLAConfig)
            self.assertTrue(weights.call_args.kwargs["strict"])
            self.assertEqual(weights.call_args.kwargs["config"].device, "cpu")
            self.assertEqual(
                processors.call_args.kwargs["preprocessor_overrides"], {"device_processor": {"device": "cpu"}}
            )
            self.assertEqual(loaded.state_dim, 20)


@unittest.skipIf(msgpack is None, "Install lerobot[smolvla-server] for WebSocket tests")
class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = engine()
        self.client = TestClient(TestServer(create_app(self.engine)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def receive(self, ws):
        message = await ws.receive(timeout=5)
        self.assertEqual(message.type, WSMsgType.BINARY)
        return msgpack.unpackb(message.data, raw=False)

    async def send(self, ws, observation):
        await ws.send_bytes(msgpack.packb(observation, use_bin_type=True))

    async def test_metadata_and_persistent_client(self):
        from examples.tutorial.smolvla.inference_client import SmolVLAClient

        self.assertEqual(await (await self.client.get("/health")).json(), {"status": "ok", "busy": False})
        info = await (await self.client.get("/info")).json()
        self.assertEqual((info["state_dim"], info["chunk_size"]), (20, 32))
        self.assertEqual(info["transport"], "websocket")
        url = str(self.client.make_url("/infer").with_scheme("ws"))
        async with SmolVLAClient(url) as client:
            connection = client.ws
            for i in (1, 2):
                observation = payload()
                result = await client.infer(observation["state"], observation["images"])
                self.assertEqual(np.shape(result["actions"]), (32, 20))
                np.testing.assert_allclose(result["actions"], np.tile(np.arange(20) + 3 * i, (32, 1)))
                self.assertIs(client.ws, connection)
        self.assertTrue(connection.closed)
        self.assertEqual(self.engine.policy.calls, 2)

    async def test_bad_messages_and_recovery(self):
        ws = await self.client.ws_connect("/infer")
        await ws.send_str("{}")
        self.assertEqual((await self.receive(ws))["error"]["code"], "invalid_request")
        await ws.send_bytes(b"\xc1")
        self.assertEqual((await self.receive(ws))["error"]["code"], "invalid_request")
        for value in (None, {}, {**payload(), "state": [0] * 6}, {**payload(), "images": {}}):
            await self.send(ws, value)
            self.assertEqual((await self.receive(ws))["error"]["code"], "invalid_request")
        with (
            patch.object(self.engine.policy, "predict_action_chunk", side_effect=RuntimeError("test")),
            self.assertLogs("lerobot.scripts.lerobot_smolvla_server", level="ERROR"),
        ):
            await self.send(ws, {**payload(), "request_id": "failed-frame"})
            error = await self.receive(ws)
            self.assertEqual(error["error"]["code"], "inference_failed")
            self.assertEqual(error["request_id"], "failed-frame")
        await self.send(ws, payload())
        self.assertIn("actions", await self.receive(ws))

    async def test_busy_rejects_pipelined_and_other_clients(self):
        entered, release = threading.Event(), threading.Event()
        original = self.engine.policy.predict_action_chunk

        def blocking(batch):
            entered.set()
            release.wait(timeout=5)
            return original(batch)

        ws1 = await self.client.ws_connect("/infer")
        ws2 = await self.client.ws_connect("/infer")
        with patch.object(self.engine.policy, "predict_action_chunk", side_effect=blocking):
            try:
                await self.send(ws1, {**payload(), "request_id": "first"})
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                await self.send(ws1, {**payload(), "request_id": "pipelined"})
                await self.send(ws2, {**payload(), "request_id": "other-client"})
                for ws, expected in ((ws1, "pipelined"), (ws2, "other-client")):
                    result = await self.receive(ws)
                    self.assertEqual(result["error"]["code"], "busy")
                    self.assertEqual(result["request_id"], expected)
                self.assertTrue((await (await self.client.get("/health")).json())["busy"])
            finally:
                release.set()
            result = await self.receive(ws1)
            self.assertEqual(result["request_id"], "first")
        self.assertEqual(self.engine.policy.calls, 1)

    async def test_timeout_closes_client_without_reusing_stale_action(self):
        from examples.tutorial.smolvla.inference_client import SmolVLAClient

        release = threading.Event()
        original = self.engine.policy.predict_action_chunk

        def blocking(batch):
            release.wait(timeout=5)
            return original(batch)

        url = str(self.client.make_url("/infer").with_scheme("ws"))
        with patch.object(self.engine.policy, "predict_action_chunk", side_effect=blocking):
            try:
                async with SmolVLAClient(url, timeout=0.1) as client:
                    with self.assertRaises(TimeoutError):
                        await client.infer(payload()["state"], payload()["images"])
                    self.assertTrue(client.ws.closed)
            finally:
                release.set()
        for _ in range(100):
            if not (await (await self.client.get("/health")).json())["busy"]:
                break
            await asyncio.sleep(0.01)
        self.assertFalse(self.engine.lock.locked())

    async def test_message_limit(self):
        with patch("lerobot.scripts.lerobot_smolvla_server.MAX_BODY_BYTES", 64):
            ws = await self.client.ws_connect("/infer")
        await ws.send_bytes(b"x" * 65)
        result = await ws.receive(timeout=5)
        self.assertEqual(result.type, WSMsgType.CLOSE)
        self.assertEqual(result.data, 1009)
        self.assertEqual(self.engine.policy.calls, 0)

    async def test_connection_limit(self):
        for _ in range(8):
            await self.client.ws_connect("/infer")
        with self.assertRaises(WSServerHandshakeError) as error:
            await self.client.ws_connect("/infer")
        self.assertEqual(error.exception.status, 503)


if __name__ == "__main__":
    unittest.main()
