import numpy as np
import pytest

from RL.adapters.real_robot import RealRobotEnvAdapter, validate_vector_env


def test_real_robot_adapter_batches_and_converts_observation() -> None:
    sent: list[np.ndarray] = []

    def observation() -> dict[str, np.ndarray]:
        return {
            "observation.state": np.zeros(34, dtype=np.float32),
            "observation.point_cloud": np.zeros((8, 3), dtype=np.float32),
            "observation.images.wrist_left": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation.images.wrist_right": np.zeros((3, 6, 3), dtype=np.uint8),
        }

    env = RealRobotEnvAdapter(
        read_observation=observation,
        send_action=lambda action: sent.append(action),
        reward_fn=lambda *_args: 0.25,
        terminated_fn=lambda *_args: True,
    )

    reset_observation, _ = env.reset()
    assert reset_observation["observation.state"].shape == (1, 34)
    assert reset_observation["observation.point_cloud"].shape == (1, 8, 3)
    assert reset_observation["observation.images.wrist_left"].shape == (1, 3, 4, 5)
    assert reset_observation["observation.images.wrist_left"].dtype == np.float32
    assert float(reset_observation["observation.images.wrist_left"].max()) == 0.0

    next_observation, reward, terminated, truncated, info = env.step(np.zeros((1, 18)))
    assert next_observation["observation.state"].shape == (1, 34)
    np.testing.assert_allclose(reward, [0.25])
    np.testing.assert_array_equal(terminated, [True])
    np.testing.assert_array_equal(truncated, [False])
    assert info == {}
    np.testing.assert_allclose(sent[0], np.zeros((1, 18))[0])
    validate_vector_env(env, expected_num_envs=1)


def test_real_robot_adapter_requires_reset_before_step() -> None:
    env = RealRobotEnvAdapter(
        read_observation=lambda: {"observation.state": np.zeros(34, dtype=np.float32)},
        send_action=lambda _action: None,
    )
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(18, dtype=np.float32))


def test_validate_vector_env_rejects_wrong_world_count() -> None:
    class Env:
        num_envs = 2

        def reset(self, **_kwargs):
            return {}

        def step(self, _action):
            return ()

        def close(self):
            return None

    with pytest.raises(ValueError, match="num_envs"):
        validate_vector_env(Env(), expected_num_envs=1)
