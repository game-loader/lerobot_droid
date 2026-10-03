"""Load FastWAM's processor statistics without allocating the multi-billion-parameter policy."""

from lerobot.policies.factory import make_pre_post_processors


def load_processors(config, checkpoint, device="cpu"):
    """Restore BC processors and normalization without constructing policy weights."""
    return make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
