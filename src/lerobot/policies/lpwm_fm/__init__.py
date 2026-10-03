"""LPWM + ordinary flow matching, with optional GT-action world-model supervision."""

from .configuration_lpwm_fm import LPWMFMConfig

__all__ = ["LPWMFMConfig", "LPWMFMPolicy"]


def __getattr__(name: str):
    if name == "LPWMFMPolicy":
        from .modeling_lpwm_fm import LPWMFMPolicy

        return LPWMFMPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
