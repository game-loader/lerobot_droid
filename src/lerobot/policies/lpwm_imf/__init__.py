"""Encoder-only LPWM-IMF policy; modeling imports remain lazy."""

from .configuration_lpwm_imf import LPWMIMFConfig

__all__ = ["LPWMIMFConfig", "LPWMIMFPolicy", "LPWMVisualEncoder", "LPWMVisualLatents"]


def __getattr__(name: str):
    if name in {"LPWMIMFPolicy", "LPWMVisualEncoder", "LPWMVisualLatents"}:
        from . import modeling_lpwm_imf

        return getattr(modeling_lpwm_imf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
