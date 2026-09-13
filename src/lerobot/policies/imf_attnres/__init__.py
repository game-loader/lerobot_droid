#!/usr/bin/env python

from .configuration_imf_attnres import IMFAttnResConfig

__all__ = ["IMFAttnResConfig", "IMFAttnResPolicy", "make_imf_attnres_pre_post_processors"]


def __getattr__(name: str):
    if name == "IMFAttnResPolicy":
        from .modeling_imf_attnres import IMFAttnResPolicy

        return IMFAttnResPolicy
    if name == "make_imf_attnres_pre_post_processors":
        from .processor_imf_attnres import make_imf_attnres_pre_post_processors

        return make_imf_attnres_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
