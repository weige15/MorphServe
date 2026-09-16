"""Normalized package entry point for the candidate MorphServe source."""

from .engine_config import EngineConfig

__all__ = ["EngineConfig", "Engine", "RawRequest", "LlamaModel"]


def __getattr__(name):
    if name == "Engine":
        from .server.engine import Engine
        return Engine
    if name == "RawRequest":
        from .server.structs import RawRequest
        return RawRequest
    if name == "LlamaModel":
        from .worker.model import LlamaModel
        return LlamaModel
    raise AttributeError(name)
