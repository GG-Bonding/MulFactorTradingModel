"""HTTP API. The engine still decides; this module only stores and reads product state."""

from gold_signal.api.app import create_app

__all__ = ["create_app"]
