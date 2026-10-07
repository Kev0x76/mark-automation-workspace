"""Embedded V5 verification engine.

This package consumes immutable file snapshots and never imports generator
production services or normalized generator state as verification truth.
"""

from .service import run_verification

__all__ = ["run_verification"]
