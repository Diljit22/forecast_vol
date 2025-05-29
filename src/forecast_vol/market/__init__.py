"""
Market session utilities for NYSE calendar.

Generate the official NYSE calendar and expose helpers for downstream
pipelines.

Public API
----------
run_build_sessions  : Build trading day [open_ms, close_ms] mapping
"""

from __future__ import annotations

from .build_sessions import main as run_build_sessions

__all__: list[str] = [
    "run_build_sessions"
    ]
