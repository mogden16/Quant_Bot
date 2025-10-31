"""Runtime helpers for registering an active reinforcement learning policy."""
from __future__ import annotations

from typing import Optional


_ACTIVE_POLICY = None


def set_active_policy(policy) -> None:
    """Register a policy instance that should be used by the signal layer."""

    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy


def clear_active_policy() -> None:
    """Clear the currently registered policy."""

    global _ACTIVE_POLICY
    _ACTIVE_POLICY = None


def get_active_policy():
    """Return the active policy if one has been registered."""

    return _ACTIVE_POLICY

