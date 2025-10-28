"""Regime checks for enabling/disabling TradingAgents feature influences."""
from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Deque, List, Sequence

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from app.models import Metric
from core.config import ENABLE_TA_FEATURES

TA_FEATURE_ACTIVE: bool = True
_TRADE_RETURNS: Deque[float] = deque(maxlen=50)
_EVAL_INTERVAL = 10


def ta_is_active(trade_returns: Sequence[float]) -> bool:
    """Determine if TA features should remain active based on trade returns."""
    if not ENABLE_TA_FEATURES:
        return False
    window = list(trade_returns)[-50:]
    if len(window) < 10:
        return True
    arr = np.asarray(window, dtype=float)
    if np.allclose(arr, 0.0):
        return False
    std = arr.std(ddof=1)
    if np.isclose(std, 0.0):
        return False
    mean = arr.mean()
    sharpe = sqrt(len(arr)) * mean / std
    _, p_value = stats.ttest_1samp(arr, popmean=0.0, alternative="greater")
    if np.isnan(p_value):
        p_value = 1.0
    if sharpe <= 0 or p_value >= 0.05:
        return False
    return True


def is_ta_enabled() -> bool:
    """Return whether TA features are currently allowed to influence signals."""
    return ENABLE_TA_FEATURES and TA_FEATURE_ACTIVE


def reset_regime_state() -> None:
    """Reset regime evaluation state (intended for testing)."""
    global TA_FEATURE_ACTIVE
    TA_FEATURE_ACTIVE = True
    _TRADE_RETURNS.clear()


def _record_metric(active: bool, session: Session | None) -> None:
    if session is None:
        return
    session.add(Metric(name="ta_active", value=1.0 if active else 0.0))
    session.flush()


def register_trade_return(trade_return: float, session: Session | None = None) -> bool:
    """Register a TA-conditioned trade return and update the active flag."""
    global TA_FEATURE_ACTIVE

    _TRADE_RETURNS.append(float(trade_return))
    if len(_TRADE_RETURNS) < _EVAL_INTERVAL:
        TA_FEATURE_ACTIVE = True
        return TA_FEATURE_ACTIVE

    if len(_TRADE_RETURNS) % _EVAL_INTERVAL == 0 or not TA_FEATURE_ACTIVE:
        new_state = ta_is_active(list(_TRADE_RETURNS))
        if new_state != TA_FEATURE_ACTIVE:
            TA_FEATURE_ACTIVE = new_state
            _record_metric(TA_FEATURE_ACTIVE, session)
    return TA_FEATURE_ACTIVE


def recent_trade_returns() -> List[float]:
    """Return recent trade returns considered for the regime filter."""
    return list(_TRADE_RETURNS)


__all__ = [
    "TA_FEATURE_ACTIVE",
    "is_ta_enabled",
    "recent_trade_returns",
    "register_trade_return",
    "reset_regime_state",
    "ta_is_active",
]
