from __future__ import annotations

import pandas as pd
import pytest

from core.config import SignalConfig
from core import regime_ta
from core.signals import momentum_signal, reversal_signal
from core.features_ta import merge_ta_features
from app.models import Metric, upsert_agent_features


def _base_features(**overrides):
    data = {
        "ret_3d": [0.012],
        "vol_spike": [2.0],
        "atr_ratio": [1.5],
        "ret_2d": [-0.02],
        "salience": [2.0],
        "ta_bull_minus_bear": [0.0],
        "ta_risk_earnings_1d": [False],
        "ta_risk_earnings_0d": [False],
        "ta_risk_earnings_3d": [False],
    }
    for key, value in overrides.items():
        data[key] = value
    return pd.DataFrame(data)


def test_momentum_threshold_adjusts_with_positive_ta():
    regime_ta.reset_regime_state()
    feat = _base_features(
        ret_3d=[0.0095],
        ta_bull_minus_bear=[0.25],
    )
    sigs = momentum_signal(feat, SignalConfig())
    assert sigs, "Expected TA boost to create a momentum signal"


def test_ta_risk_flag_blocks_signals():
    regime_ta.reset_regime_state()
    feat = _base_features(ta_risk_earnings_1d=[True])
    assert not momentum_signal(feat, SignalConfig())
    assert not reversal_signal(feat, SignalConfig())


def test_ta_inactive_bypasses_risk_flag():
    regime_ta.reset_regime_state()
    regime_ta.TA_FEATURE_ACTIVE = False
    feat = _base_features(ret_3d=[0.02], ta_risk_earnings_1d=[True])
    sigs = momentum_signal(feat, SignalConfig())
    assert sigs, "Risk flag should not block when TA is inactive"


def test_ta_is_active_regime_logic():
    negative = [-0.01] * 50
    positive = [0.012 + (0.001 if i % 2 == 0 else -0.0005) for i in range(50)]
    assert not regime_ta.ta_is_active(negative)
    assert regime_ta.ta_is_active(positive)


def test_merge_ta_features_creates_columns(db_session):
    payload = {
        "symbol": "AAPL",
        "asof": pd.Timestamp("2024-01-10").date(),
        "bull_score": 0.7,
        "bear_score": 0.4,
        "risk_flags": ["earnings_1d", "macro_vol"],
        "thesis_hash": "demo",
    }
    upsert_agent_features(db_session, [payload])
    df = pd.DataFrame({"close": [100.0], "atr_14": [1.5]}, index=[pd.Timestamp("2024-01-10")])
    merged = merge_ta_features(df, "AAPL", pd.Timestamp("2024-01-10"), db_session)
    assert "ta_bull_minus_bear" in merged
    assert merged["ta_bull_minus_bear"].iloc[-1] == pytest.approx(0.3)
    assert bool(merged["ta_risk_earnings_1d"].iloc[-1]) is True
    assert bool(merged["ta_risk_macro_vol"].iloc[-1]) is True


def test_regime_logging_and_deactivation(db_session):
    regime_ta.reset_regime_state()
    for _ in range(10):
        regime_ta.register_trade_return(-0.02, session=db_session)
    db_session.commit()
    assert regime_ta.TA_FEATURE_ACTIVE is False
    logged = (
        db_session.query(Metric)
        .filter(Metric.name == "ta_active")
        .order_by(Metric.ts.desc())
        .first()
    )
    assert logged is not None
    assert logged.value == 0.0
