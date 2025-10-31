from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pandas as pd

from .indicators import atr, rsi, zscore


_MINUTES_PER_BAR = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1h": 60,
}


@dataclass
class FeatureConfig:
    """Feature defaults expressed in *daily* bars.

    For intraday intervals, lookbacks scale by the approximate number of bars
    per trading session (390 minutes).
    """

    ret_short_lookback: int = 1
    ret_medium_lookback: int = 3
    atr_window: int = 14
    atr_smooth_window: int = 20
    volume_window: int = 20
    salience_window: int = 60
    rsi_window: int = 14


def _bars_per_session(interval: str) -> int:
    minutes = _MINUTES_PER_BAR.get(interval)
    if minutes is None:
        return 1
    return max(1, int(round(390 / minutes)))


def _scale_lookback(base: int, interval: str, max_length: int) -> int:
    if interval == "1d":
        return max(1, min(base, max_length))
    scale = max(1, int(round(sqrt(_bars_per_session(interval)))))
    scaled = base * scale
    return max(1, min(scaled, max_length))


def compute_features(
    df: pd.DataFrame,
    interval: str = "1d",
    feature_cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    cfg = feature_cfg or FeatureConfig()
    out = df.copy()

    max_len = max(len(df), 1)

    def _cap(base: int) -> int:
        half = max_len // 2 if max_len > 1 else base
        return max(base, half)

    short_lb = _scale_lookback(cfg.ret_short_lookback, interval, _cap(cfg.ret_short_lookback))
    medium_lb = _scale_lookback(cfg.ret_medium_lookback, interval, _cap(cfg.ret_medium_lookback))
    atr_window = _scale_lookback(cfg.atr_window, interval, _cap(cfg.atr_window))
    atr_smooth = _scale_lookback(cfg.atr_smooth_window, interval, _cap(cfg.atr_smooth_window))
    volume_window = _scale_lookback(cfg.volume_window, interval, _cap(cfg.volume_window))
    salience_window = _scale_lookback(cfg.salience_window, interval, _cap(cfg.salience_window))
    rsi_window = _scale_lookback(cfg.rsi_window, interval, _cap(cfg.rsi_window))

    out.loc[:, "ret_1d"] = out["close"].pct_change(short_lb, fill_method=None)
    out.loc[:, "ret_3d"] = out["close"].pct_change(medium_lb, fill_method=None)
    out.loc[:, "atr_14"] = atr(out, atr_window)
    out.loc[:, "atr_ratio"] = out["atr_14"] / out["atr_14"].rolling(atr_smooth, min_periods=1).mean()
    out.loc[:, "vol_spike"] = out["volume"] / out["volume"].rolling(volume_window, min_periods=1).mean()
    out.loc[:, "rsi_14"] = rsi(out["close"], rsi_window)
    out.loc[:, "turnover"] = out["volume"]
    out.loc[:, "salience"] = zscore(out["ret_3d"].fillna(0), salience_window)
    return out
