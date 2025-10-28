from dataclasses import dataclass

import pandas as pd

from .config import (
    SignalConfig,
    TA_BLOCK_EARNINGS_WINDOWS,
    TA_FEATURE_MIN_BMB,
)
from .regime_ta import is_ta_enabled

@dataclass
class SignalOutput:
    side: str
    kind: str
    strength: float
    p_win: float
    payoff: float

def _ta_allows_long(feat: pd.DataFrame) -> bool:
    if not is_ta_enabled():
        return True
    for window in TA_BLOCK_EARNINGS_WINDOWS:
        col = f"ta_risk_{window}"
        if col in feat.columns and bool(feat[col].iloc[-1]):
            return False
    return True


def _ta_adjusted_momentum_threshold(feat: pd.DataFrame, base_thresh: float) -> float:
    if not is_ta_enabled():
        return base_thresh
    series = feat.get("ta_bull_minus_bear")
    if series is None or series.empty:
        return base_thresh
    latest = float(series.iloc[-1])
    if latest > TA_FEATURE_MIN_BMB:
        return max(base_thresh - 0.002, 0.004)
    return base_thresh


def momentum_signal(feat: pd.DataFrame, cfg: SignalConfig):
    s = []
    if not _ta_allows_long(feat):
        return s

    mom_thresh = _ta_adjusted_momentum_threshold(feat, cfg.mom_ret_thresh)
    cond = (
        (feat['ret_3d'] > mom_thresh) &
        (feat['vol_spike'] > cfg.vol_spike_mult) &
        (feat['atr_ratio'] > cfg.atr_mult_thresh)
    )
    if cond.iloc[-1]:
        s.append(SignalOutput(side="long", kind="momentum",
                              strength=float(feat['ret_3d'].iloc[-1]), p_win=0.55, payoff=1.2))
    return s

def reversal_signal(feat: pd.DataFrame, cfg: SignalConfig):
    s = []
    if not _ta_allows_long(feat):
        return s
    if 'ret_2d' not in feat:
        feat['ret_2d'] = feat['close'].pct_change(2)
    cond = (feat['ret_2d'] < cfg.rev_ret_thresh) & (feat['salience'] > cfg.salience_z)
    if cond.iloc[-1]:
        s.append(SignalOutput(side="long", kind="reversal",
                              strength=float(abs(feat['ret_2d'].iloc[-1])), p_win=0.53, payoff=1.1))
    return s

def combine_signals(feat: pd.DataFrame, cfg: SignalConfig):
    out = []
    out += momentum_signal(feat, cfg)
    out += reversal_signal(feat, cfg)
    return out
