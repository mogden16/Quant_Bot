import pandas as pd
from dataclasses import dataclass
from .config import SignalConfig

@dataclass
class SignalOutput:
    side: str
    kind: str
    strength: float
    p_win: float
    payoff: float

def momentum_signal(feat: pd.DataFrame, cfg: SignalConfig):
    s = []
    cond = (
        (feat['ret_3d'] > cfg.mom_ret_thresh) &
        (feat['vol_spike'] > cfg.vol_spike_mult) &
        (feat['atr_ratio'] > cfg.atr_mult_thresh)
    )
    if cond.iloc[-1]:
        s.append(SignalOutput(side="long", kind="momentum",
                              strength=float(feat['ret_3d'].iloc[-1]), p_win=0.55, payoff=1.2))
    return s

def reversal_signal(feat: pd.DataFrame, cfg: SignalConfig):
    s = []
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
