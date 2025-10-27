from dataclasses import dataclass

@dataclass
class RiskConfig:
    max_positions: int = 5
    risk_per_trade: float = 0.005
    stop_atr_mult: float = 1.0
    target_rr: float = 1.5
    trailing_atr_mult: float = 0.7

@dataclass
class SizingConfig:
    kelly_fraction: float = 0.5
    min_edge: float = 0.0

@dataclass
class SignalConfig:
    mom_ret_lookback: int = 3
    mom_ret_thresh: float = 0.01
    rev_ret_lookback: int = 2
    rev_ret_thresh: float = -0.012
    vol_spike_mult: float = 1.5
    atr_mult_thresh: float = 1.2
    salience_z: float = 1.0
