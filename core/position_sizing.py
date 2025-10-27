from .config import SizingConfig

def kelly_fraction(p_win: float, payoff: float) -> float:
    q = 1.0 - p_win
    if payoff <= 0:
        return 0.0
    f = p_win - q / payoff
    return max(0.0, f)

def fractional_kelly_equity_fraction(p_win: float, payoff: float, cfg: SizingConfig) -> float:
    base = kelly_fraction(p_win, payoff)
    return cfg.kelly_fraction * base
