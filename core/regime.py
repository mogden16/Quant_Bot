import pandas as pd

class RegimeGate:
    def __init__(self, window_trades: int = 50, min_expectancy: float = 0.0, min_sharpe: float = 0.2):
        self.window_trades = window_trades
        self.min_expectancy = min_expectancy
        self.min_sharpe = min_sharpe

    def allow(self, trade_returns: list[float]) -> bool:
        if len(trade_returns) < max(10, self.window_trades//2):
            return True
        s = pd.Series(trade_returns[-self.window_trades:])
        expectancy = s.mean()
        sharpe = s.mean() / (s.std() + 1e-12)
        return (expectancy >= self.min_expectancy) and (sharpe >= self.min_sharpe)
