import pandas as pd
import numpy as np

def demo_price_series(n=300, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    shock_idx = rng.choice(n, size=max(5, n//50), replace=False)
    rets[shock_idx] += rng.normal(0.02, 0.03, len(shock_idx))
    price = 100 * (1 + pd.Series(rets)).cumprod()
    high = price * (1 + rng.uniform(0, 0.01, n))
    low = price * (1 - rng.uniform(0, 0.01, n))
    openp = price.shift(1).fillna(price.iloc[0])
    volume = rng.integers(100000, 500000, n)
    dt = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": price, "volume": volume}, index=dt)

def load_prices(symbol: str) -> pd.DataFrame:
    return demo_price_series()
