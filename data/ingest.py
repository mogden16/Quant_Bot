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

def load_prices(symbol: str = "SPY", start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("Warning: yfinance is not installed; using demo price series instead.")
        return demo_price_series(n=600)

    try:
        data = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as err:  # pragma: no cover - network failure path
        print(f"Warning: failed to download {symbol} from Yahoo Finance ({err}); using demo data.")
        return demo_price_series(n=600)

    if data is None or data.empty:
        print(f"Warning: no data returned for {symbol}; using demo data instead.")
        return demo_price_series(n=600)

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    data = data.sort_index()

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    if isinstance(data.columns, pd.MultiIndex):
        if symbol in data.columns.get_level_values(0):
            data = data.xs(symbol, axis=1, level=0)
        elif symbol in data.columns.get_level_values(-1):
            data = data.xs(symbol, axis=1, level=-1)
        else:
            data = data.droplevel(0, axis=1)

    data = data.rename(columns=rename_map)
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        print(f"Warning: missing columns {missing_cols} for {symbol}; using demo data instead.")
        return demo_price_series(n=600)

    data = data[required_cols].dropna()
    if data.empty:
        print(f"Warning: all data rows contained NaNs for {symbol}; using demo data instead.")
        return demo_price_series(n=600)

    return data.copy()
