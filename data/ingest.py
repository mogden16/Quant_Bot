from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}
VALID_INTERVALS = tuple(sorted(INTRADAY_INTERVALS | {"1d"}))


def demo_price_series(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    shock_idx = rng.choice(n, size=max(5, n // 50), replace=False)
    rets[shock_idx] += rng.normal(0.02, 0.03, len(shock_idx))
    price = 100 * (1 + pd.Series(rets)).cumprod()
    dt = pd.date_range("2023-01-01", periods=n, freq="B")
    price.index = dt
    high = price * (1 + rng.uniform(0, 0.01, n))
    low = price * (1 - rng.uniform(0, 0.01, n))
    openp = price.shift(1).fillna(price.iloc[0])
    volume = rng.integers(100000, 500000, n)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": price, "volume": volume},
        index=dt,
    )


def _interval_to_yf(interval: str) -> str:
    if interval == "1h":
        return "60m"
    return interval


def _default_period(interval: str) -> str:
    if interval == "1m":
        return "7d"
    if interval == "2m":
        return "60d"
    if interval in {"5m", "15m", "30m"}:
        return "60d"
    if interval in {"60m", "1h"}:
        return "730d"
    return "max"


def _normalize_price_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
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
        "Adj Close": "adj_close",
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
        raise ValueError(f"missing columns {missing_cols}")

    data = data[required_cols].dropna()
    if data.empty:
        raise ValueError("all data rows contained NaNs")

    return data.copy()


def _max_period_days(interval: str) -> Optional[int]:
    return {
        "1m": 7,
        "2m": 60,
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "60m": 730,
        "1h": 730,
    }.get(interval)


def _infer_period_from_dates(start: Optional[str], end: Optional[str], interval: str) -> Optional[str]:
    if start is None and end is None:
        return None
    try:
        start_dt = pd.to_datetime(start) if start is not None else None
        end_dt = pd.to_datetime(end) if end is not None else pd.Timestamp(datetime.utcnow())
    except Exception:
        return None

    if start_dt is None:
        return None

    delta_days = max((end_dt - start_dt).days, 1)
    max_days = _max_period_days(interval)
    if max_days is not None and delta_days > max_days:
        delta_days = max_days
    return f"{delta_days}d"


def load_prices(
    symbol: str = "SPY",
    start: Optional[str] = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    interval = interval.lower()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Valid options: {', '.join(VALID_INTERVALS)}")

    yf_interval = _interval_to_yf(interval)
    is_intraday = yf_interval != "1d"

    try:
        import yfinance as yf
    except ImportError:
        print("Warning: yfinance is not installed; using demo price series instead.")
        df_demo = demo_price_series(n=600)
        print(f"Loaded demo data ({len(df_demo)} bars).")
        return df_demo

    yf_kwargs: dict[str, object] = {
        "interval": yf_interval,
        "auto_adjust": False,
        "progress": False,
    }

    if is_intraday:
        period = _infer_period_from_dates(start, end, interval) or _default_period(interval)
        if start is not None or end is not None:
            print(
                f"Info: intraday data requested with start/end; using period={period} for Yahoo Finance download."
            )
        yf_kwargs["period"] = period
        yf_kwargs.pop("auto_adjust", None)
    else:
        yf_kwargs["start"] = start
        yf_kwargs["end"] = end

    try:
        data = yf.download(symbol, **yf_kwargs)
    except Exception as err:  # pragma: no cover - network failure path
        print(f"Warning: failed to download {symbol} from Yahoo Finance ({err}); attempting secondary source.")
        data = pd.DataFrame()

    source_name = "Yahoo"
    if data is None or data.empty:
        try:
            from pandas_datareader import data as pdr

            stooq_df = pdr.DataReader(symbol, "stooq", start=start, end=end)
            data = stooq_df.sort_index()
            source_name = "Stooq"
            if is_intraday:
                print(
                    "Warning: Stooq provides daily bars; consider adjusting expectations for intraday tests."
                )
        except Exception as err:  # pragma: no cover - network failure path
            print(
                "Warning: secondary source (Stooq) failed to load data for "
                f"{symbol} ({err}); using demo data instead."
            )
            demo = demo_price_series(n=600)
            print(f"Loaded demo data ({len(demo)} bars).")
            return demo

    try:
        normalized = _normalize_price_frame(data, symbol)
    except ValueError as err:
        print(f"Warning: data normalization failed for {symbol} ({err}); using demo data instead.")
        demo = demo_price_series(n=600)
        print(f"Loaded demo data ({len(demo)} bars).")
        return demo

    print(f"Loaded {len(normalized)} bars for {symbol} from {source_name} using interval={interval}.")
    return normalized
