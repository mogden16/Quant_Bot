import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, n: int = 14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return pd.Series(tr).rolling(n).mean()

def rsi(series: pd.Series, n: int = 14):
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def rolling_vol(series: pd.Series, n: int = 20):
    return series.pct_change(fill_method=None).rolling(n).std()

def zscore(series: pd.Series, n: int = 20):
    mean = series.rolling(n).mean()
    std = series.rolling(n).std()
    return (series - mean) / (std + 1e-12)
