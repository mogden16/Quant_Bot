import pandas as pd
from .indicators import atr, rsi, zscore

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret_1d'] = out['close'].pct_change(1)
    out['ret_3d'] = out['close'].pct_change(3)
    out['atr_14'] = atr(out, 14)
    out['atr_ratio'] = out['atr_14'] / (out['atr_14'].rolling(20).mean())
    out['vol_spike'] = out['volume'] / (out['volume'].rolling(20).mean())
    out['rsi_14'] = rsi(out['close'], 14)
    out['turnover'] = out['volume']
    out['salience'] = zscore(out['ret_3d'].fillna(0), 60)
    return out
