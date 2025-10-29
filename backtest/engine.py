import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from core.config import RiskConfig, SizingConfig, SignalConfig
from core.features import compute_features
from core.signals import combine_signals
from core.position_sizing import fractional_kelly_equity_fraction
from core.risk import atr_stop_target
from core.regime import RegimeGate
from data.ingest import load_prices


def backtest(symbol: str = "SPY", start: str = "2020-01-01", end: str | None = None, equity: float = 100000.0):
    risk_cfg = RiskConfig()
    size_cfg = SizingConfig()
    sig_cfg = SignalConfig()
    gate = RegimeGate()

    print(f"Downloading data for {symbol} from {start} to {end or 'present'}...")
    df = load_prices(symbol=symbol, start=start, end=end)
    print(f"Loaded {len(df)} bars.")
    feat = compute_features(df)

    trades = []
    open_pos = None
    ret_history = []

    for i in range(30, len(feat)):
        row = feat.iloc[i]
        today_price = row['close']
        today_atr = row['atr_14']

        if open_pos is not None:
            high_mark = max(open_pos['high_mark'], today_price)
            open_pos['high_mark'] = high_mark
            trail = high_mark - risk_cfg.trailing_atr_mult * today_atr
            stop = max(open_pos['stop'], trail)
            open_pos['stop'] = stop

            if today_price <= stop:
                pnl = (stop - open_pos['entry']) * open_pos['shares']
                ret = pnl / equity
                ret_history.append(ret)
                trades.append({**open_pos, "exit_price": stop, "pnl": pnl})
                equity += pnl
                open_pos = None
                continue
            if today_price >= open_pos['target'] or (i - open_pos['i']) >= 5:
                exit_price = today_price
                pnl = (exit_price - open_pos['entry']) * open_pos['shares']
                ret = pnl / equity
                ret_history.append(ret)
                trades.append({**open_pos, "exit_price": exit_price, "pnl": pnl})
                equity += pnl
                open_pos = None
                continue

        if open_pos is None and gate.allow(ret_history):
            sigs = combine_signals(feat.iloc[:i+1].copy(), sig_cfg)
            if not sigs:
                continue
            s = sorted(sigs, key=lambda x: x.strength, reverse=True)[0]
            f_equity = fractional_kelly_equity_fraction(s.p_win, s.payoff, size_cfg)
            f_equity = min(max(f_equity, 0.0), 0.02)
            dollar_risk = equity * f_equity
            stop, target = atr_stop_target(today_price, today_atr, risk_cfg.stop_atr_mult, risk_cfg.target_rr)
            per_share_risk = max(1e-6, today_price - stop)
            shares = max(0, int(dollar_risk / per_share_risk))
            if shares <= 0:
                continue
            open_pos = {
                "i": i,
                "symbol": symbol,
                "entry": float(today_price),
                "shares": shares,
                "stop": float(stop),
                "target": float(target),
                "high_mark": float(today_price),
                "kind": s.kind
            }

    if trades:
        pnl = sum(t["pnl"] for t in trades)
        ret_series = pd.Series([t["pnl"] for t in trades]) / 100000.0
        sharpe = ret_series.mean() / (ret_series.std() + 1e-12) * np.sqrt(252/5)
    else:
        pnl = 0.0
        sharpe = 0.0

    print(f"Symbol: {symbol}")
    print(f"Trades: {len(trades)}, PnL: {pnl:.2f}, Approx Sharpe: {sharpe:.2f}")
    return trades

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading backtest")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to backtest")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    backtest(symbol=args.symbol, start=args.start, end=args.end)
