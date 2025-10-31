import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.config import (
    BacktestConfig,
    ExecutionConfig,
    RiskConfig,
    SignalConfig,
    SizingConfig,
)
from core.features import FeatureConfig, compute_features
from core.position_sizing import fractional_kelly_equity_fraction
from core.regime import RegimeGate
from core.risk import atr_stop_target
from core.signals import combine_signals
from data.ingest import INTRADAY_INTERVALS, VALID_INTERVALS, load_prices


def _bars_per_session(interval: str) -> int:
    if interval == "1d":
        return 1
    minutes_map = {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
        "1h": 60,
    }
    minutes = minutes_map.get(interval, 1)
    return max(1, int(round(390 / minutes)))


def _session_minutes(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def _should_skip_entry(ts: pd.Timestamp, cfg: ExecutionConfig, interval: str) -> bool:
    if interval == "1d" or (cfg.skip_first_minutes <= 0 and cfg.skip_last_minutes <= 0):
        return False
    if ts.normalize() == ts and ts.hour == 0 and ts.minute == 0:
        # Fallback data (e.g., demo/Stooq) may not carry intraday timestamps.
        return False
    minute = _session_minutes(ts)
    session_start = 9 * 60 + 30
    session_end = 16 * 60
    if cfg.skip_first_minutes > 0 and minute < session_start + cfg.skip_first_minutes:
        return True
    if cfg.skip_last_minutes > 0 and minute > session_end - cfg.skip_last_minutes:
        return True
    return False


def backtest(
    symbol: str = "SPY",
    start: Optional[str] = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    equity: float = 100000.0,
):
    interval = interval.lower()
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Valid options: {', '.join(VALID_INTERVALS)}")

    risk_cfg = RiskConfig()
    size_cfg = SizingConfig()
    sig_cfg = SignalConfig()
    feat_cfg = FeatureConfig()
    backtest_cfg = BacktestConfig()
    exec_cfg = ExecutionConfig()
    if interval in INTRADAY_INTERVALS:
        exec_cfg = ExecutionConfig(commission_bps=1.5, slippage_bps=2.0, skip_first_minutes=15, skip_last_minutes=10)
        sig_cfg.mom_ret_thresh = max(0.004, sig_cfg.mom_ret_thresh / 2)
        sig_cfg.rev_ret_thresh = min(-0.008, sig_cfg.rev_ret_thresh / 2)
        sig_cfg.salience_z = max(0.5, sig_cfg.salience_z * 0.8)
        sig_cfg.vol_spike_mult = max(1.1, sig_cfg.vol_spike_mult * 0.8)
        sig_cfg.atr_mult_thresh = max(1.0, sig_cfg.atr_mult_thresh * 0.85)

    gate = RegimeGate()

    print(f"Downloading data for {symbol} ({interval})...")
    df = load_prices(symbol=symbol, start=start, end=end, interval=interval)
    print(f"Backtest dataset contains {len(df)} bars.")

    feat = compute_features(df, interval=interval, feature_cfg=feat_cfg)

    trades: list[dict[str, object]] = []
    open_pos: Optional[dict[str, object]] = None
    ret_history: list[float] = []
    per_share_cost = (exec_cfg.commission_bps + exec_cfg.slippage_bps) / 10000.0
    required_cols = ["ret_3d", "atr_14", "vol_spike", "salience"]
    valid_mask = feat[required_cols].notna().all(axis=1).to_numpy() if len(feat) else np.array([])
    if valid_mask.any():
        first_valid = int(np.flatnonzero(valid_mask)[0])
    else:
        first_valid = len(feat)
    warmup_guess = max(30, _bars_per_session(interval) * max(1, feat_cfg.salience_window // 4))
    warmup = min(max(first_valid, warmup_guess), max(len(feat) - 1, 0))
    initial_equity = equity

    for i in range(int(warmup), len(feat)):
        row = feat.iloc[i]
        today_price = float(row["close"])
        today_atr = float(row["atr_14"])
        ts = feat.index[i]

        if open_pos is not None:
            bars_held = i - int(open_pos["i"])
            high_mark = max(float(open_pos["high_mark"]), today_price)
            open_pos["high_mark"] = high_mark
            trail = high_mark - risk_cfg.trailing_atr_mult * today_atr
            stop_price = max(float(open_pos["stop"]), trail)
            open_pos["stop"] = stop_price

            hit_stop = today_price <= stop_price
            target_hit = today_price >= float(open_pos["target"])
            expired = (
                interval in INTRADAY_INTERVALS and bars_held >= backtest_cfg.max_holding_bars
            )
            if (
                not expired
                and backtest_cfg.flatten_eod
                and interval in INTRADAY_INTERVALS
                and i + 1 < len(feat)
            ):
                next_ts = feat.index[i + 1]
                if next_ts.date() != ts.date():
                    expired = True

            if hit_stop or target_hit or expired:
                exit_price = stop_price if hit_stop else today_price
                entry_price = float(open_pos["entry"])
                shares = int(open_pos["shares"])
                total_cost = (entry_price + exit_price) * per_share_cost * shares
                pnl = (exit_price - entry_price) * shares - total_cost
                equity += pnl
                ret = pnl / max(initial_equity, 1e-9)
                ret_history.append(ret)
                trades.append({**open_pos, "exit_price": float(exit_price), "pnl": float(pnl)})
                open_pos = None
                continue

        if open_pos is None and gate.allow(ret_history):
            if _should_skip_entry(ts, exec_cfg, interval):
                continue
            sigs = combine_signals(feat.iloc[: i + 1].copy(), sig_cfg)
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
                "kind": s.kind,
                "timestamp": ts,
            }

    if open_pos is not None:
        exit_price = float(feat.iloc[-1]["close"])
        entry_price = float(open_pos["entry"])
        shares = int(open_pos["shares"])
        total_cost = (entry_price + exit_price) * per_share_cost * shares
        pnl = (exit_price - entry_price) * shares - total_cost
        equity += pnl
        ret = pnl / max(initial_equity, 1e-9)
        ret_history.append(ret)
        trades.append({**open_pos, "exit_price": exit_price, "pnl": float(pnl)})
        open_pos = None

    if trades:
        pnl = float(sum(t["pnl"] for t in trades))
        ret_series = pd.Series([t["pnl"] for t in trades], dtype=float) / max(initial_equity, 1e-9)
        bars_per_year = 252 * _bars_per_session(interval)
        sharpe = float(ret_series.mean() / (ret_series.std(ddof=0) + 1e-12) * np.sqrt(bars_per_year))
    else:
        pnl = 0.0
        sharpe = 0.0

    print(f"Symbol: {symbol} | Interval: {interval} | Bars: {len(df)}")
    print(f"Trades: {len(trades)}, PnL: {pnl:.2f}, Approx Sharpe: {sharpe:.2f}")
    return trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading backtest")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to backtest")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--interval",
        default="1d",
        choices=VALID_INTERVALS,
        help="Bar interval (e.g. 1m, 5m, 1d)",
    )
    args = parser.parse_args()

    backtest(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval)
