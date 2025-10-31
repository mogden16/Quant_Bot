"""Ablation backtest comparing baseline signals with TA-augmented variants."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from agents.ta_client import fetch_batch
from app.db import SessionLocal, init_db
from app.models import upsert_agent_features
from core import regime_ta
import core.config as cfg
from core.config import RiskConfig, SignalConfig, SizingConfig
from core.features import compute_features
from core.features_ta import merge_ta_features
from core.position_sizing import fractional_kelly_equity_fraction
from core.regime import RegimeGate
from core.risk import atr_stop_target
from core.signals import combine_signals
from data.ingest import load_prices

INITIAL_EQUITY = 100_000.0


@dataclass
class BacktestTrade:
    symbol: str
    entry_dt: pd.Timestamp
    exit_dt: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    kind: str


@dataclass
class BacktestResult:
    trades: List[BacktestTrade]
    returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]
    equity_index: List[pd.Timestamp]
    metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate TA feature impact on the strategy.")
    parser.add_argument("--symbols", type=str, default="DEMO", help="Comma separated symbol list")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--enable-ta", type=str, default="true", help="Whether to run TA ablation (true/false)")
    parser.add_argument("--mock-ta", type=str, default="false", help="Force TA mock mode (true/false)")
    parser.add_argument("--use-rl", type=str, default="false", help="Evaluate an RL policy (true/false)")
    parser.add_argument("--rl-policy", type=str, default=None, help="Path to a saved RL policy checkpoint")
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def ensure_agent_features(session, symbols: Sequence[str], dates: Iterable[pd.Timestamp]) -> None:
    unique_dates = sorted({pd.Timestamp(d).normalize().date() for d in dates})
    for asof in unique_dates:
        payloads = fetch_batch(symbols, asof)
        upsert_agent_features(session, payloads)
    session.commit()


def _compute_equity_curve(returns: Sequence[float]) -> Tuple[List[float], List[float]]:
    equity = [INITIAL_EQUITY]
    drawdown = [0.0]
    peak = INITIAL_EQUITY
    for r in returns:
        next_equity = equity[-1] * (1 + r)
        equity.append(next_equity)
        peak = max(peak, next_equity)
        dd = 0.0 if peak <= 0 else (next_equity - peak) / peak
        drawdown.append(dd)
    return equity, drawdown


def _compute_metrics(trades: List[BacktestTrade], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    returns = [t.return_pct for t in trades]
    equity_curve, drawdown_curve = _compute_equity_curve(returns)
    duration_days = max((end - start).days, 1)
    duration_years = duration_days / 365.25
    total_return = 0.0 if not equity_curve else equity_curve[-1] / INITIAL_EQUITY - 1.0
    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (mean_return / std_return * np.sqrt(len(returns))) if std_return > 0 else 0.0
    downside = [min(0.0, r) for r in returns]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_return / abs(downside_std) * np.sqrt(len(returns))) if downside_std != 0 else 0.0
    win_rate = float(sum(r > 0 for r in returns) / len(returns)) if returns else 0.0
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = sum(t.pnl for t in trades if t.pnl < 0)
    profit_factor = float(gains / abs(losses)) if losses < 0 else float("inf") if gains > 0 else 0.0
    expectancy = mean_return
    stdev = std_return
    max_dd = float(min(drawdown_curve)) if drawdown_curve else 0.0
    cagr = ((equity_curve[-1] / INITIAL_EQUITY) ** (1 / duration_years) - 1) if returns else 0.0
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Stdev": stdev,
        "MaxDD": max_dd,
        "Win%": win_rate,
        "ProfitFactor": profit_factor,
        "Expectancy": expectancy,
        "Trades": float(len(trades)),
        "TotalReturn": total_return,
    }


def _build_plot(equity_curve: Sequence[float], drawdown_curve: Sequence[float], x_axis: Sequence[pd.Timestamp], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=equity_curve, mode="lines", name="Equity"))
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=drawdown_curve,
            mode="lines",
            name="Drawdown",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Trade",
        yaxis_title="Equity",
        yaxis2=dict(title="Drawdown", overlaying="y", side="right"),
        template="plotly_white",
    )
    return fig


def _ensure_ta_state(enabled: bool):
    cfg.ENABLE_TA_FEATURES = enabled
    if not enabled:
        regime_ta.reset_regime_state()


def run_backtest(symbols: Sequence[str], price_data: Dict[str, pd.DataFrame], use_ta: bool, session) -> BacktestResult:
    risk_cfg = RiskConfig()
    size_cfg = SizingConfig()
    sig_cfg = SignalConfig()
    gate = RegimeGate()

    trades: List[BacktestTrade] = []
    returns: List[float] = []
    equity = INITIAL_EQUITY

    for symbol in symbols:
        df = price_data[symbol]
        feat = compute_features(df)
        open_pos = None

        for i in range(30, len(feat)):
            window = feat.iloc[: i + 1]
            if use_ta:
                window = merge_ta_features(window, symbol, window.index[-1], session)
            row = window.iloc[-1]
            today_price = row["close"]
            today_atr = row["atr_14"]
            if pd.isna(today_atr):
                continue

            if open_pos is not None:
                high_mark = max(open_pos["high_mark"], today_price)
                open_pos["high_mark"] = high_mark
                trail = high_mark - risk_cfg.trailing_atr_mult * today_atr
                stop = max(open_pos["stop"], trail)
                open_pos["stop"] = stop

                if today_price <= stop:
                    pnl = (stop - open_pos["entry"]) * open_pos["shares"]
                    ret = pnl / equity
                    returns.append(ret)
                    trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            entry_dt=open_pos["entry_dt"],
                            exit_dt=window.index[-1],
                            entry_price=open_pos["entry"],
                            exit_price=float(stop),
                            pnl=float(pnl),
                            return_pct=float(ret),
                            kind=open_pos["kind"],
                        )
                    )
                    equity += pnl
                    if use_ta:
                        regime_ta.register_trade_return(ret, session=session)
                    open_pos = None
                    continue

                if today_price >= open_pos["target"] or (i - open_pos["i"]) >= 5:
                    exit_price = float(today_price)
                    pnl = (exit_price - open_pos["entry"]) * open_pos["shares"]
                    ret = pnl / equity
                    returns.append(ret)
                    trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            entry_dt=open_pos["entry_dt"],
                            exit_dt=window.index[-1],
                            entry_price=open_pos["entry"],
                            exit_price=exit_price,
                            pnl=float(pnl),
                            return_pct=float(ret),
                            kind=open_pos["kind"],
                        )
                    )
                    equity += pnl
                    if use_ta:
                        regime_ta.register_trade_return(ret, session=session)
                    open_pos = None
                    continue

            if open_pos is None and gate.allow(returns):
                sigs = combine_signals(window, sig_cfg)
                if not sigs:
                    continue
                s = sorted(sigs, key=lambda x: x.strength, reverse=True)[0]
                f_equity = fractional_kelly_equity_fraction(s.p_win, s.payoff, size_cfg)
                f_equity = min(max(f_equity, 0.0), 0.02)
                stop, target = atr_stop_target(today_price, today_atr, risk_cfg.stop_atr_mult, risk_cfg.target_rr)
                per_share_risk = max(1e-6, today_price - stop)
                dollar_risk = equity * f_equity
                shares = int(dollar_risk / per_share_risk)
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
                    "entry_dt": window.index[-1],
                }

    start = min(df.index[0] for df in price_data.values())
    end = max(df.index[-1] for df in price_data.values())
    metrics = _compute_metrics(trades, start, end)
    equity_curve, drawdown_curve = _compute_equity_curve(returns)
    equity_index = [start] + [trade.exit_dt for trade in trades]
    return BacktestResult(trades, returns, equity_curve, drawdown_curve, equity_index, metrics)


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    use_rl = _parse_bool(args.use_rl)
    rl_policy_path = args.rl_policy

    if _parse_bool(args.mock_ta):
        os.environ["TA_MOCK"] = "1"

    init_db()
    session = SessionLocal()

    clear_active_policy = None
    if use_rl:
        if rl_policy_path is None:
            raise SystemExit("--rl-policy must be provided when --use-rl is true")
        from rl.policy import SimpleActorCriticPolicy
        from rl import set_active_policy, clear_active_policy as _clear

        policy = SimpleActorCriticPolicy.load(rl_policy_path)
        set_active_policy(policy)
        cfg.USE_RL_POLICY = True
        clear_active_policy = _clear

    try:
        price_data = {}
        for symbol in symbols:
            df = load_prices(symbol)
            df = df.loc[(df.index >= start) & (df.index <= end)]
            price_data[symbol] = df

        if _parse_bool(args.enable_ta):
            ensure_agent_features(session, symbols, (idx for df in price_data.values() for idx in df.index))

        # Baseline (TA disabled)
        _ensure_ta_state(False)
        baseline_result = run_backtest(symbols, price_data, use_ta=False, session=session)

        ta_result = None
        if _parse_bool(args.enable_ta):
            regime_ta.reset_regime_state()
            _ensure_ta_state(True)
            ta_result = run_backtest(symbols, price_data, use_ta=True, session=session)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / f"ablation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        def serialize_trades(trades: List[BacktestTrade]) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "symbol": t.symbol,
                        "entry_dt": t.entry_dt.isoformat(),
                        "exit_dt": t.exit_dt.isoformat(),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.pnl,
                        "return": t.return_pct,
                        "kind": t.kind,
                    }
                    for t in trades
                ]
            )

        baseline_df = serialize_trades(baseline_result.trades)
        baseline_df.to_csv(output_dir / "baseline.csv", index=False)

        summary = {"baseline": baseline_result.metrics}

        baseline_plot = _build_plot(
            baseline_result.equity_curve,
            baseline_result.drawdown_curve,
            baseline_result.equity_index,
            "Baseline Equity & Drawdown",
        )
        baseline_plot.write_html(output_dir / "equity_baseline.html")

        if ta_result is not None:
            ta_df = serialize_trades(ta_result.trades)
            ta_df.to_csv(output_dir / "ta.csv", index=False)

            ta_plot = _build_plot(
                ta_result.equity_curve,
                ta_result.drawdown_curve,
                ta_result.equity_index,
                "TA Equity & Drawdown",
            )
            ta_plot.write_html(output_dir / "equity_ta.html")

            summary["ta"] = ta_result.metrics
            if baseline_result.returns and ta_result.returns:
                stat, pvalue = stats.ttest_ind(ta_result.returns, baseline_result.returns, equal_var=False)
                summary["welch_t"] = {"statistic": float(stat), "pvalue": float(pvalue)}
                print(f"Welch t-test (TA vs baseline): stat={stat:.4f}, p-value={pvalue:.4f}")
            else:
                summary["welch_t"] = {"statistic": None, "pvalue": None}
                print("Welch t-test unavailable (insufficient trades).")

            metrics_table = pd.DataFrame(
                [
                    {"Scenario": "Baseline", **baseline_result.metrics},
                    {"Scenario": "TA", **ta_result.metrics},
                ]
            ).set_index("Scenario")
        else:
            metrics_table = pd.DataFrame(
                [{"Scenario": "Baseline", **baseline_result.metrics}]
            ).set_index("Scenario")

        metrics_table.to_csv(output_dir / "summary.csv")
        print(metrics_table.to_string(float_format=lambda v: f"{v:0.4f}"))

        with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    finally:
        if clear_active_policy is not None:
            clear_active_policy()
            cfg.USE_RL_POLICY = False
        session.close()


if __name__ == "__main__":
    main()
