"""Gym-style reinforcement learning environment for the trading loop."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator

from core.features import compute_features
from core.features_ta import merge_ta_features
from core.position_sizing import fractional_kelly_equity_fraction
from core.regime import RegimeGate
from core.risk import atr_stop_target
from core import regime_ta
from core.config import RiskConfig, SignalConfig, SizingConfig

INITIAL_EQUITY = 100_000.0


def _safe_float(value: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


@dataclass
class EnvironmentState:
    """Container tracking the current portfolio state."""

    position: Optional[dict]
    equity: float
    returns: List[float]


class RLTradingEnv:
    """Reinforcement learning interface around the deterministic backtest loop.

    The observation vector is arranged as:

    ``[feature_0, feature_1, ..., feature_n, position_flag, entry_rel_close, stop_rel_close, equity_rel]``

    Where the ``feature`` values correspond to the numeric columns produced by
    :func:`core.features.compute_features` (and optional TA augmentations) ordered
    alphabetically for determinism. ``position_flag`` is ``1.0`` when a long
    position is open, ``entry_rel_close`` and ``stop_rel_close`` are expressed as
    ratios of the current close price (0 if no position), and ``equity_rel`` is
    the current equity normalised by the initial equity. This layout is reused by
    the signal bridge so policies trained in this environment receive identical
    observations when driving live decisions.
    """

    ACTION_HOLD = 0
    ACTION_OPEN_LONG = 1
    ACTION_CLOSE_POSITION = 2

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        session,
        symbols: Optional[Sequence[str]] = None,
        use_ta: bool = False,
        risk_cfg: Optional[RiskConfig] = None,
        sizing_cfg: Optional[SizingConfig] = None,
        signal_cfg: Optional[SignalConfig] = None,
        warmup: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        self.price_data = price_data
        self.session = session
        self.symbols = list(symbols or price_data.keys())
        if not self.symbols:
            raise ValueError("At least one symbol is required for the environment")
        self.use_ta = use_ta
        self.risk_cfg = risk_cfg or RiskConfig()
        self.sizing_cfg = sizing_cfg or SizingConfig()
        self.signal_cfg = signal_cfg or SignalConfig()
        self.warmup = warmup
        self._rng: Generator = np.random.default_rng(seed)
        self._gate = RegimeGate()
        self._state: Optional[EnvironmentState] = None
        self._feature_frame: Optional[pd.DataFrame] = None
        self._symbol: Optional[str] = None
        self._index: int = 0
        self._feature_columns: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int]) -> None:
        """Reseed the environment's RNG."""

        self._rng = np.random.default_rng(seed)

    def reset(self, symbol: Optional[str] = None) -> np.ndarray:
        """Reset the environment and return the initial observation."""

        if symbol is None:
            symbol = self._rng.choice(self.symbols)
        if symbol not in self.price_data:
            raise KeyError(f"Symbol '{symbol}' not found in price data")

        df = self.price_data[symbol]
        feat = compute_features(df)
        self._feature_frame = feat
        self._symbol = symbol
        self._index = max(self.warmup, 1)
        self._state = EnvironmentState(position=None, equity=INITIAL_EQUITY, returns=[])
        self._gate = RegimeGate()
        self._feature_columns = None

        window = self._current_window()
        obs = self._build_observation(window)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Advance the environment by one bar using ``action``."""

        if self._state is None or self._feature_frame is None or self._symbol is None:
            raise RuntimeError("Environment must be reset before stepping")

        window = self._current_window()
        row = window.iloc[-1]
        price = float(row["close"])
        atr_value = float(row.get("atr_14", np.nan))

        if math.isnan(atr_value):
            atr_value = 0.0
        reward = 0.0
        info: Dict[str, float] = {"symbol": self._symbol, "equity": self._state.equity}

        # Update trailing stop / take profit when a position is open.
        if self._state.position is not None:
            pos = self._state.position
            high_mark = max(pos["high_mark"], price)
            pos["high_mark"] = high_mark
            if atr_value > 0:
                trail = high_mark - self.risk_cfg.trailing_atr_mult * atr_value
                pos["stop"] = max(pos["stop"], float(trail))

            exit_price = None
            exit_kind = None
            if price <= pos["stop"]:
                exit_price = pos["stop"]
                exit_kind = "stop"
            elif price >= pos["target"]:
                exit_price = pos["target"]
                exit_kind = "target"
            elif action == self.ACTION_CLOSE_POSITION:
                exit_price = price
                exit_kind = "manual"

            if exit_price is not None:
                pnl = (exit_price - pos["entry"]) * pos["shares"]
                trade_return = pnl / max(self._state.equity, 1e-9)
                self._state.equity += pnl
                self._state.returns.append(trade_return)
                reward += trade_return
                info.update({"last_trade_return": trade_return, "last_trade_kind": {"stop": 0, "target": 1, "manual": 2}[exit_kind]})
                if self.use_ta:
                    regime_ta.register_trade_return(trade_return, session=self.session)
                self._state.position = None

        # Potentially open a new position.
        if self._state.position is None and action == self.ACTION_OPEN_LONG:
            if self._gate.allow(self._state.returns):
                if atr_value <= 0:
                    pass
                else:
                    stop, target = atr_stop_target(price, atr_value, self.risk_cfg.stop_atr_mult, self.risk_cfg.target_rr)
                    per_share_risk = max(1e-6, price - stop)
                    assumed_p_win = 0.55
                    f_equity = fractional_kelly_equity_fraction(assumed_p_win, self.risk_cfg.target_rr, self.sizing_cfg)
                    f_equity = min(max(f_equity, 0.0), self.risk_cfg.risk_per_trade)
                    dollar_risk = self._state.equity * f_equity
                    shares = int(dollar_risk / per_share_risk)
                    if shares > 0:
                        self._state.position = {
                            "entry": price,
                            "shares": shares,
                            "stop": float(stop),
                            "target": float(target),
                            "high_mark": price,
                            "kind": "rl_long",
                            "entry_dt": window.index[-1],
                        }

        # Move forward in time.
        self._index += 1
        done = self._index >= len(self._feature_frame)
        if not done:
            next_window = self._current_window()
            obs = self._build_observation(next_window)
        else:
            obs = self._build_observation(window)

        info.update({"position": 1.0 if self._state.position else 0.0})
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_window(self) -> pd.DataFrame:
        if self._feature_frame is None:
            raise RuntimeError("Feature frame not initialised")
        if self._index >= len(self._feature_frame):
            return self._feature_frame.iloc[[-1]]
        window = self._feature_frame.iloc[: self._index + 1]
        if self.use_ta:
            window = merge_ta_features(window, self._symbol, window.index[-1], self.session)
        return window

    def _build_observation(self, window: pd.DataFrame) -> np.ndarray:
        row = window.iloc[-1]
        close_price = _safe_float(row.get("close", 0.0))
        if self._feature_columns is None:
            self._feature_columns = _extract_numeric_columns(window)
        features = [_safe_float(row.get(col, 0.0)) for col in self._feature_columns]
        position_flag = 1.0 if self._state and self._state.position else 0.0
        entry_rel = 0.0
        stop_rel = 0.0
        if position_flag and close_price != 0:
            entry_rel = (self._state.position["entry"] - close_price) / close_price
            stop_rel = (self._state.position["stop"] - close_price) / close_price
        equity_rel = (self._state.equity if self._state else INITIAL_EQUITY) / INITIAL_EQUITY
        obs = np.asarray(
            [
                *features,
                _safe_float(position_flag),
                _safe_float(entry_rel),
                _safe_float(stop_rel),
                _safe_float(equity_rel),
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def feature_columns(self) -> Sequence[str]:
        if self._feature_columns is None:
            raise RuntimeError("Environment must be reset before accessing feature columns")
        return self._feature_columns


def build_observation_from_window(window: pd.DataFrame, feature_columns: Optional[Sequence[str]] = None) -> np.ndarray:
    """Utility used by the signal bridge to recreate environment observations."""

    cols = list(feature_columns or _extract_numeric_columns(window))
    row = window.iloc[-1]
    features = [_safe_float(row.get(col, 0.0)) for col in cols]
    obs = np.asarray([*features, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return obs


def _extract_numeric_columns(frame: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            cols.append(col)
    return sorted(cols)


__all__ = [
    "RLTradingEnv",
    "EnvironmentState",
    "build_observation_from_window",
]

