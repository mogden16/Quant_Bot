"""Helpers for blending TradingAgents features into price-derived datasets."""
from __future__ import annotations

import json
from datetime import date
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy.orm import Session

from app.models import AgentFeature
from core.config import ENABLE_TA_FEATURES


DEFAULT_FLAGS: Sequence[str] = (
    "earnings_0d",
    "earnings_1d",
    "earnings_3d",
    "guidance_watch",
    "macro_vol",
)


def _ensure_columns(df: pd.DataFrame, flags: Iterable[str]) -> pd.DataFrame:
    df_out = df.copy()
    if "ta_bull_minus_bear" not in df_out:
        df_out["ta_bull_minus_bear"] = 0.0
    for flag in flags:
        col = f"ta_risk_{flag}"
        if col not in df_out:
            df_out[col] = False
    return df_out


def _parse_flags(row: AgentFeature | None) -> set[str]:
    if row is None or not row.risk_flags:
        return set()
    try:
        parsed = json.loads(row.risk_flags)
    except json.JSONDecodeError:
        return set()
    return {str(flag) for flag in parsed}


def _resolve_asof(asof: pd.Timestamp | date) -> date:
    if isinstance(asof, pd.Timestamp):
        return asof.to_pydatetime().date()
    return asof


def merge_ta_features(
    df: pd.DataFrame,
    symbol: str,
    asof: pd.Timestamp | date,
    session: Session,
) -> pd.DataFrame:
    """Merge the latest TradingAgents features for a symbol onto the dataframe."""

    df_out = _ensure_columns(df, DEFAULT_FLAGS)

    if not ENABLE_TA_FEATURES:
        df_out["ta_bull_minus_bear"] = 0.0
        for flag in DEFAULT_FLAGS:
            df_out[f"ta_risk_{flag}"] = False
        return df_out

    query_asof = _resolve_asof(asof)
    row = (
        session.query(AgentFeature)
        .filter(AgentFeature.symbol == symbol.upper(), AgentFeature.asof <= query_asof)
        .order_by(AgentFeature.asof.desc())
        .first()
    )

    if row is None:
        df_out["ta_bull_minus_bear"] = 0.0
        for flag in DEFAULT_FLAGS:
            df_out[f"ta_risk_{flag}"] = False
        return df_out

    flags = _parse_flags(row)
    all_flags = set(DEFAULT_FLAGS).union(flags)
    for flag in all_flags:
        df_out[f"ta_risk_{flag}"] = flag in flags

    df_out["ta_bull_minus_bear"] = float(row.bull_score - row.bear_score)
    return df_out


__all__ = ["merge_ta_features"]
