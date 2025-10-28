"""FastAPI router exposing TradingAgents feature ingestion endpoints."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import List, Sequence

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from agents.ta_client import fetch_batch, get_features
from app.db import get_session
from app.models import AgentFeature, upsert_agent_features


router = APIRouter(prefix="/agents", tags=["agents"])


def _serialize_agent_feature(row: AgentFeature) -> dict:
    risk_flags = []
    if row.risk_flags:
        try:
            risk_flags = json.loads(row.risk_flags)
        except Exception:  # pragma: no cover - fallback for bad data
            risk_flags = []
    return {
        "id": row.id,
        "symbol": row.symbol,
        "asof": row.asof,
        "bull_score": row.bull_score,
        "bear_score": row.bear_score,
        "risk_flags": risk_flags,
        "thesis_hash": row.thesis_hash,
    }


@router.post("/fetch")
def fetch_agent_feature(
    symbol: str,
    asof: date | None = None,
    db: Session = Depends(get_session),
) -> dict:
    """Fetch features for a single symbol and persist them."""
    asof = asof or date.today()
    payload = get_features(symbol, asof)
    payload["asof"] = datetime.fromisoformat(payload["asof"]).date()
    rows = upsert_agent_features(db, [payload])
    db.commit()
    return _serialize_agent_feature(rows[0])


@router.post("/fetch_batch")
def fetch_agent_feature_batch(
    symbols: Sequence[str] = Body(..., embed=True),
    asof: date | None = None,
    db: Session = Depends(get_session),
) -> List[dict]:
    """Fetch features for multiple symbols and persist them."""
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols array is required")
    asof = asof or date.today()
    payloads = fetch_batch(list(symbols), asof)
    for payload in payloads:
        payload["asof"] = datetime.fromisoformat(payload["asof"]).date()
    rows = upsert_agent_features(db, payloads)
    db.commit()
    return [_serialize_agent_feature(row) for row in rows]


@router.get("/latest")
def latest_agent_feature(
    symbol: str,
    db: Session = Depends(get_session),
) -> dict:
    row = (
        db.query(AgentFeature)
        .filter(AgentFeature.symbol == symbol.upper())
        .order_by(AgentFeature.asof.desc())
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No agent features for symbol")
    return _serialize_agent_feature(row)
