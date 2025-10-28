from __future__ import annotations

import json
from datetime import date, datetime
from typing import Iterable, List

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Session
from app.db import Base

class Price(Base):
    __tablename__ = "prices"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    ts = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'ts', name='uq_symbol_ts'),)

class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    kind = Column(String)
    strength = Column(Float)
    p_win = Column(Float)
    payoff = Column(Float)
    active = Column(Boolean, default=True)

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    symbol = Column(String, index=True)
    entry_ts = Column(DateTime, index=True)
    entry_price = Column(Float)
    size = Column(Float)
    stop = Column(Float)
    target = Column(Float)
    exit_ts = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    status = Column(String, default="open")
    pnl = Column(Float, default=0.0)

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    name = Column(String, index=True)
    value = Column(Float)


class AgentFeature(Base):
    __tablename__ = "agent_features"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True, nullable=False)
    asof = Column(Date, index=True, nullable=False)
    bull_score = Column(Float, nullable=False)
    bear_score = Column(Float, nullable=False)
    risk_flags = Column(Text, nullable=False)
    thesis_hash = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "asof", name="uq_agent_features_symbol_asof"),
    )


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported asof type: {type(value)!r}")


def upsert_agent_features(session: Session, rows: Iterable[dict]) -> List[AgentFeature]:
    """Insert or update agent features while keeping symbol/asof unique."""

    persisted: List[AgentFeature] = []
    for row in rows:
        symbol = row["symbol"].upper()
        asof = _coerce_date(row["asof"])
        bull_score = float(row["bull_score"])
        bear_score = float(row["bear_score"])
        risk_flags = row.get("risk_flags", [])
        thesis_hash = str(row.get("thesis_hash", ""))

        risk_flags_text = json.dumps(sorted(risk_flags), separators=(",", ":"))
        instance = (
            session.query(AgentFeature)
            .filter(AgentFeature.symbol == symbol, AgentFeature.asof == asof)
            .one_or_none()
        )
        if instance is None:
            instance = AgentFeature(
                symbol=symbol,
                asof=asof,
                bull_score=bull_score,
                bear_score=bear_score,
                risk_flags=risk_flags_text,
                thesis_hash=thesis_hash,
            )
            session.add(instance)
        else:
            instance.bull_score = bull_score
            instance.bear_score = bear_score
            instance.risk_flags = risk_flags_text
            instance.thesis_hash = thesis_hash
        persisted.append(instance)

    session.flush()
    return persisted
