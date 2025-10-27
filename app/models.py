from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, UniqueConstraint
from app.db import Base
from datetime import datetime

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
