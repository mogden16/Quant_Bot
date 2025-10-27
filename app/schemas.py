from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SignalOut(BaseModel):
    id: int
    ts: datetime
    symbol: str
    side: str
    kind: str
    strength: float
    p_win: float
    payoff: float
    active: bool
    class Config:
        from_attributes = True

class TradeIn(BaseModel):
    signal_id: Optional[int] = None
    symbol: str
    entry_price: float
    size: float
    stop: float
    target: float

class TradeOut(BaseModel):
    id: int
    symbol: str
    entry_ts: datetime
    entry_price: float
    size: float
    stop: float
    target: float
    status: str
    pnl: float
    class Config:
        from_attributes = True
