from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.db import get_session
from app.models import Trade
from app.schemas import TradeIn, TradeOut

router = APIRouter()

@router.post("/", response_model=TradeOut)
def open_trade(payload: TradeIn, db: Session = Depends(get_session)):
    t = Trade(
        signal_id=payload.signal_id,
        symbol=payload.symbol,
        entry_ts=datetime.utcnow(),
        entry_price=payload.entry_price,
        size=payload.size,
        stop=payload.stop,
        target=payload.target,
        status="open",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

@router.get("/", response_model=list[TradeOut])
def list_trades(db: Session = Depends(get_session)):
    return db.query(Trade).order_by(Trade.entry_ts.desc()).limit(200).all()
