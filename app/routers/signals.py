from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_session
from app.models import Signal
from app.schemas import SignalOut

router = APIRouter()

@router.get("/", response_model=list[SignalOut])
def list_signals(db: Session = Depends(get_session)):
    return db.query(Signal).order_by(Signal.ts.desc()).limit(100).all()
