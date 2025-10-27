from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_session
from app.models import Metric

router = APIRouter()

@router.get("/")
def list_metrics(db: Session = Depends(get_session)):
    rows = db.query(Metric).order_by(Metric.ts.desc()).limit(200).all()
    return [{"ts": r.ts, "name": r.name, "value": r.value} for r in rows]
