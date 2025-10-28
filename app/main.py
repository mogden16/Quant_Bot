from fastapi import FastAPI
from app.db import init_db
from app.routers import agents, metrics, signals, trades

app = FastAPI(title="Quant Edge System API", version="0.1.0")

@app.on_event("startup")
def startup_event():
    init_db()

app.include_router(signals.router, prefix="/signals", tags=["signals"])
app.include_router(trades.router, prefix="/trades", tags=["trades"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
app.include_router(agents.router)

@app.get("/health")
def health():
    return {"status": "ok"}
