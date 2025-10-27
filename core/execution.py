from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Order:
    symbol: str
    side: str
    qty: float
    price: float
    ts: datetime

class PaperBroker:
    def __init__(self, slippage_bps: float = 5, fee_per_trade: float = 0.0):
        self.slippage_bps = slippage_bps
        self.fee_per_trade = fee_per_trade

    def _slip(self, price: float, side: str) -> float:
        slip = price * (self.slippage_bps / 10000.0)
        return price + slip if side == "buy" else price - slip

    def market_order(self, symbol: str, side: str, qty: float, price: float, ts: Optional[datetime] = None) -> Order:
        fill = self._slip(price, "buy" if side.lower() == "buy" else "sell")
        return Order(symbol=symbol, side=side, qty=qty, price=fill, ts=ts or datetime.utcnow())
