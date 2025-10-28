"""Client for fetching agent-driven features from TradingAgents service."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import requests


DEFAULT_BASE_URL = "https://api.tradingagents.local"
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 3
CACHE_DIR_ENV = "TA_CACHE_DIR"
MOCK_ENV = "TA_MOCK"
BASE_URL_ENV = "TA_BASE_URL"
TIMEOUT_ENV = "TA_TIMEOUT"
RETRIES_ENV = "TA_RETRIES"


@dataclass(frozen=True)
class AgentFeaturePayload:
    """Normalized representation of agent features returned by TradingAgents."""

    symbol: str
    asof: date
    bull_score: float
    bear_score: float
    risk_flags: List[str]
    thesis_hash: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "risk_flags": self.risk_flags,
            "thesis_hash": self.thesis_hash,
        }


def _get_cache_dir() -> Path:
    base = os.getenv(CACHE_DIR_ENV, str(Path("data") / "ta_cache"))
    cache_dir = Path(base)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(symbol: str, asof: date) -> Path:
    sanitized_symbol = symbol.replace("/", "-")
    return _get_cache_dir() / f"{sanitized_symbol}_{asof.isoformat()}.json"


def _load_cache(symbol: str, asof: date) -> dict | None:
    cache_file = _cache_path(symbol, asof)
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return None


def _write_cache(payload: AgentFeaturePayload) -> None:
    cache_file = _cache_path(payload.symbol, payload.asof)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(payload.to_dict(), f, sort_keys=True)


def _deterministic_scores(symbol: str, asof: date) -> AgentFeaturePayload:
    seed_material = f"{symbol}|{asof.isoformat()}".encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    # map bytes into deterministic floats in [0,1]
    bull_raw = int.from_bytes(digest[:8], "big") / 2**64
    bear_raw = int.from_bytes(digest[8:16], "big") / 2**64
    bull_score = round(min(max(bull_raw, 0.0), 1.0), 4)
    bear_score = round(min(max(bear_raw, 0.0), 1.0), 4)
    # deterministic risk flags
    flags: List[str] = []
    risk_markers = [
        "earnings_0d",
        "earnings_1d",
        "earnings_3d",
        "guidance_watch",
        "macro_vol",
    ]
    threshold_bytes = digest[16:16 + len(risk_markers)]
    for marker, b in zip(risk_markers, threshold_bytes):
        if b % 3 == 0:
            flags.append(marker)
    thesis_hash = hashlib.sha256(seed_material + b"thesis").hexdigest()
    return AgentFeaturePayload(
        symbol=symbol,
        asof=asof,
        bull_score=bull_score,
        bear_score=bear_score,
        risk_flags=flags,
        thesis_hash=thesis_hash,
    )


def _request_payload(symbols: Iterable[str], asof: date) -> List[AgentFeaturePayload]:
    base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")
    timeout = int(os.getenv(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    retries = int(os.getenv(RETRIES_ENV, DEFAULT_RETRIES))

    url = f"{base_url}/features"
    payload = {"symbols": list(symbols), "asof": asof.isoformat()}

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            results = []
            for row in data:
                results.append(
                    AgentFeaturePayload(
                        symbol=row["symbol"],
                        asof=datetime.fromisoformat(row["asof"]).date(),
                        bull_score=float(row.get("bull_score", 0.0)),
                        bear_score=float(row.get("bear_score", 0.0)),
                        risk_flags=list(row.get("risk_flags", [])),
                        thesis_hash=str(row.get("thesis_hash", "")),
                    )
                )
            return results
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 5))
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def get_features(symbol: str, asof: date) -> dict:
    """Fetch features for a single symbol, respecting caching and mock mode."""
    cached = _load_cache(symbol, asof)
    if cached is not None:
        return cached

    if os.getenv(MOCK_ENV, "0") == "1":
        payload = _deterministic_scores(symbol, asof)
        _write_cache(payload)
        return payload.to_dict()

    payloads = _request_payload([symbol], asof)
    if not payloads:
        raise ValueError(f"TradingAgents returned no data for {symbol} @ {asof}")
    payload = payloads[0]
    _write_cache(payload)
    return payload.to_dict()


def fetch_batch(symbols: Iterable[str], asof: date) -> List[dict]:
    """Fetch features for multiple symbols with caching and mock support."""
    results: List[dict] = []
    missing_symbols: List[str] = []

    for symbol in symbols:
        cached = _load_cache(symbol, asof)
        if cached is not None:
            results.append(cached)
        else:
            missing_symbols.append(symbol)

    if missing_symbols:
        if os.getenv(MOCK_ENV, "0") == "1":
            payloads = [_deterministic_scores(sym, asof) for sym in missing_symbols]
        else:
            payloads = _request_payload(missing_symbols, asof)
        for payload in payloads:
            _write_cache(payload)
            results.append(payload.to_dict())

    # Ensure stable ordering corresponding to requested symbols
    order = {symbol: idx for idx, symbol in enumerate(symbols)}
    results.sort(key=lambda item: order.get(item["symbol"], 0))
    return results


__all__ = [
    "get_features",
    "fetch_batch",
]
