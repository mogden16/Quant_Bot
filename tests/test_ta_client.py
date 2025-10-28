from __future__ import annotations

import json
from datetime import date

import pytest

from agents import ta_client


@pytest.mark.usefixtures("temp_cache_dir")
def test_get_features_uses_cache(monkeypatch, temp_cache_dir):
    monkeypatch.setenv("TA_MOCK", "1")
    monkeypatch.setenv("TA_CACHE_DIR", str(temp_cache_dir))
    asof = date(2024, 1, 5)

    first = ta_client.get_features("AAPL", asof)
    cache_file = temp_cache_dir / "AAPL_2024-01-05.json"
    assert cache_file.exists()

    with cache_file.open() as fh:
        cached_json = json.load(fh)
    assert cached_json == first

    # disable mock to ensure we fall back to cache instead of remote call
    monkeypatch.setenv("TA_MOCK", "0")

    call_count = 0

    def fake_request(symbols, asof):  # pragma: no cover - should not run
        nonlocal call_count
        call_count += 1
        return []

    monkeypatch.setattr(ta_client, "_request_payload", fake_request)

    second = ta_client.get_features("AAPL", asof)
    assert second == first
    assert call_count == 0


def test_fetch_batch_only_requests_missing(monkeypatch, temp_cache_dir):
    monkeypatch.setenv("TA_CACHE_DIR", str(temp_cache_dir))
    monkeypatch.setenv("TA_MOCK", "1")
    asof = date(2024, 2, 1)
    ta_client.get_features("AAPL", asof)

    monkeypatch.setenv("TA_MOCK", "0")
    requested = []

    def fake_request(symbols, request_asof):
        requested.append(list(symbols))
        return [ta_client._deterministic_scores(sym, request_asof) for sym in symbols]

    monkeypatch.setattr(ta_client, "_request_payload", fake_request)

    results = ta_client.fetch_batch(["AAPL", "MSFT", "GOOG"], asof)
    assert len(results) == 3
    assert requested == [["MSFT", "GOOG"]]
