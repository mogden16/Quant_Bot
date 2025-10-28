from __future__ import annotations

import pandas as pd
import pytest

from backtest import ablation
from data.ingest import demo_price_series


@pytest.mark.usefixtures("temp_cache_dir")
def test_ablation_run_produces_metrics(monkeypatch, db_session, temp_cache_dir):
    monkeypatch.setenv("TA_MOCK", "1")
    monkeypatch.setenv("TA_CACHE_DIR", str(temp_cache_dir))
    symbols = ["MOCK"]
    df = demo_price_series(n=80)
    price_data = {"MOCK": df}

    ablation.ensure_agent_features(db_session, symbols, df.index)
    regime_ta = ablation.regime_ta
    regime_ta.reset_regime_state()
    ablation._ensure_ta_state(True)

    result = ablation.run_backtest(symbols, price_data, use_ta=True, session=db_session)
    assert isinstance(result.metrics, dict)
    assert result.metrics["Trades"] >= 0
    assert len(result.equity_curve) == len(result.returns) + 1
