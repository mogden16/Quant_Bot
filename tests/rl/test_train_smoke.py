import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rl import runtime


def _demo_price_frame() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=70, freq="D")
    base = np.linspace(50, 55, len(idx))
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + np.sin(np.linspace(0, 3.14, len(idx))),
            "volume": np.linspace(500_000, 700_000, len(idx)),
        },
        index=idx,
    )
    return df


def test_train_cli_smoke(monkeypatch, tmp_path):
    from rl import train as train_module

    monkeypatch.setattr("data.ingest.load_prices", lambda symbol: _demo_price_frame())
    output_dir = tmp_path / "rl_runs"

    argv = [
        "rl/train.py",
        "--symbols",
        "DEMO",
        "--episodes",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setenv("TA_MOCK", "1")
    monkeypatch.setattr(sys, "argv", argv)

    train_module.train()

    runs = list(Path(output_dir).glob("run_*/metrics.json"))
    assert runs, "Expected metrics output from training run"
    runtime.clear_active_policy()
