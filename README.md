# Quant Edge System (Day/Swing, Research-Based)
A modular Python scaffold for a research-grounded 1–5 day trading system that paper-trades via API.

## TradingAgents Integration
The system can ingest narrative/sentiment features from the [TradingAgents](https://github.com/TauricResearch/TradingAgents) service via a sidecar client. Features are cached locally, stored in the SQL database, and exposed to the signal engine behind a safety gate.

### Environment variables
Set these variables to control the integration:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ENABLE_TA_FEATURES` | Toggle TA feature usage in the signal engine | `True` (core config) |
| `TA_BASE_URL` | Base URL for the TradingAgents API | `https://api.tradingagents.local` |
| `TA_TIMEOUT` | HTTP timeout in seconds | `10` |
| `TA_RETRIES` | HTTP retries for failed requests | `3` |
| `TA_CACHE_DIR` | Directory for cached agent responses | `data/ta_cache/` |
| `TA_MOCK` | Deterministic mock mode for CI/tests | unset (`0`) |

### CLI client and caching
Agent responses are cached per-symbol/per-date under `data/ta_cache/`. Mock mode (`TA_MOCK=1`) generates deterministic pseudo-scores without network calls.

### FastAPI endpoints
* `POST /agents/fetch?symbol=XYZ` – fetch and persist a single symbol for today (or `asof` query date).
* `POST /agents/fetch_batch` with body `{ "symbols": ["AAPL", "MSFT"] }` – batch fetch and persist.
* `GET /agents/latest?symbol=XYZ` – retrieve the most recent stored agent features.

All endpoints are idempotent; repeated calls upsert existing rows.

## Backtesting & Ablation
Run an ablation that compares baseline signals with TA-augmented variants:

```bash
python backtest/ablation.py --symbols AAPL,MSFT --start 2023-01-01 --end 2023-12-31 --enable-ta true --mock-ta true
```

Outputs are saved under `reports/ablation_<timestamp>/` (CSV trades, equity plots, summary JSON/CSV) and a Welch t-test on per-trade returns is printed. You can also invoke the helper target:

```bash
make ablation
```

## Reinforcement Learning Mode
Optional reinforcement learning support lets you replace the deterministic momentum/reversal rules with a learnable policy while
keeping the existing backtest and reporting stack intact.

1. Install the optional dependency:

   ```bash
   pip install torch
   ```

   (Stable Baselines3 can be plugged in as an alternative policy backend if you already have a trained model.)

2. Train a policy using the dedicated CLI. This wraps the trading loop in a gym-style environment and writes checkpoints and
   metrics to `reports/rl/`:

   ```bash
   python rl/train.py --symbols AAPL --start 2023-01-01 --end 2023-06-30 --episodes 10 --use-ta
   ```

3. Evaluate the trained policy in the existing ablation workflow by loading the checkpoint and enabling RL mode:

   ```bash
   python backtest/ablation.py --symbols AAPL --start 2023-07-01 --end 2023-12-31 --use-rl true --rl-policy reports/rl/run_<timestamp>/policy.pt
   ```

During RL evaluation the ablation script toggles the RL policy in place of the heuristic signals while preserving TA ablation and
report generation.

## Testing
```bash
pytest          # or make test
```

Unit tests cover TA caching, database upserts, merge logic, gating behaviour, and ablation reporting.

## Setup
Install dependencies with `pip install -r requirements.txt`, then start the API with `uvicorn app.main:app --reload`.
