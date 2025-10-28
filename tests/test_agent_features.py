from __future__ import annotations

from datetime import date

from app.models import AgentFeature, upsert_agent_features


def test_upsert_agent_features_insert_and_update(db_session):
    first_payload = {
        "symbol": "AAPL",
        "asof": date(2024, 1, 1),
        "bull_score": 0.6,
        "bear_score": 0.2,
        "risk_flags": ["earnings_3d", "macro_vol"],
        "thesis_hash": "abc123",
    }
    rows = upsert_agent_features(db_session, [first_payload])
    assert len(rows) == 1
    db_session.commit()

    stored = (
        db_session.query(AgentFeature)
        .filter_by(symbol="AAPL", asof=date(2024, 1, 1))
        .one()
    )
    assert stored.bull_score == 0.6
    assert stored.bear_score == 0.2

    updated_payload = {
        "symbol": "AAPL",
        "asof": date(2024, 1, 1),
        "bull_score": 0.55,
        "bear_score": 0.25,
        "risk_flags": ["earnings_1d"],
        "thesis_hash": "zzz",
    }
    rows = upsert_agent_features(db_session, [updated_payload])
    db_session.commit()

    updated = (
        db_session.query(AgentFeature)
        .filter_by(symbol="AAPL", asof=date(2024, 1, 1))
        .one()
    )
    assert updated.bull_score == 0.55
    assert updated.bear_score == 0.25
    assert "earnings_1d" in updated.risk_flags
    assert updated.thesis_hash == "zzz"

    # ensure no duplicates were created
    count = (
        db_session.query(AgentFeature)
        .filter(AgentFeature.symbol == "AAPL")
        .count()
    )
    assert count == 1
