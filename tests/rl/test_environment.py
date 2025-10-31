import numpy as np
import pandas as pd

from rl.environment import RLTradingEnv


def _demo_price_frame() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    base = np.linspace(100, 120, len(idx))
    df = pd.DataFrame(
        {
            "open": base - 0.5,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.linspace(1_000_000, 1_200_000, len(idx)),
        },
        index=idx,
    )
    return df


def test_environment_reset_and_step_are_deterministic():
    price_data = {"DEMO": _demo_price_frame()}
    env = RLTradingEnv(price_data, session=None, symbols=["DEMO"], seed=7)

    obs_first = env.reset(symbol="DEMO")
    obs_second = env.reset(symbol="DEMO")
    np.testing.assert_allclose(obs_first, obs_second)

    step_obs, reward, done, info = env.step(RLTradingEnv.ACTION_OPEN_LONG)
    assert step_obs.shape == obs_first.shape
    assert np.isfinite(reward)
    assert "position" in info


def test_environment_rollout_until_done():
    price_data = {"DEMO": _demo_price_frame()}
    env = RLTradingEnv(price_data, session=None, symbols=["DEMO"], seed=1)
    obs = env.reset(symbol="DEMO")
    total_reward = 0.0
    done = False
    steps = 0
    while not done:
        action = RLTradingEnv.ACTION_HOLD if steps else RLTradingEnv.ACTION_OPEN_LONG
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    assert steps > 0
    assert obs.shape[-1] == len(env.feature_columns) + 4
    assert np.isfinite(total_reward)
