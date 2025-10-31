"""Command line entry point for training reinforcement learning policies."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from app.db import SessionLocal, init_db
from core import regime_ta
from data.ingest import load_prices
from rl.environment import RLTradingEnv
from rl.policy import create_policy


def _parse_symbols(raw: str) -> Sequence[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _load_price_data(symbols: Sequence[str], start: str | None, end: str | None) -> Dict[str, pd.DataFrame]:
    price_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = load_prices(symbol)
        if start:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end:
            df = df.loc[df.index <= pd.Timestamp(end)]
        price_data[symbol] = df
    return price_data


def _evaluate(env: RLTradingEnv, policy, symbol: str | None = None) -> float:
    obs = env.reset(symbol=symbol)
    done = False
    total_reward = 0.0
    while not done:
        action = policy.select_action(obs, explore=False)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a reinforcement learning trading policy")
    parser.add_argument("--symbols", type=str, default="DEMO", help="Comma separated symbol list")
    parser.add_argument("--start", type=str, default=None, help="Optional start date")
    parser.add_argument("--end", type=str, default=None, help="Optional end date")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--use-ta", action="store_true", help="Enable TradingAgents features during training")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the simple policy")
    parser.add_argument(
        "--implementation",
        type=str,
        choices=("simple", "sb3"),
        default="simple",
        help="Policy backend to use",
    )
    parser.add_argument("--output-dir", type=str, default="reports/rl", help="Directory for checkpoints and metrics")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols supplied")

    init_db()
    session = SessionLocal()

    try:
        price_data = _load_price_data(symbols, args.start, args.end)
        env = RLTradingEnv(price_data, session=session, symbols=symbols, use_ta=args.use_ta, seed=args.seed)
        initial_obs = env.reset(symbol=symbols[0])
        obs_size = len(initial_obs)
        action_size = 3
        policy = create_policy(
            obs_size,
            action_size,
            implementation=args.implementation,
            lr=args.lr,
            gamma=args.gamma,
        )

        metrics = []
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            observations = []
            actions = []
            rewards = []
            total_reward = 0.0
            steps = 0
            while not done:
                action = policy.select_action(obs, explore=True)
                next_obs, reward, done, _ = env.step(action)
                observations.append(obs)
                actions.append(action)
                rewards.append(float(reward))
                total_reward += float(reward)
                obs = next_obs
                steps += 1
            try:
                update_result = policy.update_episode(observations, actions, rewards)
                loss_info = {
                    "loss": update_result.loss,
                    "actor_loss": update_result.actor_loss,
                    "critic_loss": update_result.critic_loss,
                }
            except NotImplementedError:
                loss_info = {"loss": None, "actor_loss": None, "critic_loss": None}
            metrics.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps,
                    **loss_info,
                }
            )

        eval_reward = _evaluate(env, policy, symbol=symbols[0])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {"episodes": metrics, "eval_reward": eval_reward, "symbols": list(symbols)}
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)

        if hasattr(policy, "save"):
            policy.save(output_dir / "policy.pt")

        print(f"Training complete. Evaluation reward: {eval_reward:.4f}")
    finally:
        session.close()
        if args.use_ta:
            regime_ta.reset_regime_state()


if __name__ == "__main__":
    train()

