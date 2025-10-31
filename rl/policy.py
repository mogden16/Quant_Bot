"""Policy abstractions for reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn, optim
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover - handled via requirements
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    Categorical = Any  # type: ignore[assignment]


@dataclass
class PolicyUpdateResult:
    loss: float
    actor_loss: float
    critic_loss: float


class SimpleActorCriticPolicy:
    """Minimal actor-critic policy implemented with PyTorch."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (64, 64),
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for the SimpleActorCriticPolicy")

        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.hidden_sizes = tuple(hidden_sizes)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        layers: List[nn.Module] = []
        input_dim = obs_size
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.Tanh())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, action_size))
        self.actor = nn.Sequential(*layers).to(self.device)

        critic_layers: List[nn.Module] = []
        input_dim = obs_size
        for hidden in self.hidden_sizes:
            critic_layers.append(nn.Linear(input_dim, hidden))
            critic_layers.append(nn.Tanh())
            input_dim = hidden
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits: Tensor = self.actor(obs_tensor)
        logits = torch.nan_to_num(logits)
        dist = Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(dist.probs, dim=-1)
        return int(action.item())

    def update_episode(
        self,
        observations: Sequence[np.ndarray],
        actions: Sequence[int],
        rewards: Sequence[float],
    ) -> PolicyUpdateResult:
        if not observations:
            raise ValueError("Cannot update policy with empty trajectory")
        obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        obs_tensor = torch.nan_to_num(obs_tensor)
        act_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        returns = self._discount_returns(rewards)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        logits: Tensor = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        log_probs: Tensor = dist.log_prob(act_tensor)

        values: Tensor = self.critic(obs_tensor).squeeze(-1)
        advantages = ret_tensor - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = 0.5 * (ret_tensor - values).pow(2).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return PolicyUpdateResult(float(loss.item()), float(actor_loss.item()), float(critic_loss.item()))

    def _discount_returns(self, rewards: Sequence[float]) -> List[float]:
        discounted: List[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            discounted.insert(0, running)
        if discounted:
            mean = float(np.mean(discounted))
            std = float(np.std(discounted))
            if std > 1e-8:
                discounted = [(r - mean) / std for r in discounted]
        return discounted

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "config": {
                "obs_size": self.obs_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "device": str(self.device),
                "hidden_sizes": list(self.hidden_sizes),
            },
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SimpleActorCriticPolicy":
        checkpoint = torch.load(Path(path), map_location="cpu")
        config = checkpoint["config"]
        policy = cls(
            config["obs_size"],
            config["action_size"],
            hidden_sizes=tuple(config.get("hidden_sizes", (64, 64))),
            gamma=config.get("gamma", 0.99),
            device=config.get("device"),
        )
        policy.actor.load_state_dict(checkpoint["actor"])
        policy.critic.load_state_dict(checkpoint["critic"])
        policy.actor.to(policy.device)
        policy.critic.to(policy.device)
        return policy


class StableBaselinesPolicyWrapper:
    """Adapter that forwards to a Stable Baselines model if supplied."""

    def __init__(self, model) -> None:
        self.model = model

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        action, _ = self.model.predict(observation, deterministic=not explore)
        if isinstance(action, (list, tuple, np.ndarray)):
            return int(np.asarray(action).item())
        return int(action)

    def update_episode(self, *args: Any, **kwargs: Any) -> PolicyUpdateResult:
        raise NotImplementedError("Stable Baselines policies must be trained externally")

    def save(self, path: str | Path) -> None:
        self.model.save(path)


def create_policy(
    obs_size: int,
    action_size: int,
    implementation: str = "simple",
    **kwargs: Any,
):
    """Factory returning a policy implementation."""

    if implementation == "simple":
        return SimpleActorCriticPolicy(obs_size, action_size, **kwargs)
    if implementation == "sb3":
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Stable Baselines policy requires a pre-trained model via the 'model' kwarg")
        return StableBaselinesPolicyWrapper(model)
    raise ValueError(f"Unknown policy implementation '{implementation}'")


__all__ = [
    "SimpleActorCriticPolicy",
    "StableBaselinesPolicyWrapper",
    "create_policy",
    "PolicyUpdateResult",
]

