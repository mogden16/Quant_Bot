import numpy as np

from rl.policy import SimpleActorCriticPolicy


def test_simple_actor_critic_save_and_load(tmp_path):
    obs_size = 8
    action_size = 3
    policy = SimpleActorCriticPolicy(obs_size, action_size, hidden_sizes=(16,), lr=1e-3)

    observations = [np.random.randn(obs_size).astype(np.float32) for _ in range(5)]
    actions = [i % action_size for i in range(5)]
    rewards = [0.1 * i for i in range(5)]

    result = policy.update_episode(observations, actions, rewards)
    assert result.loss is not None

    out_path = tmp_path / "policy.pt"
    policy.save(out_path)

    restored = SimpleActorCriticPolicy.load(out_path)
    obs = observations[0]
    action = restored.select_action(obs, explore=False)
    assert isinstance(action, int)
    assert 0 <= action < action_size
