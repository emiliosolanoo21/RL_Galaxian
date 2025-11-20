# common/envs.py
from __future__ import annotations
from typing import Optional

import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordEpisodeStatistics

ENV_ID = "ALE/Galaxian-v5"

gym.register_envs(ale_py)


def make_env_galaxian(seed: Optional[int] = None) -> gym.Env:
    """Create Galaxian environment for training."""
    env = gym.make(ENV_ID, obs_type="rgb")  # default ALE RGB observations
    env = RecordEpisodeStatistics(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
