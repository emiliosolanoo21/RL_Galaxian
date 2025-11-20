# common/envs.py
from __future__ import annotations
from typing import Optional

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics

ENV_ID = "ALE/Galaxian-v5"

gym.register_envs(ale_py)


def make_env_galaxian(seed: Optional[int] = None, render_mode: Optional[str] = None) -> gym.Env:
    """Create preprocessed Galaxian environment (84x84 grayscale)."""
    # Disable built-in frameskip, AtariPreprocessing will handle it
    base_kwargs = {}
    if render_mode is not None:
        base_kwargs["render_mode"] = render_mode

    env = gym.make(ENV_ID, obs_type="rgb", frameskip=1, **base_kwargs)

    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=True,  # shape: (84, 84, 1)
        frame_skip=4,
        scale_obs=False,
    )
    env = RecordEpisodeStatistics(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
