# common/envs.py
from __future__ import annotations
from typing import Optional

import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordEpisodeStatistics

ENV_ID = "ALE/Galaxian-v5"

gym.register_envs(ale_py)


def make_env_galaxian(
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    enable_stats: bool = True,
) -> gym.Env:
    """Create Galaxian environment with Atari preprocessing."""
    base_kwargs = {}
    if render_mode is not None:
        base_kwargs["render_mode"] = render_mode

    # frameskip=1, AtariPreprocessing hará el frame_skip
    env = gym.make(ENV_ID, obs_type="rgb", frameskip=1, **base_kwargs)

    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=True,  # -> (84, 84, 1)
        frame_skip=4,
        scale_obs=False,
    )

    if enable_stats:
        env = RecordEpisodeStatistics(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
