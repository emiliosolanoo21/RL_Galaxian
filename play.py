# play.py
# Emilio José Solano Orozco
# Carné 21212

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol
import argparse

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

gym.register_envs(ale_py)

STUDENT_EMAIL = "sol21212@uvg.edu.gt"
ENV_ID = "ALE/Galaxian-v5"
VIDEO_DIR = Path("videos")


class Agent(Protocol):
    """Interface for any policy-based agent."""

    def act(self, obs: np.ndarray, info: dict, action_space: gym.Space) -> int: ...


def record_episode(policy: Callable[[np.ndarray, dict, gym.Space], int]) -> Path:
    """Run one episode, record .mp4, return path."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda eid: True,
        name_prefix="tmp",
    )

    obs, info = env.reset(seed=42)
    total_reward = 0.0

    while True:
        action = int(policy(obs, info, env.action_space))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()

    ts = datetime.now().strftime("%Y%m%d%H%M")
    user = STUDENT_EMAIL.split("@")[0]
    score = int(total_reward)

    latest_mp4 = max(VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    final_path = VIDEO_DIR / f"{user}_{ts}_{score}.mp4"
    latest_mp4.replace(final_path)

    print(f"[OK] Saved video -> {final_path}")
    return final_path


class RandomAgent:
    """Stateless random agent."""

    def act(self, obs: np.ndarray, info: dict, action_space: gym.Space) -> int:
        return int(action_space.sample())


def random_policy(obs: np.ndarray, info: dict, action_space: gym.Space) -> int:
    """Adapter for RandomAgent so it matches the policy callable."""
    return int(action_space.sample())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random"],  # later: ["random", "dqn", "a2c", ...]
        help="Which policy to use for the episode.",
    )
    args = parser.parse_args()

    tmp = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        try:
            print("Action meanings:", tmp.unwrapped.get_action_meanings())
        except Exception:
            print("Action meanings not available.")
    finally:
        tmp.close()

    if args.mode == "random":
        policy_fn = random_policy
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    record_episode(policy_fn)


if __name__ == "__main__":
    main()
