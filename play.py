# play.py
# Emilio José Solano Orozco
# Carné 21212

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol
import argparse
from common.envs import make_env_galaxian

import numpy as np
import gymnasium as gym
import ale_py
import torch
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from agents.dqn_agent import DQNAgent

gym.register_envs(ale_py)

STUDENT_EMAIL = "sol21212@uvg.edu.gt"
ENV_ID = "ALE/Galaxian-v5"
VIDEO_DIR = Path("videos")
DEFAULT_DQN_MODEL = Path("models/dqn/best_dqn.pt")


class Agent(Protocol):
    """Interface for any agent with an act method."""

    def act(self, obs: np.ndarray, info: dict, action_space: gym.Space) -> int: ...


def record_episode(policy: Callable[[np.ndarray, dict, gym.Space], int]) -> Path:
    """Run one episode, record .mp4, return path."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env_galaxian(seed=42, render_mode="rgb_array")
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
    """Random stateless policy."""
    return int(action_space.sample())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Galaxian with different policies.")
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "dqn"],
        help="Which policy to use for the episode.",
    )
    parser.add_argument(
        "--dqn-model-path",
        type=str,
        default=str(DEFAULT_DQN_MODEL),
        help="Path to the trained DQN model (.pt file).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for DQN: 'cuda' or 'cpu'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tmp = make_env_galaxian(render_mode="rgb_array")
    try:
        try:
            print("Action meanings:", tmp.unwrapped.get_action_meanings())
        except Exception:
            print("Action meanings not available.")
        obs_space = tmp.observation_space
        action_space = tmp.action_space
    finally:
        tmp.close()

    policy_fn: Callable[[np.ndarray, dict, gym.Space], int]

    if args.mode == "random":
        policy_fn = random_policy

    elif args.mode == "dqn":
        model_path = Path(args.dqn_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"DQN model not found at {model_path}. Train with train_dqn.py first."
            )

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = "cpu"

        # Epsilon is set to zero for purely greedy evaluation
        agent = DQNAgent(
            obs_space=obs_space,
            action_space=action_space,
            device=device,
            eps_start=0.0,
            eps_end=0.0,
        )
        agent.load(str(model_path))
        print(f"[OK] Loaded DQN model from {model_path} on device={device}")

        def dqn_policy(obs: np.ndarray, info: dict, env_action_space: gym.Space) -> int:
            # env_action_space is ignored, we use the action_space from initialization
            return agent.act(obs, info, action_space)

        policy_fn = dqn_policy

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    record_episode(policy_fn)


if __name__ == "__main__":
    main()
