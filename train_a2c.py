# train_a2c.py
from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import math

import numpy as np
import torch

from common.envs import make_env_galaxian
from agents.a2c_agent import A2CAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A2C agent on ALE/Galaxian-v5")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="Total environment steps for training.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="Learning rate.")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Value loss coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy regularization coefficient.")
    parser.add_argument("--rollout-length", type=int, default=5,
                        help="Number of steps per rollout update.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between logging.")
    parser.add_argument("--eval-window", type=int, default=20,
                        help="Episodes used to compute moving average reward.")
    parser.add_argument("--save-dir", type=str, default="models/a2c",
                        help="Directory to save best A2C model.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    env = make_env_galaxian(seed=args.seed, enable_stats=True)
    obs_space = env.observation_space
    action_space = env.action_space

    agent = A2CAgent(
        obs_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_a2c.pt"

    total_steps = 0
    episode_idx = 0
    best_mean_reward = None
    recent_rewards = deque(maxlen=args.eval_window)

    print("Starting A2C training...")
    while total_steps < args.total_steps:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []

        train_stats = {}

        while not done and total_steps < args.total_steps:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_rewards.append(float(reward))
            rollout_dones.append(done)

            obs = next_obs
            episode_reward += float(reward)
            total_steps += 1

            if (
                len(rollout_obs) >= args.rollout_length
                or done
                or total_steps >= args.total_steps
            ):
                last_obs = obs
                last_done = done
                train_stats = agent.update_from_rollout(
                    rollout_obs,
                    rollout_actions,
                    rollout_rewards,
                    rollout_dones,
                    last_obs,
                    last_done,
                )

                rollout_obs = []
                rollout_actions = []
                rollout_rewards = []
                rollout_dones = []

        episode_idx += 1
        recent_rewards.append(episode_reward)

        if episode_idx % args.log_interval == 0:
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            print(
                f"[Episode {episode_idx:5d}] "
                f"steps={total_steps:7d} "
                f"reward={episode_reward:7.2f} "
                f"mean_reward={mean_reward:7.2f}"
            )

            if recent_rewards:
                window_size = min(len(recent_rewards), args.eval_window)
                window_rewards = list(recent_rewards)[-window_size:]
                window_mean = float(np.mean(window_rewards))

                if best_mean_reward is None or window_mean > best_mean_reward:
                    best_mean_reward = window_mean
                    agent.save(str(best_model_path))
                    print(
                        f"  -> New best mean reward {best_mean_reward:.2f}, "
                        f"saved to {best_model_path}"
                    )

    env.close()
    print("Training finished.")
    if best_mean_reward is None:
        print("Best mean reward: N/A (no episodes recorded).")
    else:
        print(f"Best mean reward: {best_mean_reward:.2f}")
    if best_model_path.exists():
        print(f"Best model stored at: {best_model_path}")
    else:
        print("No model was saved. Try increasing total steps or episodes.")


if __name__ == "__main__":
    main()
