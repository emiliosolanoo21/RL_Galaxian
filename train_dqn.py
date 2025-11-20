# train_dqn.py
from __future__ import annotations
from collections import deque
from pathlib import Path
import argparse
import math

import numpy as np
import torch

from common.envs import make_env_galaxian
from agents.dqn_agent import DQNAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent on ALE/Galaxian-v5")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="Total environment steps for training.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DQN updates.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--buffer-capacity", type=int, default=100_000,
                        help="Replay buffer capacity.")
    parser.add_argument("--target-update-freq", type=int, default=10_000,
                        help="Target network update frequency (in steps).")
    parser.add_argument("--eps-start", type=float, default=1.0,
                        help="Initial epsilon for epsilon-greedy.")
    parser.add_argument("--eps-end", type=float, default=0.05,
                        help="Final epsilon for epsilon-greedy.")
    parser.add_argument("--eps-decay", type=int, default=200_000,
                        help="Decay rate for epsilon schedule.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Episodes between logging.")
    parser.add_argument("--eval-window", type=int, default=20,
                        help="Episodes used to compute moving average reward.")
    parser.add_argument("--save-dir", type=str, default="models/dqn",
                        help="Directory to save best DQN model.")
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

    env = make_env_galaxian(seed=args.seed)
    obs_space = env.observation_space
    action_space = env.action_space

    agent = DQNAgent(
        obs_space=obs_space,
        action_space=action_space,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_update_freq=args.target_update_freq,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_dqn.pt"

    total_steps = 0
    episode_idx = 0
    best_mean_reward = -math.inf
    recent_rewards = deque(maxlen=args.eval_window)

    print("Starting DQN training...")
    while total_steps < args.total_steps:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done and total_steps < args.total_steps:
            action = agent.act(obs, info, action_space)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            train_stats = agent.train_step()

            obs = next_obs
            episode_reward += float(reward)
            total_steps += 1

        episode_idx += 1
        recent_rewards.append(episode_reward)

        if episode_idx % args.log-interval == 0:
            mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            eps = train_stats.get("epsilon", 0.0) if train_stats else 0.0
            print(
                f"[Episode {episode_idx:5d}] "
                f"steps={total_steps:7d} "
                f"reward={episode_reward:7.2f} "
                f"mean_reward={mean_reward:7.2f} "
                f"eps={eps:.3f}"
            )

            if len(recent_rewards) == args.eval_window and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                agent.save(str(best_model_path))
                print(f"  -> New best mean reward {best_mean_reward:.2f}, saved to {best_model_path}")

    env.close()
    print("Training finished.")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    if best_model_path.exists():
        print(f"Best model stored at: {best_model_path}")


if __name__ == "__main__":
    main()
