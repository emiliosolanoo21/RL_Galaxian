# agents/dqn_agent.py
from __future__ import annotations
import math
import random
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from common.networks import DQNAtariCNN
from common.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """Deep Q-Network agent for ALE/Galaxian."""

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-4,
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 200_000,
        target_update_freq: int = 10_000,
    ) -> None:
        self.device = torch.device(device)
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        obs_shape = obs_space.shape  # (H, W, C)
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]

        self.q_net = DQNAtariCNN((c, h, w), action_space.n).to(self.device)
        self.target_net = DQNAtariCNN((c, h, w), action_space.n).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_shape=obs_shape,
        )

    def act(self, obs: np.ndarray, info: Dict[str, Any], action_space: gym.Space) -> int:
        """Epsilon-greedy action selection."""
        self.step_count += 1
        eps = self._current_epsilon()
        if random.random() < eps:
            return int(action_space.sample())

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)  # (1, H, W, C)
        obs_t = obs_t.permute(0, 3, 1, 2)  # (1, C, H, W)
        obs_t = obs_t / 255.0

        with torch.no_grad():
            q_values = self.q_net(obs_t)
            action = int(q_values.argmax(dim=1).item())
        return action

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition into replay buffer."""
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )
        self.replay_buffer.add(transition)

    def train_step(self) -> Dict[str, float]:
        """Sample from buffer and update Q-network."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Compute current Q-values
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item()), "epsilon": self._current_epsilon()}

    def _current_epsilon(self) -> float:
        """Compute epsilon for epsilon-greedy schedule."""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.step_count / self.eps_decay
        )

    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model parameters."""
        state_dict = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.q_net.state_dict())
