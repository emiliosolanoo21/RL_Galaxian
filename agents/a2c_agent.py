# agents/a2c_agent.py
from __future__ import annotations

from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

from common.networks import ActorCriticAtariCNN


class A2CAgent:
    """Advantage Actor-Critic agent for ALE/Galaxian."""

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 7e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.action_space = action_space
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        obs_shape = obs_space.shape  # (H, W, C) after preprocessing
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]

        self.net = ActorCriticAtariCNN((c, h, w), action_space.n).to(self.device)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, alpha=0.99, eps=1e-5)

    def act(self, obs: np.ndarray, info: Dict[str, Any], action_space: gym.Space) -> int:
        """Greedy action selection for evaluation."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)  # (1, H, W, C)
        obs_t = obs_t.permute(0, 3, 1, 2)  # (1, C, H, W)
        obs_t = obs_t / 255.0

        with torch.no_grad():
            logits, _ = self.net(obs_t)
            action = int(logits.argmax(dim=1).item())
        return action

    def select_action(self, obs: np.ndarray):
        """Sample action for training."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)
        obs_t = obs_t.permute(0, 3, 1, 2)
        obs_t = obs_t / 255.0

        logits, value = self.net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            log_prob.squeeze(0),
            value.squeeze(0),
        )

    def _value_from_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Compute V(s) for a single observation."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.unsqueeze(0)
        obs_t = obs_t.permute(0, 3, 1, 2)
        obs_t = obs_t / 255.0
        with torch.no_grad():
            _, value = self.net(obs_t)
        return value.squeeze(0)

    def update_from_rollout(
        self,
        obs_list: List[np.ndarray],
        actions_list: List[int],
        rewards_list: List[float],
        dones_list: List[bool],
        last_obs: np.ndarray,
        last_done: bool,
    ):
        """Update network using one rollout."""
        if len(obs_list) == 0:
            return {}

        device = self.device
        T = len(obs_list)

        obs_np = np.stack(obs_list, axis=0)  # (T, H, W, C)
        actions_np = np.array(actions_list, dtype=np.int64)
        rewards_np = np.array(rewards_list, dtype=np.float32)
        dones_np = np.array(dones_list, dtype=np.bool_)

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device) / 255.0
        obs = obs.permute(0, 3, 1, 2)  # (T, C, H, W)

        actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones_np.astype(np.float32), dtype=torch.float32, device=device)

        logits, values = self.net(obs)  # values: (T,)

        if last_done:
            next_value = torch.zeros(1, device=device)
        else:
            next_value = self._value_from_obs(last_obs).unsqueeze(0)

        returns = torch.zeros(T, device=device)
        R = next_value
        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * (1.0 - dones[t]) * R
            returns[t] = R

        advantages = returns - values

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str) -> None:
        """Save model parameters."""
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model parameters."""
        state_dict = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dict)
