# common/replay_buffer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Cyclic replay buffer for off-policy algorithms like DQN."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.pos = 0
        self.full = False

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, transition: Transition) -> None:
        idx = self.pos
        self.obs[idx] = transition.obs
        self.next_obs[idx] = transition.next_obs
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.dones[idx] = transition.done

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device: torch.device):
        max_index = self.capacity if self.full else self.pos
        idxs = np.random.randint(0, max_index, size=batch_size)

        obs = torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=device) / 255.0
        next_obs = torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=device) / 255.0
        actions = torch.as_tensor(self.actions[idxs], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device)

        # Convert to (B, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
        next_obs = next_obs.permute(0, 3, 1, 2)
        return obs, actions, rewards, next_obs, dones
