# common/networks.py
from __future__ import annotations
from typing import Tuple

import torch
from torch import nn


class DQNAtariCNN(nn.Module):
    """Simple Atari-style CNN for Q-value approximation."""

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()
        c, h, w = obs_shape  # (C, H, W)

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.features(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) in [0, 1]
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)
