"""F, B, and Actor networks for MORL-FB.

Architecture follows Touati et al. (2023) "Does Zero-Shot RL Exist?" with a
preprocessing stage that projects (s, z) and (s, a, z) into a shared feature
space before producing the d_z-dimensional Forward / Backward outputs.

  Q(s, a, z) = F(s, a, z)^T z       (Eq. 1, 17)
  pi(s; z)   = arg max_a F(s, a, z)^T z
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: list[int], activation: nn.Module = nn.ReLU, output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Preprocessor(nn.Module):
    """Maps (s, z) and optionally a -> a feature vector of dim feat_dim."""

    def __init__(self, in_dim: int, feat_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardNetwork(nn.Module):
    """F(s, a, z) -> R^{d_z}.

    Two preprocessors (s,z) and (s,a,z) per Touati et al. 2023; their features
    are concatenated and run through an MLP that outputs a d_z-dim vector.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        feat_dim: int = 512,
        hidden: int = 1024,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.pre_sz = Preprocessor(state_dim + z_dim, feat_dim, hidden)
        self.pre_saz = Preprocessor(state_dim + action_dim + z_dim, feat_dim, hidden)
        self.trunk = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h_sz = self.pre_sz(torch.cat([s, z], dim=-1))
        h_saz = self.pre_saz(torch.cat([s, a, z], dim=-1))
        return self.trunk(torch.cat([h_sz, h_saz], dim=-1))


class BackwardNetwork(nn.Module):
    """B(s, a) -> R^{d_z}.  State-action-based backward representation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        hidden: int = 512,
        state_action: bool = True,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.state_action = state_action
        in_dim = state_dim + action_dim if state_action else state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        if self.state_action:
            assert a is not None
            x = torch.cat([s, a], dim=-1)
        else:
            x = s
        return self.net(x)


class Actor(nn.Module):
    """Deterministic policy pi(s; z) -> a in [-max_action, max_action]."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        z_dim: int,
        max_action: float = 1.0,
        feat_dim: int = 512,
        hidden: int = 1024,
    ):
        super().__init__()
        self.max_action = max_action
        self.pre = Preprocessor(state_dim + z_dim, feat_dim, hidden)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.pre(torch.cat([s, z], dim=-1))
        return self.max_action * self.trunk(h)
