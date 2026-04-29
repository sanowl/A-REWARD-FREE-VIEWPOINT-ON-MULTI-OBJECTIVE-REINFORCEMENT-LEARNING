"""Helpers for preference sampling and z-vector normalization."""
from __future__ import annotations

import math
import numpy as np
import torch


def sample_preference(reward_dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a preference vector uniformly from the (d-1)-simplex (Dirichlet(1))."""
    rng = rng or np.random.default_rng()
    return rng.dirichlet(np.ones(reward_dim)).astype(np.float32)


def sample_preference_batch(reward_dim: int, batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.dirichlet(np.ones(reward_dim), size=batch_size).astype(np.float32)


def normalize_z(z: torch.Tensor, z_dim: int, eps: float = 1e-8) -> torch.Tensor:
    """z <- sqrt(d_z) * z / ||z||  (Algorithm 2 line 13, line 33)."""
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    return math.sqrt(z_dim) * z / norm
