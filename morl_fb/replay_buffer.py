"""Replay buffer that stores vector-valued (multi-objective) rewards."""
from __future__ import annotations

import numpy as np
import torch


class MOReplayBuffer:
    """Circular buffer for transitions (s, a, r_vec, lambda, s', done).

    Stored on CPU as numpy; moved to device in `sample`.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        reward_dim: int,
        device: str | torch.device = "cpu",
    ):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = torch.device(device)

        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((self.capacity, reward_dim), dtype=np.float32)
        self.lam = np.zeros((self.capacity, reward_dim), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        lam: np.ndarray,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.lam[self.ptr] = lam
        self.s2[self.ptr] = s2
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _to(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).to(self.device)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "s": self._to(self.s[idx]),
            "a": self._to(self.a[idx]),
            "r": self._to(self.r[idx]),
            "lam": self._to(self.lam[idx]),
            "s2": self._to(self.s2[idx]),
            "done": self._to(self.done[idx]),
        }

    def sample_non_terminal(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Used by PG-Explore (Algorithm 1, line 11): non-terminal transitions."""
        if self.size == 0:
            return self.sample(batch_size)
        nonterm = np.where(self.done[: self.size, 0] == 0.0)[0]
        if nonterm.size == 0:
            return self.sample(batch_size)
        idx = nonterm[np.random.randint(0, nonterm.size, size=batch_size)]
        return {
            "s": self._to(self.s[idx]),
            "a": self._to(self.a[idx]),
            "r": self._to(self.r[idx]),
            "lam": self._to(self.lam[idx]),
            "s2": self._to(self.s2[idx]),
            "done": self._to(self.done[idx]),
        }

    def __len__(self) -> int:
        return self.size
