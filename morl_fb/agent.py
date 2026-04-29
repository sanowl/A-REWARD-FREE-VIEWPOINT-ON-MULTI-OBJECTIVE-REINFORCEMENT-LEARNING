"""MORL-FB agent.

Implements Algorithm 2 of "A Reward-Free Viewpoint on Multi-Objective RL"
(ICLR 2026):

  - PG-Explore (Algorithm 1, lines 11-13): z is built from preference-weighted
    rewards drawn from the replay buffer.
  - Measure loss (Eq. 14)         updates F (theta) and B (omega).
  - Auxiliary Q loss (Eq. 15)     updates F using observed reward vectors.
  - Orthonormality loss (Eq. 16)  updates B.
  - Policy loss (Eq. 17)          updates the actor (delayed, TD3-style).
  - Soft target updates with coefficient tau.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from morl_fb.networks import Actor, BackwardNetwork, ForwardNetwork
from morl_fb.replay_buffer import MOReplayBuffer
from morl_fb.utils import normalize_z


@dataclass
class MORLFBConfig:
    state_dim: int
    action_dim: int
    reward_dim: int
    max_action: float = 1.0

    z_dim: int = 300                     # latent dim (Table 12)
    feat_dim: int = 512                  # preprocessing feature dim
    hidden: int = 1024                   # MLP hidden width

    discount: float = 0.99
    tau: float = 0.01                    # target smoothing coefficient
    lr: float = 1e-4
    batch_size: int = 1024
    interface_batch: int = 5120          # n_s for PG-Explore
    policy_delay: int = 10               # update actor / targets every n_u
    target_noise: float = 0.2            # TD3 target policy smoothing
    target_noise_clip: float = 0.5
    expl_noise: float = 0.1              # action exploration noise

    q_loss_coef: float = 1.0
    measure_loss_coef: float = 1.0
    ortho_loss_coef: float = 1.0

    device: str = "cuda"


class MORLFBAgent:
    def __init__(self, cfg: MORLFBConfig):
        self.cfg = cfg
        d = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.device = d

        self.F = ForwardNetwork(cfg.state_dim, cfg.action_dim, cfg.z_dim, cfg.feat_dim, cfg.hidden).to(d)
        self.B = BackwardNetwork(cfg.state_dim, cfg.action_dim, cfg.z_dim, hidden=cfg.feat_dim).to(d)
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.z_dim, cfg.max_action, cfg.feat_dim, cfg.hidden).to(d)

        self.F_target = copy.deepcopy(self.F).requires_grad_(False)
        self.B_target = copy.deepcopy(self.B).requires_grad_(False)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)

        self.fb_optim = torch.optim.Adam(
            list(self.F.parameters()) + list(self.B.parameters()), lr=cfg.lr
        )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self._update_step = 0

    # ---------- z construction ----------

    @torch.no_grad()
    def pg_explore(self, lam: torch.Tensor, buffer: MOReplayBuffer) -> torch.Tensor:
        """Algorithm 1 / line 30-35 of Algorithm 2.

        z = (1/n_s) * sum B(s, a) (lam^T r),  then normalized.
        `lam` is shape [reward_dim] or [batch, reward_dim]; returns z of shape
        [batch, z_dim] (or [z_dim] if input was 1-D).
        """
        squeeze = (lam.dim() == 1)
        if squeeze:
            lam = lam.unsqueeze(0)
        bsz = lam.shape[0]
        n_s = self.cfg.interface_batch
        batch = buffer.sample_non_terminal(n_s)
        s, a, r = batch["s"], batch["a"], batch["r"]
        b_sa = self.B_target(s, a)                         # [n_s, z_dim]
        # scalarized reward per sample under each preference: [bsz, n_s]
        scalar_r = lam @ r.T                               # [bsz, n_s]
        # weighted average of B by scalar reward: z[bsz] = (scalar_r @ b_sa) / n_s
        z = (scalar_r @ b_sa) / n_s                        # [bsz, z_dim]
        z = normalize_z(z, self.cfg.z_dim)
        return z.squeeze(0) if squeeze else z

    @torch.no_grad()
    def warmup_z(self, batch_size: int = 1) -> torch.Tensor:
        z = torch.randn(batch_size, self.cfg.z_dim, device=self.device)
        z = normalize_z(z, self.cfg.z_dim)
        return z.squeeze(0) if batch_size == 1 else z

    # ---------- action selection ----------

    @torch.no_grad()
    def select_action(self, state: np.ndarray, z: torch.Tensor, noise: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_in = z.unsqueeze(0) if z.dim() == 1 else z
        a = self.actor(s, z_in)
        if noise > 0:
            a = a + noise * torch.randn_like(a)
            a = a.clamp(-self.cfg.max_action, self.cfg.max_action)
        return a.squeeze(0).cpu().numpy()

    # ---------- losses & update ----------

    def _compute_losses(
        self, batch1: dict[str, torch.Tensor], batch2: dict[str, torch.Tensor], z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Returns (L_M, L_Q, L_n, info).  z has shape [batch, z_dim]."""
        cfg = self.cfg
        s, a, r, s2, done, lam = batch1["s"], batch1["a"], batch1["r"], batch1["s2"], batch1["done"], batch1["lam"]
        s_b, a_b = batch2["s"], batch2["a"]      # independent (s', a') batch for off-diag terms

        # --- targets ---
        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.target_noise).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
            a2 = (self.actor_target(s2, z) + noise).clamp(-cfg.max_action, cfg.max_action)
            F_tgt_next = self.F_target(s2, a2, z)              # [bs, z_dim]
            B_tgt_b = self.B_target(s_b, a_b)                  # [bs, z_dim]

        # --- F(s,a,z), B(s_b,a_b), B(s2, a2_policy) ---
        F_sa = self.F(s, a, z)                                 # [bs, z_dim]
        B_b = self.B(s_b, a_b)                                 # [bs, z_dim]

        # ============== Measure loss (Eq. 14) ==============
        # off-diagonal: (F(s,a,z)^T B(s_b,a_b) - gamma * F_tgt(s2, a2, z)^T B_tgt(s_b,a_b))^2
        not_done = 1.0 - done                                  # [bs, 1]
        # Outer products: [bs_F, bs_B]
        FB = F_sa @ B_b.T
        FB_tgt = F_tgt_next @ B_tgt_b.T
        # apply terminal mask along F-axis
        target = cfg.discount * not_done * FB_tgt
        L_M_off = (FB - target).pow(2).mean()

        # diagonal: -2 E[F(s,a,z)^T B(s2, a_next)] using on-policy a_next from target actor
        with torch.no_grad():
            a_next = (self.actor_target(s2, z) + noise).clamp(-cfg.max_action, cfg.max_action)
        B_next = self.B(s2, a_next)                            # grad through B only
        L_M_diag = -2.0 * (F_sa * B_next).sum(-1).mean()
        L_M = L_M_off + L_M_diag

        # ============== Auxiliary Q loss (Eq. 15) ==============
        # L_Q = (F(s,a,z)^T z - (lam^T r + gamma * F_tgt(s2, a2, z)^T z))^2
        q = (F_sa * z).sum(-1, keepdim=True)                   # [bs, 1]
        with torch.no_grad():
            scalar_r = (lam * r).sum(-1, keepdim=True)         # [bs, 1]
            q_tgt = (F_tgt_next * z).sum(-1, keepdim=True)
            q_target = scalar_r + cfg.discount * not_done * q_tgt
        L_Q = F_nn.mse_loss(q, q_target)

        # ============== Orthonormality loss (Eq. 16) ==============
        # E[(B(s,a)^T B(s',a'))^2 - ||B(s,a)||^2 - ||B(s',a')||^2]
        B_a = self.B(s, a)
        cov = B_a @ B_b.T                                      # [bs, bs]
        L_n = cov.pow(2).mean() - B_a.pow(2).sum(-1).mean() - B_b.pow(2).sum(-1).mean()

        info = {
            "loss/M_off": L_M_off.item(),
            "loss/M_diag": L_M_diag.item(),
            "loss/Q": L_Q.item(),
            "loss/ortho": L_n.item(),
            "stats/q_mean": q.mean().item(),
            "stats/q_target_mean": q_target.mean().item(),
            "stats/F_norm": F_sa.norm(dim=-1).mean().item(),
            "stats/B_norm": B_a.norm(dim=-1).mean().item(),
        }
        return L_M, L_Q, L_n, info

    def update(self, buffer: MOReplayBuffer, lam_np: np.ndarray) -> dict[str, float]:
        """One gradient step (Algorithm 2 lines 17-28).

        `lam_np` is the preference vector for this iteration (np.ndarray of
        shape [reward_dim])."""
        cfg = self.cfg
        if len(buffer) < cfg.batch_size:
            return {}

        # Sample two independent batches (one for the main term, one for off-diag B)
        batch1 = buffer.sample(cfg.batch_size)
        batch2 = buffer.sample(cfg.batch_size)

        # z for this gradient step: use PG-Explore w/ the current preference, broadcast to batch
        lam = torch.as_tensor(lam_np, dtype=torch.float32, device=self.device)
        z_single = self.pg_explore(lam, buffer)                # [z_dim]
        z = z_single.unsqueeze(0).expand(cfg.batch_size, -1)   # [bs, z_dim]

        L_M, L_Q, L_n, info = self._compute_losses(batch1, batch2, z)
        loss_fb = (
            cfg.measure_loss_coef * L_M
            + cfg.q_loss_coef * L_Q
            + cfg.ortho_loss_coef * L_n
        )

        self.fb_optim.zero_grad(set_to_none=True)
        loss_fb.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.F.parameters()) + list(self.B.parameters()), max_norm=10.0
        )
        self.fb_optim.step()

        self._update_step += 1

        # Delayed actor + target update
        if self._update_step % cfg.policy_delay == 0:
            actor_info = self._update_actor(batch1["s"], z)
            info.update(actor_info)
            self._soft_update(self.F, self.F_target, cfg.tau)
            self._soft_update(self.B, self.B_target, cfg.tau)
            self._soft_update(self.actor, self.actor_target, cfg.tau)

        return info

    def _update_actor(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
        # Freeze F so the policy gradient does not modify the critic
        for p in self.F.parameters():
            p.requires_grad_(False)
        a_pi = self.actor(s, z)
        q_pi = (self.F(s, a_pi, z) * z).sum(-1)
        loss_pi = -q_pi.mean()

        self.actor_optim.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optim.step()
        for p in self.F.parameters():
            p.requires_grad_(True)
        return {"loss/pi": loss_pi.item()}

    @staticmethod
    @torch.no_grad()
    def _soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
        for p, p_t in zip(net.parameters(), target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # ---------- save / load ----------

    def state_dict(self) -> dict:
        return {
            "F": self.F.state_dict(),
            "B": self.B.state_dict(),
            "actor": self.actor.state_dict(),
            "F_target": self.F_target.state_dict(),
            "B_target": self.B_target.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "fb_optim": self.fb_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "step": self._update_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.F.load_state_dict(sd["F"])
        self.B.load_state_dict(sd["B"])
        self.actor.load_state_dict(sd["actor"])
        self.F_target.load_state_dict(sd["F_target"])
        self.B_target.load_state_dict(sd["B_target"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.fb_optim.load_state_dict(sd["fb_optim"])
        self.actor_optim.load_state_dict(sd["actor_optim"])
        self._update_step = sd.get("step", 0)
