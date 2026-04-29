"""Evaluation utilities: Utility (UT), Hypervolume (HV), Episodic Dominance (ED).

Definitions follow Sec. 4 ("Evaluation Metrics") of the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from morl_fb.agent import MORLFBAgent, MORLFBConfig
from morl_fb.replay_buffer import MOReplayBuffer
from morl_fb.utils import sample_preference_batch


# --------- Hypervolume ---------

def hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    """HV of a set of d-dim points dominating ref. Uses pymoo if available."""
    try:
        from pymoo.indicators.hv import HV
        # pymoo HV minimizes by default; multi-objective MORL maximizes, so flip signs.
        ind = HV(ref_point=-ref)
        return float(ind(-points))
    except ImportError:
        # Fallback: 2-D hypervolume only.
        if points.shape[1] != 2:
            raise
        pts = points[points[:, 0].argsort()[::-1]]
        hv = 0.0
        prev_y = ref[1]
        for x, y in pts:
            if x <= ref[0] or y <= ref[1]:
                continue
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
        return float(hv)


# --------- Pareto front ---------

def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return non-dominated rows of `points` (assuming maximization)."""
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (points >= points[i]).all(axis=1) & (points > points[i]).any(axis=1)
        keep[dominated] = False
    return points[keep]


# --------- Rollout ---------

@torch.no_grad()
def rollout(agent: MORLFBAgent, env, lam: np.ndarray, buffer: MOReplayBuffer | None = None,
            max_steps: int | None = None) -> np.ndarray:
    obs, _ = env.reset()
    if buffer is not None and len(buffer) > 0:
        z = agent.pg_explore(torch.as_tensor(lam, device=agent.device), buffer)
    else:
        z = agent.warmup_z(1)
    total = np.zeros(env.unwrapped.reward_space.shape[0], dtype=np.float64)
    done = trunc = False
    t = 0
    while not (done or trunc):
        a = agent.select_action(obs, z, noise=0.0)
        obs, r_vec, done, trunc, _ = env.step(a)
        total += np.asarray(r_vec, dtype=np.float64)
        t += 1
        if max_steps is not None and t >= max_steps:
            break
    return total


def evaluate(agent: MORLFBAgent, env, n_preferences: int = 100, seed: int = 0,
             ref_point: np.ndarray | None = None, buffer: MOReplayBuffer | None = None
             ) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    reward_dim = env.unwrapped.reward_space.shape[0]
    prefs = sample_preference_batch(reward_dim, n_preferences, rng)
    returns = np.stack([rollout(agent, env, lam, buffer) for lam in prefs])    # [N, d]
    utility = float((prefs * returns).sum(-1).mean())
    pf = pareto_front(returns)
    hv = float("nan") if ref_point is None else hypervolume(pf, np.asarray(ref_point))
    return {
        "UT": utility,
        "HV": hv,
        "n_pareto": int(pf.shape[0]),
        "mean_return_vec": returns.mean(0).tolist(),
    }


def episodic_dominance(returns_a: np.ndarray, returns_b: np.ndarray, prefs: np.ndarray) -> float:
    """ED(A, B) = E_lam [ 1{ lam^T g(tau_A) >= lam^T g(tau_B) } ]."""
    sa = (prefs * returns_a).sum(-1)
    sb = (prefs * returns_b).sum(-1)
    return float((sa >= sb).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n-prefs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref", type=float, nargs="+", default=None,
                   help="HV reference point (e.g. --ref 0 -8000)")
    args = p.parse_args()

    import mo_gymnasium as mo_gym
    env = mo_gym.make(args.env)
    env.reset(seed=args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MORLFBConfig(**ckpt["cfg"])
    agent = MORLFBAgent(cfg)
    agent.load_state_dict(ckpt["state"])
    agent.F.eval(); agent.B.eval(); agent.actor.eval()

    metrics = evaluate(agent, env, n_preferences=args.n_prefs, seed=args.seed,
                       ref_point=np.asarray(args.ref) if args.ref else None)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
