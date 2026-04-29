"""Training loop for MORL-FB (Algorithm 2)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from morl_fb.agent import MORLFBAgent, MORLFBConfig
from morl_fb.replay_buffer import MOReplayBuffer
from morl_fb.utils import sample_preference


def make_env(env_id: str, seed: int):
    """Create a multi-objective Gymnasium env. Falls back to mo-gymnasium."""
    try:
        import mo_gymnasium as mo_gym
    except ImportError as e:
        raise ImportError(
            "mo-gymnasium is required: pip install mo-gymnasium"
        ) from e
    env = mo_gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate(agent: MORLFBAgent, env, n_preferences: int = 10, n_episodes: int = 1, seed: int = 0):
    """Quick eval: average scalarized return across uniformly sampled preferences."""
    rng = np.random.default_rng(seed)
    reward_dim = env.unwrapped.reward_space.shape[0]
    returns = []
    for _ in range(n_preferences):
        lam = sample_preference(reward_dim, rng)
        z = agent.pg_explore(torch.as_tensor(lam, device=agent.device), agent._eval_buffer) \
            if hasattr(agent, "_eval_buffer") and len(agent._eval_buffer) > 0 \
            else agent.warmup_z(1)
        ep_returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = trunc = False
            scal_ret = 0.0
            while not (done or trunc):
                a = agent.select_action(obs, z, noise=0.0)
                obs, r_vec, done, trunc, _ = env.step(a)
                scal_ret += float(np.dot(lam, r_vec))
            ep_returns.append(scal_ret)
        returns.append(np.mean(ep_returns))
    return float(np.mean(returns)), float(np.std(returns))


def train(args: argparse.Namespace) -> None:
    env = make_env(args.env, args.seed)
    obs, _ = env.reset(seed=args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reward_dim = env.unwrapped.reward_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"[env] {args.env}  s={state_dim}  a={action_dim}  d={reward_dim}  |A|max={max_action}")

    cfg = MORLFBConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        max_action=max_action,
        z_dim=args.z_dim,
        feat_dim=args.feat_dim,
        hidden=args.hidden,
        discount=args.gamma,
        tau=args.tau,
        lr=args.lr,
        batch_size=args.batch_size,
        interface_batch=args.interface_batch,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        target_noise_clip=args.target_noise_clip,
        expl_noise=args.expl_noise,
        device=args.device,
    )
    agent = MORLFBAgent(cfg)
    buffer = MOReplayBuffer(args.buffer_size, state_dim, action_dim, reward_dim, device=agent.device)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path(args.log_dir) / f"{args.env}_seed{args.seed}_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[log] {log_dir}")

    # Algorithm 2 main loop -------------------------------------------------
    obs, _ = env.reset(seed=args.seed)
    lam = sample_preference(reward_dim, rng)
    if len(buffer) > 0:
        z_t = agent.pg_explore(torch.as_tensor(lam, device=agent.device), buffer)
    else:
        z_t = agent.warmup_z(1)

    ep_return_vec = np.zeros(reward_dim, dtype=np.float32)
    ep_len = 0
    ep_scal_return = 0.0
    rollout_steps = args.steps_per_episode
    total_steps = args.total_steps
    warmup_steps = args.warmup_steps

    t0 = time.time()
    for step in range(1, total_steps + 1):
        # action with exploration noise
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, z_t, noise=cfg.expl_noise * max_action)

        next_obs, r_vec, terminated, truncated, _ = env.step(action)
        done_for_buffer = bool(terminated)  # do not bootstrap through real terminations
        buffer.add(obs, action, np.asarray(r_vec, dtype=np.float32), lam, next_obs, done_for_buffer)

        ep_return_vec += np.asarray(r_vec, dtype=np.float32)
        ep_scal_return += float(np.dot(lam, r_vec))
        ep_len += 1
        obs = next_obs

        if terminated or truncated or ep_len >= rollout_steps:
            obs, _ = env.reset()
            # Resample preference and z for next rollout (lines 4-10 of Alg. 2)
            lam = sample_preference(reward_dim, rng)
            if step <= warmup_steps or len(buffer) < cfg.batch_size:
                z_t = agent.warmup_z(1)
            else:
                z_t = agent.pg_explore(torch.as_tensor(lam, device=agent.device), buffer)
            ep_return_vec[:] = 0.0
            ep_scal_return = 0.0
            ep_len = 0

        # gradient updates start once buffer has enough samples
        if step > warmup_steps and len(buffer) >= cfg.batch_size:
            info = agent.update(buffer, lam)
            if step % args.log_every == 0 and info:
                elapsed = time.time() - t0
                fps = step / max(elapsed, 1e-9)
                print(
                    f"[{step:>8d}] fps={fps:6.1f}  "
                    f"L_M={info.get('loss/M_off',0):.3f}/{info.get('loss/M_diag',0):.3f}  "
                    f"L_Q={info.get('loss/Q',0):.3f}  "
                    f"L_n={info.get('loss/ortho',0):.3f}  "
                    f"L_pi={info.get('loss/pi', float('nan')):.3f}  "
                    f"|F|={info.get('stats/F_norm',0):.2f}  |B|={info.get('stats/B_norm',0):.2f}"
                )

        if step % args.eval_every == 0 and step > warmup_steps:
            agent._eval_buffer = buffer  # type: ignore[attr-defined]
            ret_mean, ret_std = evaluate(agent, env, n_preferences=args.eval_prefs)
            print(f"[eval @ {step}] mean scalarized return = {ret_mean:.2f} +/- {ret_std:.2f}")

        if step % args.save_every == 0 and step > warmup_steps:
            ckpt_path = log_dir / f"ckpt_{step}.pt"
            torch.save({"cfg": cfg.__dict__, "state": agent.state_dict()}, ckpt_path)
            print(f"[save] {ckpt_path}")

    final = log_dir / "ckpt_final.pt"
    torch.save({"cfg": cfg.__dict__, "state": agent.state_dict()}, final)
    print(f"[done] saved {final}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="mo-halfcheetah-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--steps-per-episode", type=int, default=1000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)

    # Network / agent (Table 12 defaults)
    p.add_argument("--z-dim", type=int, default=300)
    p.add_argument("--feat-dim", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--interface-batch", type=int, default=5120)
    p.add_argument("--policy-delay", type=int, default=10)
    p.add_argument("--target-noise", type=float, default=0.2)
    p.add_argument("--target-noise-clip", type=float, default=0.5)
    p.add_argument("--expl-noise", type=float, default=0.1)

    # logging / eval
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-prefs", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
