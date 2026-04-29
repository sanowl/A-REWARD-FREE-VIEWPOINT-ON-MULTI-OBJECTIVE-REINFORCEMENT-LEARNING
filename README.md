# MORL-FB (re-implementation)

> **This is an unofficial PyTorch re-implementation.** All algorithmic ideas
> (MORL-FB, Preference-Guided Exploration, the Auxiliary Q loss, etc.)
> are due to the original authors. Please cite their paper, not this repo.

Paper: **"A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning"**, ICLR 2026 — arXiv:2604.24532.

Authors: Ying-Tu Chen¹, Wei Hung¹, Bing-Shu Wu¹, Zhang-Wei Hong², Ping-Chun Hsieh¹.
¹ National Yang Ming Chiao Tung University, Taiwan. ² Massachusetts Institute of Technology.
Project page: https://rl-bandits-lab.github.io/MORL-FB/

```bibtex
@inproceedings{chen2026morlfb,
  title     = {A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning},
  author    = {Chen, Ying-Tu and Hung, Wei and Wu, Bing-Shu and Hong, Zhang-Wei and Hsieh, Ping-Chun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

This repo is a from-scratch reproduction by a third party — no code from the
authors was used. It targets clarity over feature parity with the official
release; consult the paper / official code for definitive results.

---

## Method overview

MORL-FB adapts Forward-Backward (FB) reward-free RL representations
(Touati & Ollivier, 2021; Touati et al., 2023) to the multi-objective setting.
The Q-function for any preference-weighted reward `λᵀ r` is factorized as

```
Q(s, a, z) = F(s, a, z)ᵀ z,        z_R = E_{(s,a) ~ D}[ B(s, a) · λᵀ r(s, a) ]
```

so a single (F, B) pair can be queried zero-shot for any preference `λ`.

The three contributions over vanilla FB:

1. **Preference-Guided Exploration (PG-Explore)** — sample `z` from
   preference-weighted minibatches of replay-buffer rewards instead of
   `N(0, I)`, which keeps `z` close to the true `z_λ` for the MORL task.
2. **Auxiliary Q loss** — TD-error on the observed reward vector,
   `(F(s,a,z)ᵀz − (λᵀr + γ F̄(s', π̄(s',z), z)ᵀz))²`,
   replacing the pseudo-reward signal used in vanilla FB.
3. **Measure + orthonormality + policy losses** trained jointly with a
   TD3 backbone (delayed actor + target policy smoothing).

## Layout

```
morl_fb/
  networks.py       F, B, Actor (preprocessor + MLP)
  replay_buffer.py  Multi-objective replay buffer
  agent.py          MORL-FB agent: PG-Explore + all four losses
  utils.py          Preference sampling on the simplex, z normalization
  train.py          Algorithm 2 training loop
  eval.py           UT / HV / ED metrics + Pareto front
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m morl_fb.train --env mo-halfcheetah-v4 --total-steps 3000000 --seed 0
```

Defaults follow Table 12 of the paper (TD3 backbone, `d_z = 300`, `lr = 1e-4`,
batch 1024, interface batch 5120, policy delay 10, `τ = 0.01`).

## Evaluate

```bash
python -m morl_fb.eval --env mo-halfcheetah-v4 \
       --checkpoint runs/.../ckpt_final.pt --ref 0 -8000
```

Reports Utility (UT) and Hypervolume (HV) over a uniform sample of preferences
from the d-simplex; Episodic Dominance (ED) is exposed as a helper for
pairwise comparisons against another algorithm's returns.

## Status

| Component | Status |
|---|---|
| F / B / Actor networks (preprocessor + MLP) | done |
| Multi-objective replay buffer | done |
| PG-Explore (Algorithm 1) | done |
| Measure loss (Eq. 14) | done |
| Auxiliary Q loss (Eq. 15) | done |
| Orthonormality loss (Eq. 16) | done |
| Policy loss + delayed actor (Eq. 17) | done |
| UT / HV / Pareto-front evaluation | done |
| Algorithm 2 training loop with warmup | done |
| Full benchmark sweep (Halfcheetah/Walker/Hopper/Ant/Humanoid + DST/FTN) | not run |

The training loop and all four losses pass a CPU smoke test on synthetic
transitions; large-scale results matching the paper's Figure 2 have not been
reproduced in this repo.

## Caveats / known simplifications

- A single F network rather than twin critics; the paper's "TD3 backbone"
  refers mainly to delayed actor updates and target policy smoothing,
  both of which are implemented.
- The diagonal term of the measure loss (`-2 E[F(s,a,z)ᵀ B(s', a_next)]`) uses
  `a_next = π̄(s', z) + clipped noise` instead of a stored next-action,
  matching common online FB conventions.
- Discrete-action environments (Deep Sea Treasure, Fruit Tree Navigation)
  are not wrapped; the codebase currently assumes a continuous Box action
  space.
