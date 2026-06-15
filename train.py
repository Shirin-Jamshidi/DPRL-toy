# """
# Diffusion Q-Learning for CartPole-v1
# =====================================
# Combines:
#   - QVPO  : Q-guided score-based policy via diffusion (https://arxiv.org/abs/2407.xxxxx)
#   - Hy-Q  : Hybrid offline + online Q-learning with importance-weighted mixing
#              (https://arxiv.org/abs/2312.xxxxx)

# Architecture
# ------------
#   OfflineBuffer      : fixed demonstration data (40k transitions from Q-learning agent)
#   OnlineBuffer       : ring buffer filled during environment interaction
#   DiffusionPolicy    : DDPM-style denoising network conditioned on (state, Q-guidance)
#   QNetwork (x2)      : twin critics for pessimistic value estimation
#   HyQMixer           : samples mixed batches with Hy-Q priority weighting
#   Trainer            : drives the offline-pretraining → online-finetuning loop

# QVPO key ideas implemented
# ---------------------------
#   1. Score network   s_θ(a_t, t, s) approximates ∇_{a_t} log p(a_t | s)
#   2. Q-guidance      during reverse diffusion, inject α·∇_a Q(s,a) into the score
#   3. Policy loss     combines denoising score matching + in-support BC regularisation

# Hy-Q key ideas implemented
# ---------------------------
#   1. Unified replay  single buffer interface over offline + online data
#   2. Priority mixing offline samples weighted by |TD-error| from a reference Q*
#   3. Smooth ramp     β anneals from 1 (pure offline) → 0 (pure online) over N steps
# """

# import os
# import copy
# import argparse
# from typing import Tuple, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from env import set_seed, build_config
# from buffer import OfflineBuffer, OnlineBuffer, HyQMixer
# from models import ScoreNetwork, QNetwork, DiscreteGaussianDiffusion

# # ──────────────────────────────────────────────
# # Reproducibility helpers
# # ──────────────────────────────────────────────

# # `set_seed` and configuration live in `env.py`.


# # Replay buffers and model classes were moved to `buffer.py` and `models.py`.
# def qvpo_policy_loss(
#     score_net:  ScoreNetwork,
#     diffusion:  DiscreteGaussianDiffusion,
#     q1:         QNetwork,
#     q2:         QNetwork,
#     batch:      dict,
#     bc_weight:  float = 0.5,
# ) -> torch.Tensor:
#     """
#     L_QVPO = L_DSM + λ_BC · L_BC

#     L_DSM : Denoising Score Matching — makes diffusion model fit action distribution
#     L_BC  : Behaviour Cloning term  — weighted by advantage to stay in-support of
#             high-return demonstrations
#     """
#     states  = batch["states"]
#     actions = batch["actions"]   # (B,) long

#     # ── 1. DSM loss ──────────────────────────────────────────────────
#     l_dsm = diffusion.p_losses(score_net, actions, states)

#     # ── 2. BC loss weighted by Q-advantage (QVPO eq. 5) ─────────────
#     with torch.no_grad():
#         q_vals  = torch.min(q1(states), q2(states))       # (B, n_actions)
#         q_sa    = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
#         q_mean  = q_vals.mean(dim=1)                       # (B,)
#         adv     = q_sa - q_mean                            # (B,)
#         weights = torch.exp(adv.clamp(-5, 5))              # soft-max advantage weighting

#     # BC: push policy toward demonstrated actions, scaled by advantage
#     B      = states.shape[0]
#     t_bc   = torch.zeros(B, dtype=torch.long, device=states.device)  # t=0 → clean
#     logits = score_net(actions, t_bc, states)[:, :2]                 # (B, 2)
#     l_bc   = (weights * F.cross_entropy(logits, actions, reduction="none")).mean()

#     return l_dsm + bc_weight * l_bc


# # ──────────────────────────────────────────────
# # Critic update (double Q + target networks)
# # ──────────────────────────────────────────────

# def critic_loss(
#     q1: QNetwork, q2: QNetwork,
#     q1_target: QNetwork, q2_target: QNetwork,
#     batch: dict,
#     gamma: float = 0.99,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Standard twin-Q Bellman loss with target networks."""
#     s, a, r, s_next, d = (
#         batch["states"], batch["actions"], batch["rewards"],
#         batch["next_states"], batch["dones"],
#     )

#     with torch.no_grad():
#         # Pessimistic bootstrap
#         q1_next = q1_target(s_next)          # (B, n_actions)
#         q2_next = q2_target(s_next)
#         q_next  = torch.min(q1_next, q2_next).max(dim=1).values   # (B,)
#         target  = r + gamma * (1.0 - d) * q_next                  # (B,)
#         target = target.clamp(-100, 100)

#     q1_pred = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)
#     q2_pred = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)

#     td1 = target - q1_pred
#     td2 = target - q2_pred

#     l1  = F.mse_loss(q1_pred, target)
#     l2  = F.mse_loss(q2_pred, target)

#     # Return mean TD error for Hy-Q priority update
#     td_err = ((td1 + td2) / 2.0).detach().cpu().numpy()
#     return (l1 + l2), td_err


# # ──────────────────────────────────────────────
# # Main Trainer
# # ──────────────────────────────────────────────

# class DiffusionQLTrainer:
#     """
#     Orchestrates the full QVPO + Hy-Q training loop.

#     Phase 1 — Offline pretraining  (steps 0 … offline_steps)
#       Trains critics and diffusion policy purely on demonstration data.
#       β = 1.0 (all offline).

#     Phase 2 — Online finetuning  (steps offline_steps … total_steps)
#       Collects experience with the current diffusion policy.
#       Hy-Q mixer anneals β from 1.0 → β_end, blending offline + online data.
#       Priorities of offline samples updated each step.
#     """

#     def __init__(self, cfg: argparse.Namespace, device: torch.device):
#         self.cfg    = cfg
#         self.device = device

#         # ── Buffers ──────────────────────────────────────────────────
#         self.offline_buf = OfflineBuffer(cfg.demo_path, device)
#         self.online_buf  = OnlineBuffer(cfg.online_capacity, cfg.state_dim, device)

#         # ── Networks ─────────────────────────────────────────────────
#         self.score_net = ScoreNetwork(
#             cfg.state_dim, cfg.n_actions, cfg.hidden_dim,
#             cfg.time_emb_dim, cfg.n_diffusion_steps
#         ).to(device)

#         self.q1        = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
#         self.q2        = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
#         self.q1_target = copy.deepcopy(self.q1)
#         self.q2_target = copy.deepcopy(self.q2)
#         for p in self.q1_target.parameters(): p.requires_grad_(False)
#         for p in self.q2_target.parameters(): p.requires_grad_(False)

#         # ── Diffusion schedule ───────────────────────────────────────
#         self.diffusion = DiscreteGaussianDiffusion(cfg.n_diffusion_steps).to(device)

#         # ── Optimisers ───────────────────────────────────────────────
#         self.opt_score = optim.Adam(self.score_net.parameters(), lr=cfg.lr_policy)
#         self.opt_q     = optim.Adam(
#             list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q
#         )

#         # ── Hy-Q Mixer ───────────────────────────────────────────────
#         self.mixer = HyQMixer(
#             self.offline_buf, self.online_buf,
#             beta_start=1.0, beta_end=cfg.hyq_beta_end,
#             anneal_steps=cfg.hyq_anneal_steps,
#             td_alpha=cfg.hyq_td_alpha,
#         )

#         # ── Logging ──────────────────────────────────────────────────
#         self.log = dict(critic_loss=[], policy_loss=[], episode_return=[])

#     # ------------------------------------------------------------------
#     def _soft_update(self, tau: float = 0.005):
#         for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
#             pt.data.lerp_(p.data, tau)
#         for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
#             pt.data.lerp_(p.data, tau)

#     # ------------------------------------------------------------------
#     def _update_step(self, step: int):
#         cfg = self.cfg

#         # ── Sample batch from Hy-Q mixer ─────────────────────────────
#         batch, off_idx = self.mixer.sample(
#             cfg.batch_size, self.q1, self.q2, cfg.gamma
#         )

#         # ── Critic update ─────────────────────────────────────────────
#         c_loss, td_errors = critic_loss(
#             self.q1, self.q2, self.q1_target, self.q2_target, batch, cfg.gamma
#         )
#         self.opt_q.zero_grad()
#         c_loss.backward()
#         nn.utils.clip_grad_norm_(
#             list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
#         )
#         self.opt_q.step()

#         # Update Hy-Q priorities with TD errors from the offline slice
#         if off_idx is not None:
#             n_off = len(off_idx)
#             self.mixer.update_priorities(off_idx, td_errors[:n_off])

#         # ── Policy (diffusion) update ─────────────────────────────────
#         p_loss = qvpo_policy_loss(
#             self.score_net, self.diffusion,
#             self.q1, self.q2, batch, cfg.bc_weight
#         )
#         self.opt_score.zero_grad()
#         p_loss.backward()
#         nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
#         self.opt_score.step()

#         # ── Target network update ─────────────────────────────────────
#         self._soft_update(cfg.tau)

#         return c_loss.item(), p_loss.item()

#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def select_action(self, state: np.ndarray) -> int:
#         s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
#         action = self.diffusion.p_sample(
#             self.score_net, s, self.q1, self.q2,
#             guidance_scale=self.cfg.guidance_scale
#         )
#         return action.item()

#     # ------------------------------------------------------------------
#     def offline_pretrain(self):
#         """Phase 1: pure offline training."""
#         cfg = self.cfg
#         print(f"\n{'='*60}")
#         print(f"  Phase 1 — Offline Pretraining ({cfg.offline_steps:,} steps)")
#         print(f"{'='*60}")

#         for step in range(1, cfg.offline_steps + 1):
#             batch = self.offline_buf.sample(cfg.batch_size)
#             # During offline phase, bypass Hy-Q mixer and use pure offline batch
#             c_loss, td_err = critic_loss(
#                 self.q1, self.q2, self.q1_target, self.q2_target, batch, cfg.gamma
#             )
#             self.opt_q.zero_grad()
#             c_loss.backward()
#             nn.utils.clip_grad_norm_(
#                 list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
#             )
#             self.opt_q.step()

#             p_loss = qvpo_policy_loss(
#                 self.score_net, self.diffusion,
#                 self.q1, self.q2, batch, cfg.bc_weight
#             )
#             self.opt_score.zero_grad()
#             p_loss.backward()
#             nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
#             self.opt_score.step()
#             self._soft_update(cfg.tau)

#             if step % cfg.log_interval == 0:
#                 print(f"  [offline {step:6d}/{cfg.offline_steps}]  "
#                       f"critic={c_loss.item():.4f}  policy={p_loss.item():.4f}  "
#                       f"β(mixer)={self.mixer.beta:.3f}")

#         print("  Offline pretraining done.\n")

#     # ------------------------------------------------------------------
#     def online_finetune(self, env):
#         """Phase 2: online finetuning with Hy-Q mixing."""
#         cfg = self.cfg
#         print(f"{'='*60}")
#         print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
#         print(f"{'='*60}")

#         state, _ = env.reset(seed=cfg.seed)
#         ep_return = 0.0
#         ep_count  = 0
#         global_step = 0

#         for step in range(1, cfg.online_steps + 1):
#             # ── Collect one environment step ──────────────────────────
#             action    = self.select_action(state)
#             ns, reward, term, trunc, _ = env.step(action)
#             done      = term or trunc
#             self.online_buf.add(state, action, reward, ns, float(done))
#             state     = ns
#             ep_return += reward
#             global_step += 1

#             if done:
#                 self.log["episode_return"].append(ep_return)
#                 ep_count += 1
#                 if ep_count % cfg.eval_interval_eps == 0:
#                     avg_ret = np.mean(self.log["episode_return"][-20:])
#                     print(f"  [online  {step:6d}/{cfg.online_steps}]  "
#                           f"ep={ep_count}  avg_return(20)={avg_ret:.1f}  "
#                           f"β={self.mixer.beta:.3f}  "
#                           f"online_size={self.online_buf.size}")
#                 ep_return = 0.0
#                 state, _ = env.reset()

#             # ── Wait for minimal online data ──────────────────────────
#             if self.online_buf.size < cfg.batch_size:
#                 continue

#             # ── Gradient update ────────────────────────────────────────
#             c_loss, p_loss = self._update_step(step)
#             self.log["critic_loss"].append(c_loss)
#             self.log["policy_loss"].append(p_loss)

#         print("  Online finetuning done.\n")

#     # ------------------------------------------------------------------
#     def evaluate(self, env, n_episodes: int = 10) -> float:
#         """Greedy evaluation of current diffusion policy."""
#         returns = []
#         for ep in range(n_episodes):
#             state, _ = env.reset()
#             ep_ret   = 0.0
#             done     = False
#             while not done:
#                 action = self.select_action(state)
#                 state, reward, term, trunc, _ = env.step(action)
#                 ep_ret += reward
#                 done    = term or trunc
#             returns.append(ep_ret)
#             print(f"    Episode {ep+1}: return = {ep_ret}")
#         mean_ret = float(np.mean(returns))
#         print(f"  Evaluation over {n_episodes} episodes: "
#               f"mean={mean_ret:.1f}  std={np.std(returns):.1f}")
#         return mean_ret

#     # ------------------------------------------------------------------
#     def save(self, path: str):
#         torch.save({
#             "score_net":  self.score_net.state_dict(),
#             "q1":         self.q1.state_dict(),
#             "q2":         self.q2.state_dict(),
#             "q1_target":  self.q1_target.state_dict(),
#             "q2_target":  self.q2_target.state_dict(),
#         }, path)
#         print(f"  Checkpoint saved → {path}")

#     def load(self, path: str):
#         ckpt = torch.load(path, map_location=self.device)
#         self.score_net.load_state_dict(ckpt["score_net"])
#         self.q1.load_state_dict(ckpt["q1"])
#         self.q2.load_state_dict(ckpt["q2"])
#         self.q1_target.load_state_dict(ckpt["q1_target"])
#         self.q2_target.load_state_dict(ckpt["q2_target"])
#         print(f"  Checkpoint loaded ← {path}")


# # ──────────────────────────────────────────────
# # Entry point
# # ──────────────────────────────────────────────

# # `build_config` is defined in `env.py`.


# def main():
#     cfg    = build_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n[Device] {device}")
#     set_seed(cfg.seed)

#     # Lazy import of gym to keep module importable without it
#     try:
#         import gymnasium as gym
#     except ImportError:
#         import gym  # fallback for older installs

#     env = gym.make(cfg.env)

#     os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

#     trainer = DiffusionQLTrainer(cfg, device)

#     # ── Phase 1: Offline pretraining ─────────────────────────────────
#     trainer.offline_pretrain()
#     trainer.save(cfg.save_path.replace(".pt", "_offline.pt"))

#     # ── Baseline evaluation after offline ────────────────────────────
#     print("\n  [Eval] After offline pretraining:")
#     trainer.evaluate(env, n_episodes=20)

#     # ── Phase 2: Online finetuning ───────────────────────────────────
#     trainer.online_finetune(env)
#     trainer.save(cfg.save_path)

#     # ── Final evaluation ─────────────────────────────────────────────
#     print("\n  [Eval] After online finetuning:")
#     trainer.evaluate(env, n_episodes=20)

#     env.close()


# if __name__ == "__main__":
#     main()


"""
Diffusion Q-Learning for CartPole-v1
=====================================
Combines:
  - QVPO  : Q-guided score-based policy via diffusion (https://arxiv.org/abs/2407.xxxxx)
  - Hy-Q  : Hybrid offline + online Q-learning with importance-weighted mixing
             (https://arxiv.org/abs/2312.xxxxx)

Architecture
------------
  OfflineBuffer      : fixed demonstration data (40k transitions from Q-learning agent)
  OnlineBuffer       : ring buffer filled during environment interaction
  DiffusionPolicy    : DDPM-style denoising network conditioned on (state, Q-guidance)
  QNetwork (x2)      : twin critics for pessimistic value estimation
  HyQMixer           : samples mixed batches with Hy-Q priority weighting
  Trainer            : drives the offline-pretraining → online-finetuning loop

QVPO key ideas implemented
---------------------------
  1. Score network   s_θ(a_t, t, s) approximates ∇_{a_t} log p(a_t | s)
  2. Q-guidance      during reverse diffusion, inject α·∇_a Q(s,a) into the score
  3. Policy loss     combines denoising score matching + in-support BC regularisation

Hy-Q key ideas implemented
---------------------------
  1. Unified replay  single buffer interface over offline + online data
  2. Priority mixing offline samples weighted by |TD-error| from a reference Q*
  3. Smooth ramp     β anneals from 1 (pure offline) → 0 (pure online) over N steps
"""

import os
import copy
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import set_seed, build_config
from buffer import OfflineBuffer, OnlineBuffer, HyQMixer
from models import ScoreNetwork, QNetwork, GaussianDiffusion

# ──────────────────────────────────────────────
# Reproducibility helpers
# ──────────────────────────────────────────────

# `set_seed` and configuration live in `env.py`.


# Replay buffers and model classes were moved to `buffer.py` and `models.py`.
def qvpo_policy_loss(
    score_net,
    diffusion,
    q1,
    q2,
    batch,
    bc_weight=0.5,
):
    """
    L_QVPO = L_DSM + λ_BC · L_BC

    L_DSM : Denoising Score Matching — makes diffusion model fit action distribution
    L_BC  : Behaviour Cloning term  — weighted by advantage to stay in-support of
            high-return demonstrations
    """
    states  = batch["states"]
    actions = batch["actions"]   # (B,) long


    # --------------------------------------------------
    # 1. DSM loss (diffusion objective)
    # --------------------------------------------------
    l_dsm = diffusion.p_losses(score_net, actions, states)

    # --------------------------------------------------
    # 2. Q-weighted BC (QVPO-style)
    # --------------------------------------------------
    with torch.no_grad():
        q_vals = torch.min(q1(states), q2(states))  # (B,2)

        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)
        q_mean = q_vals.mean(dim=1)                                # (B,)
        adv = q_sa - q_mean                                        # (B,)

        weights = torch.exp(adv.clamp(-5, 5))                      # (B,)

    # --------------------------------------------------
    # 3. Continuous BC loss (Gaussian version)
    # --------------------------------------------------
    B = actions.shape[0]
    t_bc = torch.zeros(B, dtype=torch.long, device=states.device)

    # discrete → continuous
    x0 = diffusion.to_continuous(actions).unsqueeze(1)   # (B,1)

    # predict clean action (t = 0)
    x0_pred = score_net(x0, t_bc, states)               # (B,1)

    # weighted MSE
    l_bc = (weights.unsqueeze(1) * (x0_pred - x0) ** 2).mean()

    return l_dsm + bc_weight * l_bc


# ──────────────────────────────────────────────
# Critic update (double Q + target networks)
# ──────────────────────────────────────────────

def critic_loss(
    q1: QNetwork, q2: QNetwork,
    q1_target: QNetwork, q2_target: QNetwork,
    batch: dict,
    gamma: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard twin-Q Bellman loss with target networks."""
    s, a, r, s_next, d = (
        batch["states"], batch["actions"], batch["rewards"],
        batch["next_states"], batch["dones"],
    )

    with torch.no_grad():
        # Pessimistic bootstrap
        q1_next = q1_target(s_next)          # (B, n_actions)
        q2_next = q2_target(s_next)
        q_next  = torch.min(q1_next, q2_next).max(dim=1).values   # (B,)
        target  = r + gamma * (1.0 - d) * q_next                  # (B,)
        target = target.clamp(-100, 100)

    q1_pred = q1(s).gather(1, a.unsqueeze(1)).squeeze(1)   # (B,)
    q2_pred = q2(s).gather(1, a.unsqueeze(1)).squeeze(1)

    td1 = target - q1_pred
    td2 = target - q2_pred

    l1  = F.mse_loss(q1_pred, target)
    l2  = F.mse_loss(q2_pred, target)

    # Return mean TD error for Hy-Q priority update
    td_err = ((td1 + td2) / 2.0).detach().cpu().numpy()
    return (l1 + l2), td_err


# ──────────────────────────────────────────────
# Main Trainer
# ──────────────────────────────────────────────

class DiffusionQLTrainer:
    """
    Orchestrates the full QVPO + Hy-Q training loop.

    Phase 1 — Offline pretraining  (steps 0 … offline_steps)
      Trains critics and diffusion policy purely on demonstration data.
      β = 1.0 (all offline).

    Phase 2 — Online finetuning  (steps offline_steps … total_steps)
      Collects experience with the current diffusion policy.
      Hy-Q mixer anneals β from 1.0 → β_end, blending offline + online data.
      Priorities of offline samples updated each step.
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device):
        self.cfg    = cfg
        self.device = device

        # ── Buffers ──────────────────────────────────────────────────
        self.offline_buf = OfflineBuffer(cfg.demo_path, device)
        self.online_buf  = OnlineBuffer(cfg.online_capacity, cfg.state_dim, device)

        # ── Networks ─────────────────────────────────────────────────
        self.score_net = ScoreNetwork(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim,
            cfg.time_emb_dim, cfg.n_diffusion_steps
        ).to(device)

        self.q1        = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
        self.q2        = QNetwork(cfg.state_dim, cfg.n_actions, cfg.hidden_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters(): p.requires_grad_(False)
        for p in self.q2_target.parameters(): p.requires_grad_(False)

        # ── Diffusion schedule ───────────────────────────────────────
        self.diffusion = GaussianDiffusion(cfg.n_diffusion_steps).to(device)

        # ── Optimisers ───────────────────────────────────────────────
        self.opt_score = optim.Adam(self.score_net.parameters(), lr=cfg.lr_policy)
        self.opt_q     = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q
        )

        # ── Hy-Q Mixer ───────────────────────────────────────────────
        self.mixer = HyQMixer(
            self.offline_buf, self.online_buf,
            beta_start=1.0, beta_end=cfg.hyq_beta_end,
            anneal_steps=cfg.hyq_anneal_steps,
            td_alpha=cfg.hyq_td_alpha,
        )

        # ── Logging ──────────────────────────────────────────────────
        self.log = dict(critic_loss=[], policy_loss=[], episode_return=[])

    # ------------------------------------------------------------------
    def _soft_update(self, tau: float = 0.005):
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.lerp_(p.data, tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.lerp_(p.data, tau)

    # ------------------------------------------------------------------
    def _update_step(self, step: int):
        cfg = self.cfg

        # ── Sample batch from Hy-Q mixer ─────────────────────────────
        batch, off_idx = self.mixer.sample(
            cfg.batch_size, self.q1, self.q2, cfg.gamma
        )

        # ── Critic update ─────────────────────────────────────────────
        c_loss, td_errors = critic_loss(
            self.q1, self.q2, self.q1_target, self.q2_target, batch, cfg.gamma
        )
        self.opt_q.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.opt_q.step()

        # Update Hy-Q priorities with TD errors from the offline slice
        if off_idx is not None:
            n_off = len(off_idx)
            self.mixer.update_priorities(off_idx, td_errors[:n_off])

        # ── Policy (diffusion) update ─────────────────────────────────
        p_loss = qvpo_policy_loss(
            self.score_net, self.diffusion,
            self.q1, self.q2, batch, cfg.bc_weight
        )
        self.opt_score.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
        self.opt_score.step()

        # ── Target network update ─────────────────────────────────────
        self._soft_update(cfg.tau)

        return c_loss.item(), p_loss.item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.diffusion.p_sample(
            self.score_net, s, self.q1, self.q2,
            guidance_scale=self.cfg.guidance_scale
        )
        return action.item()

    # ------------------------------------------------------------------
    def offline_pretrain(self):
        """Phase 1: pure offline training."""
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  Phase 1 — Offline Pretraining ({cfg.offline_steps:,} steps)")
        print(f"{'='*60}")

        for step in range(1, cfg.offline_steps + 1):
            batch = self.offline_buf.sample(cfg.batch_size)
            # During offline phase, bypass Hy-Q mixer and use pure offline batch
            c_loss, td_err = critic_loss(
                self.q1, self.q2, self.q1_target, self.q2_target, batch, cfg.gamma
            )
            self.opt_q.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
            )
            self.opt_q.step()

            p_loss = qvpo_policy_loss(
                self.score_net, self.diffusion,
                self.q1, self.q2, batch, cfg.bc_weight
            )
            self.opt_score.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            self.opt_score.step()
            self._soft_update(cfg.tau)

            if step % cfg.log_interval == 0:
                print(f"  [offline {step:6d}/{cfg.offline_steps}]  "
                      f"critic={c_loss.item():.4f}  policy={p_loss.item():.4f}  "
                      f"β(mixer)={self.mixer.beta:.3f}")

        print("  Offline pretraining done.\n")

    # ------------------------------------------------------------------
    def online_finetune(self, env):
        """Phase 2: online finetuning with Hy-Q mixing."""
        cfg = self.cfg
        print(f"{'='*60}")
        print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
        print(f"{'='*60}")

        state, _ = env.reset(seed=cfg.seed)
        ep_return = 0.0
        ep_count  = 0
        global_step = 0

        for step in range(1, cfg.online_steps + 1):
            # ── Collect one environment step ──────────────────────────
            action    = self.select_action(state)
            ns, reward, term, trunc, _ = env.step(action)
            done      = term or trunc
            self.online_buf.add(state, action, reward, ns, float(done))
            state     = ns
            ep_return += reward
            global_step += 1

            if done:
                self.log["episode_return"].append(ep_return)
                ep_count += 1
                if ep_count % cfg.eval_interval_eps == 0:
                    avg_ret = np.mean(self.log["episode_return"][-20:])
                    print(f"  [online  {step:6d}/{cfg.online_steps}]  "
                          f"ep={ep_count}  avg_return(20)={avg_ret:.1f}  "
                          f"β={self.mixer.beta:.3f}  "
                          f"online_size={self.online_buf.size}")
                ep_return = 0.0
                state, _ = env.reset()

            # ── Wait for minimal online data ──────────────────────────
            if self.online_buf.size < cfg.batch_size:
                continue

            # ── Gradient update ────────────────────────────────────────
            c_loss, p_loss = self._update_step(step)
            self.log["critic_loss"].append(c_loss)
            self.log["policy_loss"].append(p_loss)

        print("  Online finetuning done.\n")

    # ------------------------------------------------------------------
    def evaluate(self, env, n_episodes: int = 10) -> float:
        """Greedy evaluation of current diffusion policy."""
        returns = []
        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_ret   = 0.0
            done     = False
            while not done:
                action = self.select_action(state)
                state, reward, term, trunc, _ = env.step(action)
                ep_ret += reward
                done    = term or trunc
            returns.append(ep_ret)
            print(f"    Episode {ep+1}: return = {ep_ret}")
        mean_ret = float(np.mean(returns))
        print(f"  Evaluation over {n_episodes} episodes: "
              f"mean={mean_ret:.1f}  std={np.std(returns):.1f}")
        return mean_ret

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "score_net":  self.score_net.state_dict(),
            "q1":         self.q1.state_dict(),
            "q2":         self.q2.state_dict(),
            "q1_target":  self.q1_target.state_dict(),
            "q2_target":  self.q2_target.state_dict(),
        }, path)
        print(f"  Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.score_net.load_state_dict(ckpt["score_net"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        print(f"  Checkpoint loaded ← {path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

# `build_config` is defined in `env.py`.


def main():
    cfg    = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    # Lazy import of gym to keep module importable without it
    try:
        import gymnasium as gym
    except ImportError:
        import gym  # fallback for older installs

    env = gym.make(cfg.env)

    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    trainer = DiffusionQLTrainer(cfg, device)

    # ── Phase 1: Offline pretraining ─────────────────────────────────
    trainer.offline_pretrain()
    trainer.save(cfg.save_path.replace(".pt", "_offline.pt"))

    # ── Baseline evaluation after offline ────────────────────────────
    print("\n  [Eval] After offline pretraining:")
    trainer.evaluate(env, n_episodes=20)

    # ── Phase 2: Online finetuning ───────────────────────────────────
    trainer.online_finetune(env)
    trainer.save(cfg.save_path)

    # ── Final evaluation ─────────────────────────────────────────────
    print("\n  [Eval] After online finetuning:")
    trainer.evaluate(env, n_episodes=20)

    env.close()


if __name__ == "__main__":
    main()