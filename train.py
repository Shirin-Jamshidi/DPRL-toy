# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import torch.nn.functional as F

# from env import ContinuousCartPole
# from buffer import ReplayBuffer
# from models import QNetwork, DiffusionPolicy, sample_action

# device = "cuda" if torch.cuda.is_available() else "cpu"

# env    = ContinuousCartPole()
# buffer = ReplayBuffer()

# buffer.load_offline("cartpole_demo_data.npz")

# # ── Networks ──────────────────────────────────────────────────────────────────
# q_net    = QNetwork().to(device)
# target_q = QNetwork().to(device)
# target_q.load_state_dict(q_net.state_dict())

# policy = DiffusionPolicy().to(device)

# # Lower Q learning rate → more stable Bellman targets
# q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
# p_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

# gamma = 0.99
# T     = policy.T

# # Cosine noise schedule (smoother than linear for small T=10)
# # alpha_bars[t] = ᾱ_t; index 0 → no noise, index T → full noise
# steps      = torch.arange(T + 1, dtype=torch.float32)
# f          = torch.cos(((steps / T) + 0.008) / 1.008 * (np.pi / 2)) ** 2
# alpha_bars = (f / f[0]).to(device)   # shape (T+1,)


# # ── Bellman / FQI update ──────────────────────────────────────────────────────
# def train_q(batch_size=256):
#     s, a, r, s2 = buffer.sample(batch_size)
#     s  = torch.tensor(s,  dtype=torch.float32, device=device)
#     a  = torch.tensor(a,  dtype=torch.float32, device=device)
#     r  = torch.tensor(r,  dtype=torch.float32, device=device)
#     s2 = torch.tensor(s2, dtype=torch.float32, device=device)

#     a_cand = torch.tensor([[0.0], [1.0]], device=device)

#     with torch.no_grad():
#         s2_flat = s2.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
#         a_flat  = a_cand.unsqueeze(0).expand(len(s2), -1, -1).reshape(-1, 1)
#         q_next  = target_q(s2_flat, a_flat).view(len(s2), 2)
#         max_q   = q_next.max(dim=1, keepdim=True).values    # (B,1)

#     target = r + gamma * max_q
#     loss   = ((q_net(s, a) - target) ** 2).mean()

#     q_opt.zero_grad(); loss.backward(); q_opt.step()

#     tau = 0.005
#     for p, tp in zip(q_net.parameters(), target_q.parameters()):
#         tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# # ── QVPO advantage weights ────────────────────────────────────────────────────
# def compute_weights(s, a):
#     # All inside no_grad: weights are treated as fixed coefficients
#     with torch.no_grad():
#         q = q_net(s, a)                              # (B,1)

#         a_samp = torch.tensor([[0.0], [1.0]], device=device)
#         s_flat = s.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
#         a_flat = a_samp.expand(len(s), -1, -1).reshape(-1, 1)
#         q_all  = q_net(s_flat, a_flat).view(len(s), 2)

#         # V(s) = E_a[Q(s,a)] per eq. 6 — NOT max
#         v = q_all.mean(dim=1, keepdim=True)

#     A       = q - v                                  # (B,1)
#     weights = torch.clamp(A, min=0.0)               # eq. 5

#     # Rescale non-zero weights to unit mean so the loss magnitude is stable
#     # even when most advantages are small. This does NOT change their ordering.
#     nz = weights[weights > 0]
#     if nz.numel() > 0:
#         weights = weights / (nz.mean() + 1e-8)
#     return weights.detach()


# # ── Diffusion policy update ───────────────────────────────────────────────────
# def train_policy(batch_size=256):
#     s, a, _, _ = buffer.sample(batch_size)
#     s = torch.tensor(s, dtype=torch.float32, device=device)
#     a = torch.tensor(a, dtype=torch.float32, device=device)

#     weights = compute_weights(s, a)                  # (B,1)

#     a_idx = a.long().view(-1)
#     a = F.one_hot(a_idx, num_classes=2).float()

#     # Sample a random diffusion timestep t ∈ [1, T]
#     t_idx   = torch.randint(1, T + 1, (len(s),), device=device)
#     # Normalise t to [0,1] so the network doesn't need to know T
#     t_float = t_idx.float().unsqueeze(-1) / T

#     noise   = torch.randn_like(a)
#     ab      = alpha_bars[t_idx].unsqueeze(-1)        # ᾱ_t  shape (B,1)
#     a_noisy = torch.sqrt(ab) * a + torch.sqrt(1.0 - ab) * noise

#     pred = policy(s, a_noisy, t_float)
#     loss = ((noise - pred) ** 2 * weights).mean()

#     p_opt.zero_grad(); loss.backward(); p_opt.step()


# # ── Action selection ──────────────────────────────────────────────────────────
# def select_action(state):
#     with torch.no_grad():
#         samples = sample_action(policy, state, alpha_bars, T)  # (K,1)
#         s_rep   = torch.tensor(state, dtype=torch.float32, device=device)
#         s_rep   = s_rep.unsqueeze(0).expand(len(samples), -1)
#         q_vals  = q_net(s_rep, samples)
#         best    = q_vals.argmax()
#     return int(samples[best].item())


# # ── Offline pre-training of Q ─────────────────────────────────────────────────
# # Without this, Q is random when online episodes start, the advantage is noise,
# # and the policy never gets a meaningful gradient.
# print("Pre-training Q on offline buffer …")
# for i in range(3000):
#     train_q()
# print("Pre-training done.")


# # ── Main online training loop ─────────────────────────────────────────────────
# episode_rewards = []

# for ep in range(20000):
#     s            = env.reset()
#     total_reward = 0
#     # Decay exploration from 30% → 5% over training
#     eps = max(0.05, 0.30 * (0.994 ** ep))

#     for step in range(500):
#         a  = np.random.randint(0, 2) if np.random.rand() < eps else select_action(s)
#         s2, r, done = env.step(a)

#         buffer.add(s, a, r, s2)
#         s = s2
#         total_reward += r

#         train_q()
#         if step % 2 == 0:
#             train_policy()

#         if done:
#             break

#     episode_rewards.append(total_reward)
#     if ep % 50 == 0:
#         recent = np.mean(episode_rewards[-20:]) if ep >= 20 else total_reward
#         print(f"Episode {ep:4d} | Reward: {total_reward:6.1f} | "
#               f"20-ep avg: {recent:6.1f} | eps: {eps:.3f}")


# # ── Plot ──────────────────────────────────────────────────────────────────────
# plt.figure(figsize=(9, 4))
# plt.plot(episode_rewards, alpha=0.35, color="steelblue", label="per-episode")
# w = 20
# smooth = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
# plt.plot(range(w - 1, len(episode_rewards)), smooth,
#          color="steelblue", linewidth=2, label=f"{w}-ep moving avg")
# plt.xlabel("Episode"); plt.ylabel("Reward")
# plt.title("DPRL (QVPO + Diffusion Policy) on CartPole")
# plt.legend(); plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("reward_curve.png"); plt.close()
# print("Saved plot to reward_curve.png")




















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
import math
import random
import argparse
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ──────────────────────────────────────────────
# Reproducibility helpers
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Replay Buffers
# ──────────────────────────────────────────────

class OfflineBuffer:
    """Wraps the static demonstration dataset."""

    def __init__(self, path: str, device: torch.device):
        raw = np.load(path)
        self.states      = torch.tensor(raw["states"],      dtype=torch.float32, device=device)
        self.actions      = torch.tensor(raw["actions"],     dtype=torch.long,    device=device)
        self.rewards      = torch.tensor(raw["rewards"],     dtype=torch.float32, device=device)
        self.next_states  = torch.tensor(raw["next_states"], dtype=torch.float32, device=device)
        # No 'dones' in file — all transitions are mid-episode (reward=1 throughout)
        if "dones" in raw:
            self.dones = torch.tensor(raw["dones"], dtype=torch.float32, device=device)
        else:
            # fallback (just in case old dataset is used)
            self.dones = (self.rewards < 1.0).float()
        self.size         = len(self.states)
        print(f"[OfflineBuffer] Loaded {self.size:,} transitions from {path}")

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        return dict(
            states     = self.states[idx],
            actions    = self.actions[idx],
            rewards    = self.rewards[idx],
            next_states= self.next_states[idx],
            dones      = self.dones[idx],
        )


class OnlineBuffer:
    """Ring buffer for online experience."""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity  = capacity
        self.device    = device
        self.ptr       = 0
        self.full      = False

        self.states     = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions    = torch.zeros((capacity,),           dtype=torch.long,    device=device)
        self.rewards    = torch.zeros((capacity,),           dtype=torch.float32, device=device)
        self.next_states= torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones      = torch.zeros((capacity,),           dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i]      = torch.tensor(state,      dtype=torch.float32, device=self.device)
        self.actions[i]     = torch.tensor(action,     dtype=torch.long,    device=self.device)
        self.rewards[i]     = torch.tensor(reward,     dtype=torch.float32, device=self.device)
        self.next_states[i] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[i]       = torch.tensor(done,       dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(
            states     = self.states[idx],
            actions    = self.actions[idx],
            rewards    = self.rewards[idx],
            next_states= self.next_states[idx],
            dones      = self.dones[idx],
        )


# ──────────────────────────────────────────────
# Hy-Q Mixer
# ──────────────────────────────────────────────

class HyQMixer:
    """
    Hy-Q hybrid batch sampler.

    At each step draws a batch that is a weighted mixture of offline and
    online data.  The offline ratio β decays linearly from β_start → β_end
    over `anneal_steps` gradient steps, mirroring Algorithm 1 in the Hy-Q
    paper.  Within the offline portion, samples are drawn with probability
    proportional to their |TD-error| computed against the current critics
    (importance weighting for high-value demonstrations).
    """

    def __init__(
        self,
        offline_buf: OfflineBuffer,
        online_buf:  OnlineBuffer,
        beta_start:  float = 1.0,
        beta_end:    float = 0.25,
        anneal_steps: int  = 50_000,
        td_alpha:    float = 0.6,   # exponent for priority weighting
    ):
        self.offline     = offline_buf
        self.online      = online_buf
        self.beta_start  = beta_start
        self.beta_end    = beta_end
        self.anneal_steps= anneal_steps
        self.td_alpha    = td_alpha
        self._step       = 0

        # Uniform priorities at start; updated lazily
        self._offline_priorities = np.ones(offline_buf.size, dtype=np.float32)

    # ------------------------------------------------------------------
    @property
    def beta(self) -> float:
        """Current offline mixing ratio."""
        frac = min(self._step / max(self.anneal_steps, 1), 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ------------------------------------------------------------------
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Called by the trainer after each critic update."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.td_alpha
        self._offline_priorities[indices] = priorities

    # ------------------------------------------------------------------
    def sample(self, batch_size: int, q1: nn.Module, q2: nn.Module,
               gamma: float) -> Tuple[dict, Optional[np.ndarray]]:
        """
        Returns a mixed batch and the offline indices used (or None).
        The batch always has exactly `batch_size` transitions.
        """
        self._step += 1
        beta = self.beta

        # How many offline vs online transitions to draw
        n_offline = int(round(beta * batch_size))
        n_online  = batch_size - n_offline

        batches = []
        offline_idx = None

        # ── Offline portion (priority-weighted) ──────────────────────
        if n_offline > 0:
            probs  = self._offline_priorities / self._offline_priorities.sum()
            offline_idx = np.random.choice(self.offline.size, size=n_offline,
                                            replace=False, p=probs)
            device = self.offline.states.device
            idx_t  = torch.tensor(offline_idx, device=device)
            batch_off = dict(
                states     = self.offline.states[idx_t],
                actions    = self.offline.actions[idx_t],
                rewards    = self.offline.rewards[idx_t],
                next_states= self.offline.next_states[idx_t],
                dones      = self.offline.dones[idx_t],
            )
            batches.append(batch_off)

        # ── Online portion ────────────────────────────────────────────
        if n_online > 0 and self.online.size >= n_online:
            batches.append(self.online.sample(n_online))
        elif n_online > 0:
            # Not enough online data yet — pad with offline
            extra = self.offline.sample(n_online)
            batches.append(extra)

        # ── Concatenate ───────────────────────────────────────────────
        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        return merged, offline_idx


# ──────────────────────────────────────────────
# Sinusoidal time embedding (standard DDPM)
# ──────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long or float in [0, T-1]"""
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


# ──────────────────────────────────────────────
# Score / Denoising Network  (QVPO)
# ──────────────────────────────────────────────

class ScoreNetwork(nn.Module):
    """
    Predicts the denoising score  s_θ(a_t, t, s) ≈ ∇_{a_t} log p(a_t | s).

    For discrete CartPole (2 actions) we parameterise the score as an
    embedding over action indices rather than a continuous vector.
    The network outputs a scalar logit per action — during reverse diffusion
    we use a Gumbel-softmax relaxation so gradients flow through.

    Input:  (noisy_action_emb, time_emb, state)
    Output: action logits of shape (B, n_actions)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256,
                 time_emb_dim: int = 64, n_diffusion_steps: int = 20):
        super().__init__()
        self.n_actions        = n_actions
        self.n_diffusion_steps= n_diffusion_steps
        action_emb_dim        = 32  # learned embedding per discrete action

        self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = action_emb_dim + time_emb_dim + state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, n_actions),   # raw logits
        )

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        noisy_action : (B,)  long — discrete noisy action token
        t            : (B,)  long — diffusion timestep in [0, T-1]
        state        : (B, state_dim)
        returns      : (B, n_actions)  score logits
        """
        a_emb = self.action_emb(noisy_action)     # (B, action_emb_dim)
        t_emb = self.time_emb(t)                   # (B, time_emb_dim)
        x     = torch.cat([a_emb, t_emb, state], dim=-1)
        return self.net(x)


# ──────────────────────────────────────────────
# Twin Q-Networks
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: (B, n_actions)"""
        return self.net(state)


# ──────────────────────────────────────────────
# DDPM noise schedule (discrete-action variant)
# ──────────────────────────────────────────────

class DiscreteGaussianDiffusion:
    """
    Implements a simple masking / discrete diffusion schedule for action tokens.

    We use the "absorbing state" formulation common for discrete diffusion:
      - Forward: at each step, independently mask each position with prob β_t
      - Reverse: learn to unmask via the score network

    For CartPole's binary action space, the absorbing/mask token is index 2.
    """

    MASK_TOKEN = 2  # special token outside {0,1}

    def __init__(self, n_steps: int = 20):
        self.T = n_steps
        # Linear schedule for mask probability
        betas = torch.linspace(0.01, 0.5, n_steps)
        # Cumulative product: probability of NOT being masked up to step t
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)
        self.betas     = betas

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward process: corrupt x0 at timestep t.
        x0 : (B,) long ∈ {0,1}
        t  : (B,) long ∈ [0, T-1]
        returns noisy actions (B,) — some entries replaced by MASK_TOKEN
        """
        alpha_bar_t = self.alpha_bar[t].to(x0.device)      # (B,)
        keep_mask   = torch.bernoulli(alpha_bar_t).bool()   # True → keep original
        noisy       = torch.where(keep_mask, x0, torch.full_like(x0, self.MASK_TOKEN))
        return noisy

    def p_losses(self, score_net: ScoreNetwork, x0: torch.Tensor,
                 state: torch.Tensor) -> torch.Tensor:
        """
        Denoising score matching loss (cross-entropy on clean action prediction).
        """
        B      = x0.shape[0]
        t      = torch.randint(0, self.T, (B,), device=x0.device)
        noisy  = self.q_sample(x0, t)
        logits = score_net(noisy, t, state)          # (B, n_actions)
        # Predict clean action class — standard DDPM "x0 parameterisation"
        return F.cross_entropy(logits[:, :score_net.n_actions], x0)

    @torch.no_grad()
    def p_sample(self, score_net: ScoreNetwork, state: torch.Tensor,
                 q1: QNetwork, q2: QNetwork,
                 guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Reverse diffusion sampling with Q-guidance (QVPO Algorithm 1).
        Starts from fully masked and iteratively denoises.

        Returns: (B,) long — sampled discrete actions
        """
        B      = state.shape[0]
        device = state.device
        # Start from fully masked
        x_t = torch.full((B,), self.MASK_TOKEN, dtype=torch.long, device=device)

        for t_val in reversed(range(self.T)):
            t = torch.full((B,), t_val, dtype=torch.long, device=device)
            logits = score_net(x_t, t, state)        # (B, n_actions)

            # ── Q-guidance  (QVPO §3.2) ──────────────────────────────
            # Q-values for valid actions; add scaled Q-advantage to logits
            if guidance_scale > 0.0:
                q_vals  = torch.min(q1(state), q2(state))   # (B, n_actions)
                q_adv   = q_vals - q_vals.mean(dim=1, keepdim=True)
                logits  = logits[:, :2] + guidance_scale * q_adv
            else:
                logits  = logits[:, :2]

            # Gumbel-softmax → hard sample
            probs = F.softmax(logits, dim=-1)
            x_t   = torch.multinomial(probs, 1).squeeze(1)  # (B,)

        return x_t


# ──────────────────────────────────────────────
# QVPO Policy loss  (QVPO §3.3)
# ──────────────────────────────────────────────

def qvpo_policy_loss(
    score_net:  ScoreNetwork,
    diffusion:  DiscreteGaussianDiffusion,
    q1:         QNetwork,
    q2:         QNetwork,
    batch:      dict,
    bc_weight:  float = 0.5,
) -> torch.Tensor:
    """
    L_QVPO = L_DSM + λ_BC · L_BC

    L_DSM : Denoising Score Matching — makes diffusion model fit action distribution
    L_BC  : Behaviour Cloning term  — weighted by advantage to stay in-support of
            high-return demonstrations
    """
    states  = batch["states"]
    actions = batch["actions"]   # (B,) long

    # ── 1. DSM loss ──────────────────────────────────────────────────
    l_dsm = diffusion.p_losses(score_net, actions, states)

    # ── 2. BC loss weighted by Q-advantage (QVPO eq. 5) ─────────────
    with torch.no_grad():
        q_vals  = torch.min(q1(states), q2(states))       # (B, n_actions)
        q_sa    = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        q_mean  = q_vals.mean(dim=1)                       # (B,)
        adv     = q_sa - q_mean                            # (B,)
        weights = torch.exp(adv.clamp(-5, 5))              # soft-max advantage weighting

    # BC: push policy toward demonstrated actions, scaled by advantage
    B      = states.shape[0]
    t_bc   = torch.zeros(B, dtype=torch.long, device=states.device)  # t=0 → clean
    logits = score_net(actions, t_bc, states)[:, :2]                 # (B, 2)
    l_bc   = (weights * F.cross_entropy(logits, actions, reduction="none")).mean()

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
        self.diffusion = DiscreteGaussianDiffusion(cfg.n_diffusion_steps)

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

def build_config() -> argparse.Namespace:
    p = argparse.ArgumentParser("Diffusion Q-Learning (QVPO + Hy-Q) — CartPole-v1")

    # Environment
    p.add_argument("--env",          default="CartPole-v1")
    p.add_argument("--state_dim",    type=int,   default=4)
    p.add_argument("--n_actions",    type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)

    # Demo data
    p.add_argument("--demo_path",    default="cartpole_demo_data.npz")

    # Network architecture
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--time_emb_dim", type=int,   default=64)
    p.add_argument("--n_diffusion_steps", type=int, default=20)

    # Training
    p.add_argument("--offline_steps",     type=int,   default=20_000)
    p.add_argument("--online_steps",      type=int,   default=100_000)
    p.add_argument("--batch_size",        type=int,   default=256)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--tau",               type=float, default=0.005)
    p.add_argument("--lr_q",              type=float, default=3e-4)
    p.add_argument("--lr_policy",         type=float, default=3e-4)
    p.add_argument("--bc_weight",         type=float, default=0.5)
    p.add_argument("--guidance_scale",    type=float, default=1.0)

    # Hy-Q
    p.add_argument("--hyq_beta_end",      type=float, default=0.25)
    p.add_argument("--hyq_anneal_steps",  type=int,   default=50_000)
    p.add_argument("--hyq_td_alpha",      type=float, default=0.6)
    p.add_argument("--online_capacity",   type=int,   default=200_000)

    # Logging
    p.add_argument("--log_interval",         type=int, default=1_000)
    p.add_argument("--eval_interval_eps",    type=int, default=10)
    p.add_argument("--save_path",            default="checkpoints/diffusion_ql.pt")

    return p.parse_args()


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
