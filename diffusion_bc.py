"""
Vanilla Diffusion Behavior Cloning (CartPole-v1)
================================================

This is a baseline diffusion policy trained ONLY on offline data.

Key features:
  - DDPM-style discrete diffusion over actions
  - No Q-learning
  - No Q-guidance
  - No Hy-Q
  - Pure denoising score matching (behavior cloning)

This learns: π(a|s) from dataset only
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------------------------------
# Offline Dataset
# --------------------------------------------------

class OfflineBuffer:
    def __init__(self, path, device):
        raw = np.load(path)
        self.states = torch.tensor(raw["states"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(raw["actions"], dtype=torch.long, device=device)
        self.size = len(self.states)
        print(f"[OfflineBuffer] Loaded {self.size:,} samples")

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.states.device)
        return dict(
            states=self.states[idx],
            actions=self.actions[idx],
        )

# --------------------------------------------------
# Time Embedding
# --------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

# --------------------------------------------------
# Score Network
# --------------------------------------------------

class ScoreNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.n_actions = n_actions

        self.action_emb = nn.Embedding(n_actions + 1, 32)
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(32 + time_emb_dim + state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, a_t, t, state):
        a_emb = self.action_emb(a_t)
        t_emb = self.time_emb(t)
        x = torch.cat([a_emb, t_emb, state], dim=-1)
        return self.net(x)

# --------------------------------------------------
# Diffusion Process
# --------------------------------------------------

class DiscreteDiffusion:
    MASK = 2

    def __init__(self, n_steps):
        self.T = n_steps
        betas = torch.linspace(0.01, 0.5, n_steps)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t):
        alpha_bar_t = self.alpha_bar[t].to(x0.device)
        keep = torch.bernoulli(alpha_bar_t).bool()
        return torch.where(keep, x0, torch.full_like(x0, self.MASK))

    def loss(self, model, actions, states):
        B = actions.shape[0]
        t = torch.randint(0, self.T, (B,), device=actions.device)
        a_t = self.q_sample(actions, t)
        logits = model(a_t, t, states)
        return F.cross_entropy(logits, actions)

    @torch.no_grad()
    def sample(self, model, state):
        B = state.shape[0]
        device = state.device

        x_t = torch.full((B,), self.MASK, dtype=torch.long, device=device)

        for t_val in reversed(range(self.T)):
            t = torch.full((B,), t_val, dtype=torch.long, device=device)
            logits = model(x_t, t, state)
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs, 1).squeeze(1)

        return x_t

# --------------------------------------------------
# Trainer
# --------------------------------------------------

class DiffusionTrainer:
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg

        self.buffer = OfflineBuffer(cfg.demo_path, device)

        self.model = ScoreNetwork(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim, cfg.time_emb_dim
        ).to(device)

        self.diffusion = DiscreteDiffusion(cfg.n_diffusion_steps)

        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train(self):
        print("\n=== Training Vanilla Diffusion BC ===\n")

        for step in range(1, self.cfg.train_steps + 1):
            batch = self.buffer.sample(self.cfg.batch_size)

            loss = self.diffusion.loss(
                self.model, batch["actions"], batch["states"]
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if step % self.cfg.log_interval == 0:
                print(f"[step {step:6d}] loss={loss.item():.4f}")

    @torch.no_grad()
    def select_action(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.diffusion.sample(self.model, s)
        return a.item()

    def evaluate(self, env, episodes=10):
        returns = []
        for ep in range(episodes):
            state, _ = env.reset()
            total = 0
            done = False
            while not done:
                action = self.select_action(state)
                state, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
            returns.append(total)
            print(f"Episode {ep+1}: {total}")

        print(f"Mean={np.mean(returns):.1f} | Std={np.std(returns):.1f}")

# --------------------------------------------------
# Config / Main
# --------------------------------------------------

def build_config():
    p = argparse.ArgumentParser()

    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--demo_path", default="cartpole_demo_data.npz")

    p.add_argument("--state_dim", type=int, default=4)
    p.add_argument("--n_actions", type=int, default=2)

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--time_emb_dim", type=int, default=64)
    p.add_argument("--n_diffusion_steps", type=int, default=200)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--log_interval", type=int, default=1000)

    return p.parse_args()


def main():
    cfg = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    try:
        import gymnasium as gym
    except:
        import gym

    env = gym.make(cfg.env)

    trainer = DiffusionTrainer(cfg, device)

    trainer.train()

    print("\n=== Evaluation ===")
    trainer.evaluate(env, 20)
    env.close()


if __name__ == "__main__":
    main()