import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from env import ContinuousCartPole
from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy, sample_action

device = "cuda" if torch.cuda.is_available() else "cpu"

env    = ContinuousCartPole()
buffer = ReplayBuffer()
buffer.load_offline("cartpole_demo_data.npz")

# ── Networks ──────────────────────────────────────────────────────────────────
q_net    = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())
policy   = DiffusionPolicy().to(device)

q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
p_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

gamma = 0.99
T     = policy.T

# Cosine noise schedule, shape (T+1,): alpha_bars[0]≈1, alpha_bars[T]≈small
steps      = torch.arange(T + 1, dtype=torch.float32)
f          = torch.cos(((steps / T) + 0.008) / 1.008 * (np.pi / 2)) ** 2
alpha_bars = (f / f[0]).to(device)

# The two real actions in the environment, as continuous values
# EVERYTHING that needs max_a or E_a uses ONLY these two points.
# Using linspace(-1,1,N) would evaluate Q at values it was NEVER trained on
# (offline data has only a∈{-1,+1}), causing wild extrapolation that makes
# V(s) garbage and kills the advantage signal entirely.
A_BOTH = torch.tensor([[-1.0], [1.0]], device=device)   # (2,1)


# ── FQI / Bellman update ──────────────────────────────────────────────────────
def train_q(batch_size=256):
    s, a, r, s2 = buffer.sample(batch_size)
    s  = torch.tensor(s,  dtype=torch.float32, device=device)
    a  = torch.tensor(a,  dtype=torch.float32, device=device)
    r  = torch.tensor(r,  dtype=torch.float32, device=device)
    s2 = torch.tensor(s2, dtype=torch.float32, device=device)

    # max_a' Q(s', a') over the TWO real actions only
    with torch.no_grad():
        # s2: (B,4) → (B,2,4); A_BOTH: (2,1) → (B,2,1)
        s2_exp = s2.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        a_exp  = A_BOTH.unsqueeze(0).expand(len(s2), -1, -1).reshape(-1, 1)
        q_next = target_q(s2_exp, a_exp).view(len(s2), 2)
        max_q  = q_next.max(dim=1, keepdim=True).values   # (B,1)

    target = r + gamma * max_q
    loss   = ((q_net(s, a) - target) ** 2).mean()

    q_opt.zero_grad(); loss.backward(); q_opt.step()

    tau = 0.005
    for p, tp in zip(q_net.parameters(), target_q.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ── QVPO advantage weights ────────────────────────────────────────────────────
def compute_weights(s, a):
    with torch.no_grad():
        q = q_net(s, a)                                   # (B,1)

        # V(s) = E_{a~π}[Q(s,a)] estimated over the TWO in-distribution actions
        s_exp  = s.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        a_exp  = A_BOTH.unsqueeze(0).expand(len(s), -1, -1).reshape(-1, 1)
        q_both = q_net(s_exp, a_exp).view(len(s), 2)
        v      = q_both.mean(dim=1, keepdim=True)         # (B,1)

    A       = q - v                                       # eq. 6
    weights = torch.clamp(A, min=0.0)                    # eq. 5

    nz = weights[weights > 0]
    if nz.numel() > 0:
        weights = weights / (nz.mean() + 1e-8)
    return weights.detach()


# ── Diffusion policy update ───────────────────────────────────────────────────
def train_policy(batch_size=256):
    s, a, _, _ = buffer.sample(batch_size)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.float32, device=device)

    weights = compute_weights(s, a)

    t_idx   = torch.randint(1, T + 1, (len(s),), device=device)
    t_norm  = t_idx.float().unsqueeze(-1) / T

    noise   = torch.randn_like(a)
    ab      = alpha_bars[t_idx].unsqueeze(-1)
    a_noisy = torch.sqrt(ab) * a + torch.sqrt(1.0 - ab) * noise

    pred = policy(s, a_noisy, t_norm)
    loss = ((noise - pred) ** 2 * weights).mean()

    p_opt.zero_grad(); loss.backward(); p_opt.step()


# ── Action selection ──────────────────────────────────────────────────────────
def select_action(state):
    with torch.no_grad():
        samples = sample_action(policy, state, alpha_bars, T)   # (K,1)
        # Snap to the two real actions before Q evaluation.
        # The diffusion model outputs a continuous sign preference;
        # evaluating Q at raw intermediate values causes OOD extrapolation.
        snapped = torch.sign(samples)
        snapped[snapped == 0] = 1.0                             # break ties

        s_rep  = torch.tensor(state, dtype=torch.float32, device=device)
        s_rep  = s_rep.unsqueeze(0).expand(len(snapped), -1)
        q_vals = q_net(s_rep, snapped)
        best   = q_vals.argmax()
    return samples[best].item()    # return the original continuous value for env


# ── Offline Q pre-training ────────────────────────────────────────────────────
print("Pre-training Q on offline buffer …")
for _ in range(5000):
    train_q()
print("Pre-training done.")


# ── Main online training loop ─────────────────────────────────────────────────
episode_rewards = []

for ep in range(1000):
    s            = env.reset()
    total_reward = 0
    eps          = max(0.05, 0.30 * (0.994 ** ep))

    for step in range(200):
        a           = np.random.uniform(-1, 1) if np.random.rand() < eps else select_action(s)
        s2, r, done = env.step(a)

        buffer.add(s, a, r, s2)
        s = s2
        total_reward += r

        train_q()
        if step % 2 == 0:
            train_policy()

        if done:
            break

    episode_rewards.append(total_reward)
    if ep % 50 == 0:
        recent = np.mean(episode_rewards[-20:]) if ep >= 20 else total_reward
        print(f"Episode {ep:4d} | Reward: {total_reward:6.1f} | "
              f"20-ep avg: {recent:6.1f} | eps: {eps:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 4))
plt.plot(episode_rewards, alpha=0.35, color="steelblue", label="per-episode")
w      = 20
smooth = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
plt.plot(range(w - 1, len(episode_rewards)), smooth,
         color="steelblue", linewidth=2, label=f"{w}-ep moving avg")
plt.xlabel("Episode"); plt.ylabel("Reward")
plt.title("DPRL (QVPO + Diffusion Policy) on CartPole")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("reward_curve.png"); plt.close()
print("Saved plot to reward_curve.png")
