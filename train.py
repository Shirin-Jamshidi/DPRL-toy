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

# ── Offline data ──────────────────────────────────────────────────────────────
# If the demo file doesn't exist, generate one from a random agent.
# NOTE: the quality of offline data matters a lot. A random agent produces
# episodes of ~20 steps, so every bootstrapped Q-target will be low and
# similar, making A(s,a) ≈ 0 everywhere → no diffusion policy gradient.
# We therefore do an offline pre-training phase AFTER loading the data.
if not os.path.exists("cartpole_demo_data.npz"):
    import gymnasium as gym
    _env = gym.make("CartPole-v1")
    ss, aa, rr, ss2 = [], [], [], []
    for _ in range(200):
        obs, _ = _env.reset()
        for _ in range(500):
            a = _env.action_space.sample()
            obs2, r, term, trunc, _ = _env.step(a)
            ss.append(obs); aa.append(a); rr.append(r); ss2.append(obs2)
            obs = obs2
            if term or trunc:
                break
    np.savez("cartpole_demo_data.npz",
             states=np.array(ss), actions=np.array(aa),
             rewards=np.array(rr), next_states=np.array(ss2))
    print(f"Generated {len(ss)} offline transitions.")

buffer.load_offline("cartpole_demo_data.npz")

# ── Networks ──────────────────────────────────────────────────────────────────
q_net    = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy().to(device)

# Lower Q learning rate → more stable Bellman targets
q_opt = torch.optim.Adam(q_net.parameters(), lr=3e-4)
p_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

gamma = 0.99
T     = policy.T

# Cosine noise schedule (smoother than linear for small T=10)
# alpha_bars[t] = ᾱ_t; index 0 → no noise, index T → full noise
steps      = torch.arange(T + 1, dtype=torch.float32)
f          = torch.cos(((steps / T) + 0.008) / 1.008 * (np.pi / 2)) ** 2
alpha_bars = (f / f[0]).to(device)   # shape (T+1,)


# ── Bellman / FQI update ──────────────────────────────────────────────────────
def train_q(batch_size=256):
    s, a, r, s2 = buffer.sample(batch_size)
    s  = torch.tensor(s,  dtype=torch.float32, device=device)
    a  = torch.tensor(a,  dtype=torch.float32, device=device)
    r  = torch.tensor(r,  dtype=torch.float32, device=device)
    s2 = torch.tensor(s2, dtype=torch.float32, device=device)

    N      = 51
    a_cand = torch.linspace(-1, 1, N, device=device).view(1, N, 1)

    with torch.no_grad():
        s2_flat = s2.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
        a_flat  = a_cand.expand(len(s2), -1, -1).reshape(-1, 1)
        q_next  = target_q(s2_flat, a_flat).view(len(s2), N)
        max_q   = q_next.max(dim=1, keepdim=True).values    # (B,1)

    target = r + gamma * max_q
    loss   = ((q_net(s, a) - target) ** 2).mean()

    q_opt.zero_grad(); loss.backward(); q_opt.step()

    tau = 0.005
    for p, tp in zip(q_net.parameters(), target_q.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ── QVPO advantage weights ────────────────────────────────────────────────────
def compute_weights(s, a):
    # All inside no_grad: weights are treated as fixed coefficients
    with torch.no_grad():
        q = q_net(s, a)                              # (B,1)

        N      = 51
        a_samp = torch.linspace(-1, 1, N, device=device).view(1, N, 1)
        s_flat = s.unsqueeze(1).expand(-1, N, -1).reshape(-1, 4)
        a_flat = a_samp.expand(len(s), -1, -1).reshape(-1, 1)
        q_all  = q_net(s_flat, a_flat).view(len(s), N)

        # V(s) = E_a[Q(s,a)] per eq. 6 — NOT max
        v = q_all.mean(dim=1, keepdim=True)

    A       = q - v                                  # (B,1)
    weights = torch.clamp(A, min=0.0)               # eq. 5

    # Rescale non-zero weights to unit mean so the loss magnitude is stable
    # even when most advantages are small. This does NOT change their ordering.
    nz = weights[weights > 0]
    if nz.numel() > 0:
        weights = weights / (nz.mean() + 1e-8)
    return weights.detach()


# ── Diffusion policy update ───────────────────────────────────────────────────
def train_policy(batch_size=256):
    s, a, _, _ = buffer.sample(batch_size)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.float32, device=device)

    weights = compute_weights(s, a)                  # (B,1)

    # Sample a random diffusion timestep t ∈ [1, T]
    t_idx   = torch.randint(1, T + 1, (len(s),), device=device)
    # Normalise t to [0,1] so the network doesn't need to know T
    t_float = t_idx.float().unsqueeze(-1) / T

    noise   = torch.randn_like(a)
    ab      = alpha_bars[t_idx].unsqueeze(-1)        # ᾱ_t  shape (B,1)
    a_noisy = torch.sqrt(ab) * a + torch.sqrt(1.0 - ab) * noise

    pred = policy(s, a_noisy, t_float)
    loss = ((noise - pred) ** 2 * weights).mean()

    p_opt.zero_grad(); loss.backward(); p_opt.step()


# ── Action selection ──────────────────────────────────────────────────────────
def select_action(state):
    with torch.no_grad():
        samples = sample_action(policy, state, alpha_bars, T)  # (K,1)
        s_rep   = torch.tensor(state, dtype=torch.float32, device=device)
        s_rep   = s_rep.unsqueeze(0).expand(len(samples), -1)
        q_vals  = q_net(s_rep, samples)
        best    = q_vals.argmax()
    return samples[best].item()


# ── Offline pre-training of Q ─────────────────────────────────────────────────
# Without this, Q is random when online episodes start, the advantage is noise,
# and the policy never gets a meaningful gradient.
print("Pre-training Q on offline buffer …")
for i in range(3000):
    train_q()
print("Pre-training done.")


# ── Main online training loop ─────────────────────────────────────────────────
episode_rewards = []

for ep in range(1000):
    s            = env.reset()
    total_reward = 0
    # Decay exploration from 30% → 5% over training
    eps = max(0.05, 0.30 * (0.994 ** ep))

    for step in range(200):
        a  = np.random.uniform(-1, 1) if np.random.rand() < eps else select_action(s)
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
w = 20
smooth = np.convolve(episode_rewards, np.ones(w) / w, mode="valid")
plt.plot(range(w - 1, len(episode_rewards)), smooth,
         color="steelblue", linewidth=2, label=f"{w}-ep moving avg")
plt.xlabel("Episode"); plt.ylabel("Reward")
plt.title("DPRL (QVPO + Diffusion Policy) on CartPole")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reward_curve.png"); plt.close()
print("Saved plot to reward_curve.png")
