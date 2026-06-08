import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy

# =======================
# ✅ Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# ✅ Environment
# =======================
env = gym.make("CartPole-v1")
state_dim = 4

buffer = ReplayBuffer()

# =======================
# ✅ Load offline dataset
# =======================
def load_offline_data(buffer, filename):
    data = np.load(filename)

    for i in range(len(data["states"])):
        buffer.add(
            data["states"][i],
            float(data["actions"][i]),
            data["rewards"][i],
            data["next_states"][i]
        )

    print("Loaded", len(data["states"]), "samples")

load_offline_data(buffer, "cartpole_demo_data.npz")


# =======================
# ✅ Models
# =======================
q_net = QNetwork(input_dim=state_dim + 1).to(device)
target_q = QNetwork(input_dim=state_dim + 1).to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy(input_dim=state_dim + 2).to(device)

q_optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99

# =======================
# ✅ Diffusion schedule
# =======================
T = 10
betas = torch.linspace(0.0001, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


# =======================
# ✅ Sample action (SAFE)
# =======================
def sample_action(state):
    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    a = torch.randn((1, 1)).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([[t / T]], dtype=torch.float32).to(device)

        eps_pred = policy(s, a, t_tensor)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        noise = torch.randn_like(a) if t > 0 else 0

        a = (1 / torch.sqrt(alpha_t)) * (
            a - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        ) + torch.sqrt(betas[t]) * noise

    a_val = torch.tanh(a).item()

    # ✅ probabilistic mapping (prevents collapse)
    prob = (a_val + 1) / 2
    return np.random.choice([0, 1], p=[1 - prob, prob])


# =======================
# ✅ Q-learning (stable)
# =======================
def train_q(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, a, r, s_next = buffer.sample(batch_size)

    s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
    a = torch.tensor(np.array(a), dtype=torch.float32).unsqueeze(1).to(device)
    r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).to(device)
    s_next = torch.tensor(np.array(s_next), dtype=torch.float32).to(device)

    n_samples = 10

    s_next_rep = s_next.unsqueeze(0).repeat(n_samples, 1, 1)
    s_next_flat = s_next_rep.view(-1, state_dim)

    a_noise = torch.randn((n_samples * batch_size, 1)).to(device)

    t = torch.randint(0, T, (n_samples * batch_size,), device=device)
    t_norm = t.float().unsqueeze(1) / T

    eps_pred = policy(s_next_flat, a_noise, t_norm)

    alpha_t = alphas[t].unsqueeze(1)
    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    a_gen = (1 / torch.sqrt(alpha_t)) * (
        a_noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
    )

    a_gen = torch.tanh(a_gen)
    a_gen = (a_gen > 0).float()

    q_vals = target_q(s_next_flat, a_gen)
    q_vals = q_vals.view(n_samples, batch_size, 1)

    max_q = q_vals.max(dim=0)[0]

    # ✅ normalized reward (stability)
    target = (r / 200.0) + gamma * max_q

    q = q_net(s, a)

    loss = ((q - target) ** 2).mean()

    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()


# =======================
# ✅ Policy training (FIXED)
# =======================
def train_policy(batch_size=64, bc_only=False):
    if len(buffer) < batch_size:
        return

    s, a, _, _ = buffer.sample(batch_size)

    s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
    a = torch.tensor(np.array(a), dtype=torch.float32).unsqueeze(1).to(device)

    t = torch.randint(0, T, (batch_size,), device=device)
    t_norm = t.float().unsqueeze(1) / T

    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    eps = torch.randn_like(a)
    noisy_a = torch.sqrt(alpha_bar_t) * a + torch.sqrt(1 - alpha_bar_t) * eps

    eps_pred = policy(s, noisy_a, t_norm)

    # ✅ STRONG behavior cloning
    bc_loss = ((eps - eps_pred) ** 2).mean()

    if bc_only:
        loss = bc_loss
    else:
        q_val = q_net(s, a)
        baseline = q_val.mean().detach()

        weights = torch.clamp(q_val - baseline, min=0)
        adv_loss = (weights * (eps - eps_pred) ** 2).mean()

        # ✅ BC DOMINATES
        loss = bc_loss + 0.01 * adv_loss

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()


# =======================
# ✅ PRETRAINING
# =======================
print("Pretraining Q...")
for _ in range(2000):
    train_q()

print("Pretraining policy (BC)...")
for _ in range(2000):
    train_policy(bc_only=True)


# =======================
# ✅ TRAINING LOOP
# =======================
reward_history = []

for ep in range(500):
    s, _ = env.reset()
    total_reward = 0

    for _ in range(200):

        # ✅ early phase uses safe exploration
        if ep < 100:
            a = np.random.choice([0, 1])
        else:
            candidates = [sample_action(s) for _ in range(20)]

            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

            q_vals = []
            for a_cand in candidates:
                a_tensor = torch.tensor([[a_cand]], dtype=torch.float32).to(device)
                q_vals.append(q_net(s_tensor, a_tensor).item())

            # ✅ epsilon-greedy on top
            if np.random.rand() < 0.2:
                a = np.random.choice([0, 1])
            else:
                a = candidates[np.argmax(q_vals)]

        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc

        # ✅ DELAY online data (CRITICAL)
        if ep > 200:
            buffer.add(s, a, r, s_next)

        train_q()
        train_policy()

        s = s_next
        total_reward += r

        if done:
            break

    # ✅ soft update
    for p, tp in zip(q_net.parameters(), target_q.parameters()):
        tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

    reward_history.append(total_reward)

    if ep % 20 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")


# =======================
# ✅ Plot results
# =======================
plt.figure()
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole Diffusion + Q-learning")
plt.savefig("cartpole_training.png")

print("✅ Done. Plot saved as cartpole_training.png")