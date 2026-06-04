import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from env import GymEnv
from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy

# =======================
# ✅ Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# ✅ Env + buffer
# =======================
env = GymEnv()
buffer = ReplayBuffer()

# =======================
# ✅ Models
# =======================
q_net = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy().to(device)

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
# ✅ Sample action (reverse diffusion)
# =======================
def sample_action(state):
    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    a = torch.randn((1, 1)).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([[t / T]], dtype=torch.float32).to(device)

        epsilon_pred = policy(s, a, t_tensor)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]

        if t > 0:
            noise = torch.randn_like(a)
        else:
            noise = 0

        a = (1 / torch.sqrt(alpha_t)) * (
            a - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
        ) + torch.sqrt(betas[t]) * noise

    return torch.tanh(a).item() * 2.0  # scale to [-2, 2]


# =======================
# ✅ Q-learning (FQI)
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
    batch_size = s_next.shape[0]

    # ✅ repeat states
    s_next_repeat = s_next.unsqueeze(0).repeat(n_samples, 1, 1)
    s_next_flat = s_next_repeat.view(-1, 3)  # (N*B,3)

    # ✅ timestep sampling
    t = torch.randint(0, T, (n_samples * batch_size,), device=device)
    t_norm = t.float().unsqueeze(1) / T

    # ✅ start from noise
    a_noise = torch.randn((n_samples * batch_size, 1)).to(device)

    # ✅ predict noise
    epsilon_pred = policy(s_next_flat, a_noise, t_norm)

    alpha_t = alphas[t].unsqueeze(1)
    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    # ✅ 1-step denoising
    actions = (1 / torch.sqrt(alpha_t)) * (
        a_noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
    )

    actions = torch.tanh(actions) * 2.0

    # ✅ compute Q
    q_vals = target_q(s_next_flat, actions)  # (N*B,1)
    q_vals = q_vals.view(n_samples, batch_size, 1)

    max_q = q_vals.max(dim=0)[0]  # (B,1)

    target = r + gamma * max_q

    q = q_net(s, a)

    loss = ((q - target) ** 2).mean()

    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()

# =======================
# ✅ Expert policy for Pendulum (offline data)
# =======================
def expert_policy(state):
    cos_theta, sin_theta, theta_dot = state

    theta = np.arctan2(sin_theta, cos_theta)

    # ✅ Energy-based term (swing-up)
    g = 10.0
    m = 1.0
    l = 1.0

    desired_energy = 0.0
    current_energy = 0.5 * m * (l * theta_dot) ** 2 + m * g * l * (1 - cos_theta)

    energy_error = current_energy - desired_energy

    # ✅ Control terms
    k_energy = 1.0
    k_p = 5.0
    k_d = 1.0

    # Swing-up + stabilization
    u = k_energy * theta_dot * energy_error - k_p * theta - k_d * theta_dot

    return np.clip(u, -2.0, 2.0)

# =======================
# ✅ Generate offline data
# =======================
def generate_offline_data(env, buffer, num_episodes=200):
    print("Generating offline dataset...")

    for ep in range(num_episodes):
        s = env.reset()

        for _ in range(200):
            a = expert_policy(s)

            s_next, r, done = env.step(a)

            buffer.add(s, a, r, s_next)

            s = s_next

            if done:
                break

    print(f"Offline dataset size: {len(buffer)}")


# ✅ Generate D_off
generate_offline_data(env, buffer, num_episodes=200)

# =======================
# ✅ Diffusion policy training (Eq 4)
# =======================
def train_policy(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, a, _, _ = buffer.sample(batch_size)

    s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
    a = torch.tensor(np.array(a), dtype=torch.float32).unsqueeze(1).to(device)

    # ✅ timestep
    t = torch.randint(0, T, (batch_size,), device=device)
    t_norm = t.float().unsqueeze(1) / T

    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    # ✅ sample noise (ε)
    epsilon = torch.randn_like(a)

    # ✅ forward diffusion
    noisy_a = torch.sqrt(alpha_bar_t) * a + torch.sqrt(1 - alpha_bar_t) * epsilon

    # ✅ predict εθ
    epsilon_pred = policy(s, noisy_a, t_norm)

    # ✅ Q weighting
    q_val = q_net(s, a)

    with torch.no_grad():
        baseline = q_val.mean()

    advantage = q_val - baseline
    weights = torch.clamp(advantage, min=0)

    # ✅ Eq (4)
    loss = (weights * (epsilon - epsilon_pred) ** 2).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()

# =======================
# ✅ Training loop
# =======================
num_episodes = 2000
max_steps = 200

reward_history = []
for ep in range(num_episodes):
    s = env.reset()
    total_reward = 0

    for t in range(max_steps):

        actions = []
        q_values = []

        for _ in range(10):
            a_candidate = sample_action(s)
            actions.append(a_candidate)

            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
            a_tensor = torch.tensor([[a_candidate]], dtype=torch.float32).to(device)

            q_val = q_net(s_tensor, a_tensor).item()
            q_values.append(q_val)

        # ✅ pick best action
        best_idx = np.argmax(q_values)
        a = actions[best_idx]

        s_next, r, done = env.step(a)

        buffer.add(s, a, r, s_next)

        train_q()
        train_policy()

        s = s_next
        total_reward += r

        if done:
            break

    # ✅ soft target update
    for param, target_param in zip(q_net.parameters(), target_q.parameters()):
        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    reward_history.append(total_reward)
    if ep % 50 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")


# =======================
# ✅ Save reward plot
# =======================
plt.figure()
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward (Pendulum Diffusion RL)")

plt.savefig("reward_plot.png")