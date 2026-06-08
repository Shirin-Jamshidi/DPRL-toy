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
# ✅ Environment (CartPole)
# =======================
env = gym.make("CartPole-v1")
state_dim = 4

buffer = ReplayBuffer()

# =======================
# ✅ Load offline dataset
# =======================
def load_offline_data(buffer, filename):
    print("Loading offline dataset...")

    data = np.load(filename)

    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_states"]

    for i in range(len(states)):
        buffer.add(states[i], float(actions[i]), rewards[i], next_states[i])

    print(f"Loaded {len(states)} samples")

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
# ✅ Sample action (continuous → discrete)
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

    # ✅ map to discrete
    a_val = torch.tanh(a).item()
    prob = (a_val + 1) / 2  # map to [0,1]
    return np.random.choice([0,1], p=[1-prob, prob])



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

    # repeat states
    s_next_repeat = s_next.unsqueeze(0).repeat(n_samples, 1, 1)
    s_next_flat = s_next_repeat.view(-1, state_dim)

    # sample noise actions
    t = torch.randint(0, T, (n_samples * batch_size,), device=device)
    t_norm = t.float().unsqueeze(1) / T

    a_noise = torch.randn((n_samples * batch_size, 1)).to(device)

    eps_pred = policy(s_next_flat, a_noise, t_norm)

    alpha_t = alphas[t].unsqueeze(1)
    alpha_bar_t = alpha_bar[t].unsqueeze(1)

    actions = (1 / torch.sqrt(alpha_t)) * (
        a_noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
    )

    # ✅ map to discrete values
    actions = torch.tanh(actions)
    actions = (actions > 0).float()  # → 0 or 1

    q_vals = target_q(s_next_flat, actions)
    q_vals = q_vals.view(n_samples, batch_size, 1)

    max_q = q_vals.max(dim=0)[0]

    target = r + gamma * max_q

    q = q_net(s, a)

    loss = ((q - target) ** 2).mean()

    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()


# =======================
# ✅ Diffusion policy training
# =======================
def train_policy(batch_size=64):
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

    # Q-weighting
    q_val = q_net(s, a)

    with torch.no_grad():
        baseline = q_val.mean()

    advantage = q_val - baseline
    weights = torch.clamp(advantage, min=0)

    bc_loss = ((eps - eps_pred) ** 2).mean()
    loss = (weights * (eps - eps_pred) ** 2).mean() + 0.1 * bc_loss


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
    s, _ = env.reset()
    total_reward = 0

    for t in range(max_steps):

        # sample multiple actions
        candidates = [sample_action(s) for _ in range(10)]

        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

        q_values = []
        for a in candidates:
            a_tensor = torch.tensor([[a]], dtype=torch.float32).to(device)
            q_values.append(q_net(s_tensor, a_tensor).item())

        a = candidates[np.argmax(q_values)]

        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        buffer.add(s, a, r, s_next)

        train_q()
        train_policy()

        s = s_next
        total_reward += r

        if done:
            break

    # target update
    for p, tp in zip(q_net.parameters(), target_q.parameters()):
        tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

    reward_history.append(total_reward)

    if ep % 50 == 0:
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

print("Training finished. Plot saved as cartpole_training.png")