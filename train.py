import torch
import torch.optim as optim
import numpy as np

from env import GymEnv
from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Env & buffer
env = GymEnv()
buffer = ReplayBuffer()

# ✅ Models
q_net = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy().to(device)

# ✅ Optimizers
q_optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99


# =======================
# ✅ Q-learning update
# =======================
def train_q(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, a, r, s_next = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).to(device)            # (B, 3)
    a = torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1)
    r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1)
    s_next = torch.tensor(s_next, dtype=torch.float32).to(device)  # (B, 3)

    with torch.no_grad():
        n_samples = 10

        s_next_repeat = s_next.unsqueeze(0).repeat(n_samples, 1, 1)  # (N, B, 3)
        s_next_flat = s_next_repeat.view(-1, 3)                     # (N*B, 3)

        noise = torch.randn_like(s_next_flat)[:, :1]                # (N*B, 1)
        actions = policy(s_next_flat, noise)                        # [-1,1]
        actions = 2.0 * actions                                     # scale to [-2,2]

        q_vals = target_q(s_next_flat, actions)                     # (N*B, 1)
        q_vals = q_vals.view(n_samples, -1, 1)                      # (N, B, 1)

        max_q = q_vals.max(dim=0)[0]                                # (B, 1)
        target = r + gamma * max_q

    q = q_net(s, a)
    loss = ((q - target) ** 2).mean()

    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()


# =======================
# ✅ Policy update (QVPO-style)
# =======================
def train_policy(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, _, _, _ = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).to(device)  # (B, 3)

    noise = torch.randn((s.shape[0], 1)).to(device)
    actions = policy(s, noise)       # [-1,1]
    actions = 2.0 * actions          # scale to [-2,2]

    q_val = q_net(s, actions)

    with torch.no_grad():
        baseline = q_val.mean()

    advantage = q_val - baseline
    weights = torch.clamp(advantage, min=0)

    # ✅ maximize Q via weighted objective
    loss = -(weights * q_val).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()


# =======================
# ✅ Training loop
# =======================
num_episodes = 2000
max_steps = 200

for ep in range(num_episodes):
    s = env.reset()
    total_reward = 0

    for t in range(max_steps):
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)  # (1,3)

        # ✅ Sample multiple candidate actions
        actions = []
        q_values = []

        for _ in range(10):
            noise = torch.randn((1, 1)).to(device)
            a = policy(s_tensor, noise)
            a = 2.0 * a  # scale

            a_val = a.item()
            actions.append(a_val)

            q_val = q_net(s_tensor, a).item()
            q_values.append(q_val)

        # ✅ Q-guided selection
        best_idx = np.argmax(q_values)
        a = actions[best_idx]

        # ✅ Environment step
        s_next, r, done = env.step(a)

        buffer.add(s, a, r, s_next)

        # ✅ Train
        train_q()
        train_policy()

        s = s_next
        total_reward += r

        if done:
            break

    # ✅ Soft target update
    for param, target_param in zip(q_net.parameters(), target_q.parameters()):
        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    if ep % 50 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")