import torch
import torch.optim as optim
import numpy as np

from env import ToyEnv
from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ToyEnv()
buffer = ReplayBuffer()

q_net = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy().to(device)

q_optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99

# ✅ Q-learning step
def train_q(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, a, r, s_next = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).unsqueeze(1).to(device)
    a = torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)
    r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
    s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        # sample actions for max approximation
        n_samples = 10
        next_actions = []
        for _ in range(n_samples):
            noise = torch.randn_like(s_next)
            action = policy(s_next, noise)
            next_actions.append(action)

        next_actions = torch.stack(next_actions)
        q_vals = target_q(
            s_next.repeat(n_samples,1,1).view(-1,1),
            next_actions.view(-1,1)
        )

        q_vals = q_vals.view(n_samples, batch_size, 1)
        max_q = q_vals.max(dim=0)[0]

        target = r + gamma * max_q

    q = q_net(s, a)
    loss = ((q - target)**2).mean()

    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()


# ✅ Diffusion policy update (QVPO style)
def train_policy(batch_size=64):
    if len(buffer) < batch_size:
        return

    s, a, r, s_next = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).unsqueeze(1).to(device)

    noise = torch.randn_like(s)
    generated_action = policy(s, noise)

    q_val = q_net(s, generated_action)

    with torch.no_grad():
        baseline = q_val.mean()

    advantage = q_val - baseline
    weights = torch.clamp(advantage, min=0)

    loss = -(weights * q_val).mean()

    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()


# ✅ Training loop
num_episodes = 3000

for ep in range(num_episodes):
    s = env.reset()
    total_reward = 0

    for _ in range(50):
        s_tensor = torch.tensor([[s]], dtype=torch.float32).to(device)

        # sample multiple actions
        actions = []
        for _ in range(10):
            noise = torch.randn_like(s_tensor)
            a = policy(s_tensor, noise).item()
            actions.append(a)

        # pick best action using Q
        q_values = []
        for a in actions:
            a_tensor = torch.tensor([[a]], dtype=torch.float32).to(device)
            q_values.append(q_net(s_tensor, a_tensor).item())

        a = actions[np.argmax(q_values)]

        s_next, r, done = env.step(a)

        buffer.add(s, a, r, s_next)

        train_q()
        train_policy()

        s = s_next
        total_reward += r

        if done:
            break

    # soft target update
    for param, target_param in zip(q_net.parameters(), target_q.parameters()):
        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")
