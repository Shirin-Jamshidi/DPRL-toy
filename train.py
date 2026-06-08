import torch
import numpy as np
import matplotlib.pyplot as plt

from env import ContinuousCartPole
from buffer import ReplayBuffer
from models import QNetwork, DiffusionPolicy, sample_action

device = "cuda" if torch.cuda.is_available() else "cpu"

env = ContinuousCartPole()
buffer = ReplayBuffer()

# load offline data
buffer.load_offline("cartpole_demo_data.npz")

q_net = QNetwork().to(device)
target_q = QNetwork().to(device)
target_q.load_state_dict(q_net.state_dict())

policy = DiffusionPolicy().to(device)

q_opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
p_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)

gamma = 0.99

# FQI Update 
def train_q(batch_size=256):

    s,a,r,s2 = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).to(device)
    a = torch.tensor(a, dtype=torch.float32).to(device)
    r = torch.tensor(r, dtype=torch.float32).to(device)
    s2 = torch.tensor(s2, dtype=torch.float32).to(device)

    # max over next action
    # number of action samples
    N = 101

    a_candidates = torch.linspace(-1,1,N, device=device).view(1,N,1)

    # expand states
    s2_expand = s2.unsqueeze(1).expand(-1, N, -1)     # (B, N, 4)
    a_expand = a_candidates.expand(len(s2), -1, -1)   # (B, N, 1)

    # flatten consistently
    s2_flat = s2_expand.reshape(-1, 4)   # (B*N, 4)
    a_flat  = a_expand.reshape(-1, 1)    # (B*N, 1)

    # compute Q
    q_vals = target_q(s2_flat, a_flat)

    # reshape back
    q_vals = q_vals.view(len(s2), N, 1)

    max_q, _ = q_vals.max(dim=1)

    target = r + gamma * max_q

    loss = ((q_net(s,a) - target.detach())**2).mean()

    q_opt.zero_grad()
    loss.backward()
    q_opt.step()

    # update target
    tau = 0.005
    for param, target_param in zip(q_net.parameters(), target_q.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# QVPO Advantage
def compute_weights(s, a):

    q = q_net(s,a)

    # estimate V(s)
    N = 101

    a_samples = torch.linspace(-1,1,N, device=device).view(1,N,1)

    s_expand = s.unsqueeze(1).expand(-1, N, -1)   # (B, N, 4)
    a_expand = a_samples.expand(len(s), -1, -1)   # (B, N, 1)

    s_flat = s_expand.reshape(-1, 4)
    a_flat = a_expand.reshape(-1, 1)

    q_all = q_net(s_flat, a_flat)
    q_all = q_all.view(len(s), N, 1)

    v, _ = q_all.max(dim=1)


    A = q - v
    # normalize
    A = (A - A.mean()) / (A.std() + 1e-6)
    # keep positive part
    weights = torch.clamp(A, min=0.0)
    return weights

# Diffusion Update
def train_policy(batch_size=256):

    s,a,_,_ = buffer.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float32).to(device)
    a = torch.tensor(a, dtype=torch.float32).to(device)

    weights = compute_weights(s,a).detach()

    t = torch.randint(0, policy.T, (len(s),1)).float().to(device)

    noise = torch.randn_like(a)

    beta = 0.02
    alpha = 1 - beta
    a_noisy = torch.sqrt(alpha)*a + torch.sqrt(1-alpha)*noise

    pred = policy(s, a_noisy, t)

    loss = ((noise - pred)**2 * weights).mean()

    p_opt.zero_grad()
    loss.backward()
    p_opt.step()

# Action Selection
def select_action(state):

    samples = sample_action(policy, state)

    s = torch.tensor(state, dtype=torch.float32).to(device)
    s = s.unsqueeze(0).repeat(len(samples),1)

    q_vals = q_net(s, samples.to(device))

    best = torch.argmax(q_vals)

    return samples[best].item()

# Main Training Loop
episode_rewards = []

episodes = 1000

for ep in range(episodes):

    s = env.reset()
    total_reward = 0

    for step in range(200):

        if np.random.rand() < 0.1:
            a = np.random.uniform(-1,1)
        else:
            a = select_action(s)
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
        print(f"Episode {ep}, Reward: {total_reward:.2f}")


plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DP + Q-Learning Training Curve")
plt.grid()

plt.savefig("reward_curve.png")
plt.close()

print("Saved plot to reward_curve.png")