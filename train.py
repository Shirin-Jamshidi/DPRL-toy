import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

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
action_dim = 2

# =======================
# ✅ Actor Network
# =======================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, s):
        logits = self.net(s)
        return torch.softmax(logits, dim=-1)


# =======================
# ✅ Critic Network
# =======================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s):
        return self.net(s)


actor = Actor().to(device)
critic = Critic().to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.99


# =======================
# ✅ Load offline data (optional pretraining)
# =======================
def load_data(filename):
    data = np.load(filename)
    return data["states"], data["actions"]


def pretrain_actor_bc(actor, states, actions, epochs=5):
    print("Pretraining actor with behavior cloning...")

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)

    for ep in range(epochs):
        logits = actor(states)
        loss = nn.CrossEntropyLoss()(logits, actions)

        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()

        print(f"BC Epoch {ep}, Loss: {loss.item():.4f}")


# ✅ load and pretrain
states, actions = load_data("cartpole_demo_data.npz")
pretrain_actor_bc(actor, states, actions, epochs=10)


# =======================
# ✅ Training loop (Actor-Critic)
# =======================
num_episodes = 500
max_steps = 200

reward_history = []

for ep in range(num_episodes):
    s, _ = env.reset()
    total_reward = 0

    for t in range(max_steps):
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)

        # sample action
        probs = actor(s_tensor)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()

        s_next, r, terminated, truncated, _ = env.step(a.item())
        done = terminated or truncated

        # critic values
        v = critic(s_tensor)
        s_next_tensor = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0).to(device)
        v_next = critic(s_next_tensor).detach()

        target = r + gamma * v_next * (1 - done)
        advantage = (target - v).detach()

        # ✅ actor loss
        actor_loss = -dist.log_prob(a) * advantage

        # ✅ critic loss
        critic_loss = (v - target) ** 2

        # update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # update critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        s = s_next
        total_reward += r

        if done:
            break

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
plt.title("CartPole Actor-Critic")
plt.savefig("cartpole_actor_critic.png")

print("✅ Done. Plot saved.")