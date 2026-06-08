import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# =======================
# ✅ Hyperparameters
# =======================
NUM_EPISODES = 200
MAX_STEPS = 200

# =======================
# ✅ Environment
# =======================
env = gym.make("Pendulum-v1")

# =======================
# ✅ Continuous Q-style policy
# =======================
def q_style_policy(state):
    cos_theta, sin_theta, theta_dot = state
    theta = np.arctan2(sin_theta, cos_theta)

    # ✅ sample candidate actions
    candidate_actions = np.linspace(-2, 2, 25)

    best_a = 0.0
    best_value = -np.inf

    for a in candidate_actions:
        # ✅ refined objective (balanced)
        value = (
            -theta**2
            - 0.1 * theta_dot**2
            + 0.5 * theta_dot * a      # encourages control
            - 0.001 * a**2
        )

        if value > best_value:
            best_value = value
            best_a = a

    # ✅ add small noise (important)
    best_a += np.random.normal(0, 0.1)

    return np.clip(best_a, -2.0, 2.0)


# =======================
# ✅ Data collection
# =======================
states = []
actions = []
rewards = []
next_states = []

episode_rewards = []

print("Collecting demonstrations...")

for ep in range(NUM_EPISODES):
    s, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        a = q_style_policy(s)

        s_next, r, term, trunc, _ = env.step([a])

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

        s = s_next
        total_reward += r

        if term or trunc:
            break

    episode_rewards.append(total_reward)

    if ep % 20 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

# =======================
# ✅ Save dataset
# =======================
states = np.array(states)
actions = np.array(actions)
rewards = np.array(rewards)
next_states = np.array(next_states)

np.savez("pendulum_demo_data.npz",
         states=states,
         actions=actions,
         rewards=rewards,
         next_states=next_states)

print("\nSaved dataset as pendulum_demo_data.npz")
print("Dataset size:", len(states))


# =======================
# ✅ Plotting
# =======================

# --- 1. Episode rewards ---
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward (Q-style Pendulum)")
plt.savefig("episode_rewards.png")

# --- 2. Action distribution ---
plt.figure()
plt.hist(actions, bins=50)
plt.xlabel("Action")
plt.ylabel("Count")
plt.title("Action Distribution")
plt.savefig("action_hist.png")

# --- 3. States ---
plt.figure()
for i in range(states.shape[1]):
    plt.plot(states[:, i], label=f"state[{i}]")

plt.legend()
plt.title("State Trajectories")
plt.savefig("states_plot.png")

# --- 4. Reward over time ---
plt.figure()
plt.plot(rewards)
plt.title("Step-wise Rewards")
plt.savefig("rewards_plot.png")

# --- 5. State vs Action ---
plt.figure()
plt.scatter(states[:, 0], actions, s=1)
plt.xlabel("cos(theta)")
plt.ylabel("action")
plt.title("State-Action Relationship")
plt.savefig("state_action.png")

print("\nSaved plots:")
print(" - episode_rewards.png")
print(" - action_hist.png")
print(" - states_plot.png")
print(" - rewards_plot.png")
print(" - state_action.png")