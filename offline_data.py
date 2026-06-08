import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1")

states, actions, rewards, next_states = [], [], [], []

# =======================
# ✅ Q-learning-style policy (sampling-based)
# =======================
def q_policy(state):
    cos_theta, sin_theta, theta_dot = state
    theta = np.arctan2(sin_theta, cos_theta)

    candidate_actions = np.linspace(-2, 2, 21)

    best_a = 0
    best_value = -np.inf

    for a in candidate_actions:
        # ✅ objective
        value = (
            -theta**2
            - 0.1 * theta_dot**2
            + 0.5 * theta_dot * a   # ✅ encourages actuation
            - 0.001 * a**2
        )

        if value > best_value:
            best_value = value
            best_a = a

    return best_a


# =======================
# ✅ Data collection
# =======================
for ep in range(200):
    s, _ = env.reset()

    for _ in range(200):
        # ✅ use Q-style policy
        a = q_policy(s)

        s_next, r, term, trunc, _ = env.step([a])

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

        s = s_next

        if term or trunc:
            break

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

print("Saved pendulum_demo_data.npz")
print("Dataset size:", len(states))


# =======================
# ✅ Plotting
# =======================

# --- 1. States ---
plt.figure()
for i in range(states.shape[1]):
    plt.plot(states[:, i], label=f"state[{i}]")
plt.legend()
plt.title("Pendulum State Trajectories")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.savefig("pendulum_states.png")


# --- 2. Actions ---
plt.figure()
plt.hist(actions, bins=50)
plt.title("Action Distribution")
plt.xlabel("Action")
plt.ylabel("Count")
plt.savefig("pendulum_actions_hist.png")


# --- 3. Rewards ---
plt.figure()
plt.plot(rewards)
plt.title("Rewards over Time")
plt.xlabel("Time step")
plt.ylabel("Reward")
plt.savefig("pendulum_rewards.png")


# --- 4. State vs Action ---
plt.figure()
plt.scatter(states[:, 0], actions, s=1)
plt.title("cos(theta) vs action")
plt.xlabel("cos(theta)")
plt.ylabel("action")
plt.savefig("pendulum_state_action.png")


print("\nSaved plots:")
print(" - pendulum_states.png")
print(" - pendulum_actions_hist.png")
print(" - pendulum_rewards.png")
print(" - pendulum_state_action.png")