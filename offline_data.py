import gymnasium as gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# =======================
# ✅ Hyperparameters
# =======================
NUM_BUCKETS = (6, 6, 12)   # cosθ, sinθ, θ_dot
NUM_ACTIONS = 21           # discretized actions

NUM_EPISODES = 3000
MAX_STEPS = 200

MIN_EXPLORE_RATE = 0.05
MIN_LEARNING_RATE = 0.1
DECAY_FACTOR = 25

GAMMA = 0.99

# =======================
# ✅ Env
# =======================
env = gym.make("Pendulum-v1")

# state bounds
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# action discretization
action_space = np.linspace(-2.0, 2.0, NUM_ACTIONS)

# Q-table
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# =======================
# ✅ Helpers
# =======================
def discretize(obs):
    ratios = [
        (obs[i] - state_bounds[i][0]) /
        (state_bounds[i][1] - state_bounds[i][0])
        for i in range(len(obs))
    ]

    new_obs = [
        int((NUM_BUCKETS[i] - 1) * ratios[i])
        for i in range(len(obs))
    ]

    new_obs = [
        min(NUM_BUCKETS[i] - 1, max(0, new_obs[i]))
        for i in range(len(obs))
    ]

    return tuple(new_obs)


def choose_action(state, explore_rate):
    if random.random() < explore_rate:
        return random.randint(0, NUM_ACTIONS - 1)
    return np.argmax(q_table[state])


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE,
               min(1.0, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE,
               min(0.5, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

# =======================
# ✅ TRAINING
# =======================
rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = discretize(obs)

    explore_rate = get_explore_rate(episode)
    learning_rate = get_learning_rate(episode)

    total_reward = 0

    for step in range(MAX_STEPS):
        action_idx = choose_action(state, explore_rate)
        action = action_space[action_idx]

        next_obs, reward, terminated, truncated, _ = env.step([action])
        next_state = discretize(next_obs)

        # ✅ Q update
        q_table[state + (action_idx,)] += learning_rate * (
            reward + GAMMA * np.max(q_table[next_state])
            - q_table[state + (action_idx,)]
        )

        state = next_state
        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"[TRAIN] Episode {episode}, Reward: {total_reward:.2f}")

env.close()

# =======================
# ✅ DEMONSTRATION COLLECTION
# =======================
print("\nCollecting demonstration data...")

states = []
actions = []
rewards_demo = []
next_states = []

NUM_DEMO_EPISODES = 200

for episode in range(NUM_DEMO_EPISODES):
    obs, _ = env.reset()
    state = discretize(obs)

    for step in range(MAX_STEPS):

        # ✅ greedy policy
        action_idx = np.argmax(q_table[state])
        action = action_space[action_idx]

        next_obs, reward, terminated, truncated, _ = env.step([action])
        next_state = discretize(next_obs)

        # ✅ store continuous values
        states.append(obs)
        actions.append(action)
        rewards_demo.append(reward)
        next_states.append(next_obs)

        obs = next_obs
        state = next_state

        if terminated or truncated:
            break

    if episode % 50 == 0:
        print(f"[DEMO] Episode {episode}")

# =======================
# ✅ SAVE DATASET
# =======================
states = np.array(states)
actions = np.array(actions)
rewards_demo = np.array(rewards_demo)
next_states = np.array(next_states)

np.savez("pendulum_qlearning_data.npz",
         states=states,
         actions=actions,
         rewards=rewards_demo,
         next_states=next_states)

print("\nSaved dataset as pendulum_qlearning_data.npz")
print("Dataset size:", len(states))

# =======================
# ✅ PLOTTING
# =======================
plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning Pendulum")
plt.savefig("pendulum_training_plot.png")

# Action histogram
plt.figure()
plt.hist(actions, bins=50)
plt.title("Action Distribution")
