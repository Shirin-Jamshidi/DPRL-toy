import gymnasium as gym
import numpy as np

env = gym.make("Pendulum-v1")

states, actions, rewards, next_states = [], [], [], []

for ep in range(200):
    s, _ = env.reset()

    for _ in range(200):
        # ✅ simple expert policy
        cos, sin, theta_dot = s
        theta = np.arctan2(sin, cos)

        a = -2.0 * theta - 0.5 * theta_dot
        a = np.clip(a, -2.0, 2.0)

        s_next, r, term, trunc, _ = env.step([a])

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

        s = s_next
        if term or trunc:
            break

np.savez("pendulum_demo_data.npz",
         states=np.array(states),
         actions=np.array(actions),
         rewards=np.array(rewards),
         next_states=np.array(next_states))

print("Saved pendulum_demo_data.npz")