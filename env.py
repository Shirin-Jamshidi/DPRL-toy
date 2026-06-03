import numpy as np

class ToyEnv:
    def __init__(self):
        self.goal = 0.0
        self.max_steps = 50

    def reset(self):
        self.state = np.random.uniform(-2.0, 2.0)
        self.steps = 0
        return self.state

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state = self.state + action
        reward = -abs(next_state - self.goal)

        self.state = next_state
        self.steps += 1

        done = self.steps >= self.max_steps or abs(next_state) < 0.01
        return next_state, reward, done