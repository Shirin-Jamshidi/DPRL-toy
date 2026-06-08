import gymnasium as gym
import numpy as np

class ContinuousCartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        # action ∈ [-1,1]
        discrete = 1 if action > 0 else 0
        obs, reward, terminated, truncated, _ = self.env.step(discrete)
        done = terminated or truncated
        return obs, reward, done

    def sample_random_action(self):
        return np.random.uniform(-1, 1)