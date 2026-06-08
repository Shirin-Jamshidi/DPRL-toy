import gymnasium as gym

class GymEnv:
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        # Gym expects shape (1,)
        action = [action]

        state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        return state, reward, done