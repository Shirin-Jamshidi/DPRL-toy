import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def add(self, s, a, r, s2):
        if self.size < self.max_size:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s2)
        else:
            idx = self.ptr
            self.states[idx] = s
            self.actions[idx] = a
            self.rewards[idx] = r
            self.next_states[idx] = s2

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        idx = np.random.choice(self.size, batch_size)

        return (
            np.array(self.states)[idx],
            np.array(self.actions)[idx].reshape(-1,1),
            np.array(self.rewards)[idx].reshape(-1,1),
            np.array(self.next_states)[idx]
        )

    def load_offline(self, path):
        data = np.load(path)

        for s,a,r,s2 in zip(
            data["states"],
            data["actions"],
            data["rewards"],
            data["next_states"]
        ):
            a_cont = 2.0 * a - 1.0  # ✅ discrete → continuous
            self.add(s, a_cont, r, s2)