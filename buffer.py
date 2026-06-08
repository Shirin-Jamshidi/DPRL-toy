import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=200_000):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.next_states = []
        self.returns     = []   # MC returns (for policy advantage weights)

    def add(self, s, a, r, s2, mc_return=None):
        entry = (s, a, r, s2, mc_return if mc_return is not None else r)
        if self.size < self.max_size:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s2)
            self.returns.append(entry[4])
        else:
            idx = self.ptr
            self.states[idx]      = s
            self.actions[idx]     = a
            self.rewards[idx]     = r
            self.next_states[idx] = s2
            self.returns[idx]     = entry[4]
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=256):
        idx = np.random.choice(self.size, batch_size, replace=True)
        return (
            np.array(self.states)[idx],
            np.array(self.actions)[idx].reshape(-1, 1),
            np.array(self.rewards)[idx].reshape(-1, 1),
            np.array(self.next_states)[idx],
            np.array(self.returns)[idx].reshape(-1, 1),
        )

    def load_offline(self, path):
        data = np.load(path)
        # Offline data: compute a rough MC return as just r (no future known)
        # Q pre-training will correct this via bootstrapping.
        for s, a, r, s2 in zip(data["states"], data["actions"],
                                data["rewards"], data["next_states"]):
            a_cont = 2.0 * a - 1.0   # discrete {0,1} -> continuous {-1,+1}
            self.add(s, a_cont, r, s2, mc_return=r)
