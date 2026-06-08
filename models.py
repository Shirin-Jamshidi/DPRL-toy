import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)
    

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim=4, T=10):
        super().__init__()
        self.T = T

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1 + 1, 128),  # s + a + t
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a, t):
        return self.net(torch.cat([s, a, t], dim=-1))
    
def sample_action(policy, state, n_samples=32):
    device = next(policy.parameters()).device

    s = torch.tensor(state, dtype=torch.float32).to(device)
    s = s.unsqueeze(0).repeat(n_samples,1)

    a = torch.randn(n_samples,1).to(device)

    for t in reversed(range(policy.T)):
        t_tensor = torch.ones((n_samples,1)).to(device) * t
        noise_pred = policy(s, a, t_tensor)
        a = a - noise_pred * 0.1

    return a