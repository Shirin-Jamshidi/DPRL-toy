import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)
    

class DiffusionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + 1 + 1, 128),  # state + noisy_action + t
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # predict noise
        )

    def forward(self, s, noisy_a, t):
        x = torch.cat([s, noisy_a, t], dim=-1)
        return self.net(x)
