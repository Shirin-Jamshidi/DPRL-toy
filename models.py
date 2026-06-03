import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
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
            nn.Linear(4, 128),  # state + noise
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, noise):
        x = torch.cat([s, noise], dim=-1)
        return torch.tanh(self.net(x))