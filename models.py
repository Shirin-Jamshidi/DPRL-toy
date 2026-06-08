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
            nn.Linear(state_dim + 1 + 1, 128),  # s + a_noisy + t
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a, t):
        return self.net(torch.cat([s, a, t], dim=-1))


# FIX: sample_action now accepts alpha_bars and uses the proper reverse schedule
def sample_action(policy, state, alpha_bars, n_samples=32):
    device = next(policy.parameters()).device

    s = torch.tensor(state, dtype=torch.float32).to(device)
    s = s.unsqueeze(0).repeat(n_samples, 1)

    # Start from pure noise (x_T)
    a = torch.randn(n_samples, 1).to(device)

    T = policy.T
    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples, 1), t, dtype=torch.float32, device=device)
        noise_pred = policy(s, a, t_tensor)

        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        # DDPM reverse step: x_{t-1} = (x_t - sqrt(1-ab_t)*eps) / sqrt(ab_t/ab_prev)
        # simplified (no added noise at inference for determinism)
        a = (a - torch.sqrt(1.0 - ab_t) * noise_pred) / torch.sqrt(ab_t / ab_prev)
        a = torch.clamp(a, -1.0, 1.0)

    return a
