import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim=4, T=10):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1 + 1, 256),  # s, a_noisy, t_normalised
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)                    # predicted noise ε
        )

    def forward(self, s, a_noisy, t_norm):
        return self.net(torch.cat([s, a_noisy, t_norm], dim=-1))


def sample_action(policy, state, alpha_bars, T, n_samples=64):
    """
    DDPM reverse process: x_T ~ N(0,I), then denoise step-by-step to x_0.
    Returns tensor of shape (n_samples, 1) on the same device as policy.
    """
    device = next(policy.parameters()).device

    s = torch.tensor(state, dtype=torch.float32, device=device)
    s = s.unsqueeze(0).expand(n_samples, -1)

    # Start from pure Gaussian noise at t=T
    x = torch.randn(n_samples, 1, device=device)

    for t in range(T, 0, -1):
        t_norm    = torch.full((n_samples, 1), t / T,
                               dtype=torch.float32, device=device)
        eps_pred  = policy(s, x, t_norm)

        ab_t      = alpha_bars[t]
        ab_prev   = alpha_bars[t - 1]
        alpha_t   = ab_t / ab_prev                        # α_t = ᾱ_t / ᾱ_{t-1}

        # DDPM mean: x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε)
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - (1.0 - alpha_t) / torch.sqrt(1.0 - ab_t) * eps_pred
        )
        # Add noise for t > 1 (stochastic reverse); deterministic at t=1
        if t > 1:
            sigma = torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - alpha_t))
            x = x + sigma * torch.randn_like(x)

    return x.clamp(-1.0, 1.0)
