# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SinusoidalTimeEmbedding(nn.Module):
# 	def __init__(self, dim: int):
# 		super().__init__()
# 		self.dim = dim

# 	def forward(self, t: torch.Tensor) -> torch.Tensor:
# 		"""t: (B,) long or float in [0, T-1]"""
# 		half  = self.dim // 2
# 		freqs = torch.exp(
# 			-math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
# 		)
# 		emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
# 		return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


# class ScoreNetwork(nn.Module):
# 	"""
# 	Predicts the denoising score  s_θ(a_t, t, s) ≈ ∇_{a_t} log p(a_t | s).

# 	For discrete CartPole (2 actions) we parameterise the score as an
# 	embedding over action indices rather than a continuous vector.
# 	The network outputs a scalar logit per action — during reverse diffusion
# 	we use a Gumbel-softmax relaxation so gradients flow through.

# 	Input:  (noisy_action_emb, time_emb, state)
# 	Output: action logits of shape (B, n_actions)
# 	"""

# 	def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256,
# 				 time_emb_dim: int = 64, n_diffusion_steps: int = 20):
# 		super().__init__()
# 		self.n_actions        = n_actions
# 		self.n_diffusion_steps= n_diffusion_steps
# 		action_emb_dim        = 32  # learned embedding per discrete action

# 		self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
# 		self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)

# 		in_dim = action_emb_dim + time_emb_dim + state_dim
# 		self.net = nn.Sequential(
# 			nn.Linear(in_dim, hidden_dim), nn.Mish(),
# 			nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
# 			nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
# 			nn.Linear(hidden_dim, n_actions),   # raw logits
# 		)

# 	def forward(self, noisy_action: torch.Tensor, t: torch.Tensor,
# 				state: torch.Tensor) -> torch.Tensor:
# 		"""
# 		noisy_action : (B,)  long — discrete noisy action token
# 		t            : (B,)  long — diffusion timestep in [0, T-1]
# 		state        : (B, state_dim)
# 		returns      : (B, n_actions)  score logits
# 		"""
# 		a_emb = self.action_emb(noisy_action)     # (B, action_emb_dim)
# 		t_emb = self.time_emb(t)                   # (B, time_emb_dim)
# 		x     = torch.cat([a_emb, t_emb, state], dim=-1)
# 		return self.net(x)



# class QNetwork(nn.Module):
# 	def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
# 		super().__init__()
# 		self.net = nn.Sequential(
# 			nn.Linear(state_dim, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, n_actions),
# 		)

# 	def forward(self, state: torch.Tensor) -> torch.Tensor:
# 		"""Returns Q-values for all actions: (B, n_actions)"""
# 		return self.net(state)


# class DiscreteGaussianDiffusion(nn.Module):
#     MASK_TOKEN = 2  # special token outside {0,1}

#     def __init__(self, n_steps: int = 20):
#         super().__init__()
#         self.T = n_steps

#         betas = torch.linspace(0.01, 0.5, n_steps)
#         alphas = 1 - betas
#         alpha_bar = torch.cumprod(alphas, dim=0)

#         # ✅ buffer → automatically moves with .to(device)
#         self.register_buffer("alpha_bar", alpha_bar)

#     def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         alpha_bar_t = self.alpha_bar[t]
#         keep_mask = torch.bernoulli(alpha_bar_t).bool()
#         return torch.where(
#             keep_mask,
#             x0,
#             torch.full_like(x0, self.MASK_TOKEN)
#         )

#     def p_losses(self, score_net, x0: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
#         B = x0.shape[0]
#         t = torch.randint(0, self.T, (B,), device=x0.device)
#         noisy = self.q_sample(x0, t)
#         logits = score_net(noisy, t, state)
#         return F.cross_entropy(logits[:, :score_net.n_actions], x0)

#     @torch.no_grad()
#     def p_sample(self, score_net, state: torch.Tensor,
#                  q1, q2, guidance_scale: float = 1.0) -> torch.Tensor:

#         B = state.shape[0]
#         device = state.device

#         x_t = torch.full((B,), self.MASK_TOKEN, dtype=torch.long, device=device)

#         for t_val in reversed(range(self.T)):
#             t = torch.full((B,), t_val, dtype=torch.long, device=device)
#             logits = score_net(x_t, t, state)

#             if guidance_scale > 0.0:
#                 q_vals = torch.min(q1(state), q2(state))
#                 q_adv = q_vals - q_vals.mean(dim=1, keepdim=True)
#                 logits = logits[:, :2] + guidance_scale * q_adv
#             else:
#                 logits = logits[:, :2]

#             probs = F.softmax(logits, dim=-1)
#             x_t = torch.multinomial(probs, 1).squeeze(1)

#         return x_t

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Time embedding (unchanged)
# --------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------
# ✅ CONTINUOUS Score Network
# --------------------------------------------------

class ScoreNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256,
                 time_emb_dim: int = 64, n_diffusion_steps: int = 20):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = 1 + time_emb_dim + state_dim   # ✅ continuous input (1)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1),  # ✅ predict noise scalar
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor):
        # ✅ ensure shape (B,1)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        elif x_t.dim() > 2:
            x_t = x_t.view(x_t.size(0), -1)

        t_emb = self.time_emb(t)
        x = torch.cat([x_t, t_emb, state], dim=-1)
        return self.net(x)


# --------------------------------------------------
# Q Network (unchanged)
# --------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor):
        return self.net(state)


# --------------------------------------------------
# ✅ CONTINUOUS Gaussian Diffusion (DDPM)
# --------------------------------------------------

class GaussianDiffusion(nn.Module):
    def __init__(self, n_steps: int = 20):
        super().__init__()
        self.T = n_steps

        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    # ------------------------------------------
    # discrete ↔ continuous mapping
    # ------------------------------------------

    def to_continuous(self, a):
        return a.float() * 2.0 - 1.0  # {0,1} → {-1,+1}

    def to_discrete(self, x):
        prob = torch.sigmoid(x)
        return torch.bernoulli(prob).long().squeeze(1)

    # ------------------------------------------

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].unsqueeze(1)

        return (
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

    # ------------------------------------------

    def p_losses(self, score_net, actions, state):
        B = actions.shape[0]

        x0 = self.to_continuous(actions).unsqueeze(1)

        t = torch.randint(0, self.T, (B,), device=actions.device)

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        pred_noise = score_net(x_t, t, state)

        return F.mse_loss(pred_noise, noise)

    # ------------------------------------------

    @torch.no_grad()
    def p_sample(self, score_net, state, q1, q2, guidance_scale=1.0):
        B = state.shape[0]
        device = state.device

        x_t = torch.randn(B, 1, device=device)

        for t_val in reversed(range(self.T)):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            pred_noise = score_net(x_t, t, state)

            alpha_t = self.alphas[t].unsqueeze(1)
            alpha_bar_t = self.alpha_bar[t].unsqueeze(1)
            beta_t = self.betas[t].unsqueeze(1)

            # ✅ proper DDPM update
            noise = torch.randn_like(x_t) if t_val > 0 else 0.0

            x_t = (
                1 / torch.sqrt(alpha_t)
                * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
                + torch.sqrt(beta_t) * noise
            )

            # ✅ Q-guidance (kept intact conceptually)
            if guidance_scale > 0:
                actions_disc = self.to_discrete(x_t)
                q_vals = torch.min(q1(state), q2(state))
                q_adv = q_vals - q_vals.mean(dim=1, keepdim=True)

                grad = torch.gather(q_adv, 1, actions_disc.unsqueeze(1)).float()
                x_t = x_t + guidance_scale * grad

        return self.to_discrete(x_t)
