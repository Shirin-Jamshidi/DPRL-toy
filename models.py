import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		"""t: (B,) long or float in [0, T-1]"""
		half  = self.dim // 2
		freqs = torch.exp(
			-math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
		)
		emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
		return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


class ScoreNetwork(nn.Module):
	"""
	Predicts the denoising score  s_θ(a_t, t, s) ≈ ∇_{a_t} log p(a_t | s).

	For discrete CartPole (2 actions) we parameterise the score as an
	embedding over action indices rather than a continuous vector.
	The network outputs a scalar logit per action — during reverse diffusion
	we use a Gumbel-softmax relaxation so gradients flow through.

	Input:  (noisy_action_emb, time_emb, state)
	Output: action logits of shape (B, n_actions)
	"""

	def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256,
				 time_emb_dim: int = 64, n_diffusion_steps: int = 20):
		super().__init__()
		self.n_actions        = n_actions
		self.n_diffusion_steps= n_diffusion_steps
		action_emb_dim        = 32  # learned embedding per discrete action

		self.action_emb = nn.Embedding(n_actions + 1, action_emb_dim)
		self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)

		in_dim = action_emb_dim + time_emb_dim + state_dim
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), nn.Mish(),
			nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
			nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
			nn.Linear(hidden_dim, n_actions),   # raw logits
		)

	def forward(self, noisy_action: torch.Tensor, t: torch.Tensor,
				state: torch.Tensor) -> torch.Tensor:
		"""
		noisy_action : (B,)  long — discrete noisy action token
		t            : (B,)  long — diffusion timestep in [0, T-1]
		state        : (B, state_dim)
		returns      : (B, n_actions)  score logits
		"""
		a_emb = self.action_emb(noisy_action)     # (B, action_emb_dim)
		t_emb = self.time_emb(t)                   # (B, time_emb_dim)
		x     = torch.cat([a_emb, t_emb, state], dim=-1)
		return self.net(x)



class QNetwork(nn.Module):
	def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, n_actions),
		)

	def forward(self, state: torch.Tensor) -> torch.Tensor:
		"""Returns Q-values for all actions: (B, n_actions)"""
		return self.net(state)



class DiscreteGaussianDiffusion:
	"""
	Implements a simple masking / discrete diffusion schedule for action tokens.

	We use the "absorbing state" formulation common for discrete diffusion:
	  - Forward: at each step, independently mask each position with prob β_t
	  - Reverse: learn to unmask via the score network

	For CartPole's binary action space, the absorbing/mask token is index 2.
	"""

	MASK_TOKEN = 2  # special token outside {0,1}

	def __init__(self, n_steps: int = 20):
		self.T = n_steps
		# Linear schedule for mask probability
		betas = torch.linspace(0.01, 0.5, n_steps)
		# Cumulative product: probability of NOT being masked up to step t
		alphas = 1.0 - betas
		self.alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)
		self.betas     = betas

	def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""
		Forward process: corrupt x0 at timestep t.
		x0 : (B,) long ∈ {0,1}
		t  : (B,) long ∈ [0, T-1]
		returns noisy actions (B,) — some entries replaced by MASK_TOKEN
		"""
		alpha_bar_t = self.alpha_bar[t].to(x0.device)      # (B,)
		keep_mask   = torch.bernoulli(alpha_bar_t).bool()   # True → keep original
		noisy       = torch.where(keep_mask, x0, torch.full_like(x0, self.MASK_TOKEN))
		return noisy

	def p_losses(self, score_net: ScoreNetwork, x0: torch.Tensor,
				 state: torch.Tensor) -> torch.Tensor:
		"""
		Denoising score matching loss (cross-entropy on clean action prediction).
		"""
		B      = x0.shape[0]
		t      = torch.randint(0, self.T, (B,), device=x0.device)
		noisy  = self.q_sample(x0, t)
		logits = score_net(noisy, t, state)          # (B, n_actions)
		# Predict clean action class — standard DDPM "x0 parameterisation"
		return F.cross_entropy(logits[:, :score_net.n_actions], x0)

	@torch.no_grad()
	def p_sample(self, score_net: ScoreNetwork, state: torch.Tensor,
				 q1: QNetwork, q2: QNetwork,
				 guidance_scale: float = 1.0) -> torch.Tensor:
		"""
		Reverse diffusion sampling with Q-guidance (QVPO Algorithm 1).
		Starts from fully masked and iteratively denoises.

		Returns: (B,) long — sampled discrete actions
		"""
		B      = state.shape[0]
		device = state.device
		# Start from fully masked
		x_t = torch.full((B,), self.MASK_TOKEN, dtype=torch.long, device=device)

		for t_val in reversed(range(self.T)):
			t = torch.full((B,), t_val, dtype=torch.long, device=device)
			logits = score_net(x_t, t, state)        # (B, n_actions)

			# ── Q-guidance  (QVPO §3.2) ──────────────────────────────
			# Q-values for valid actions; add scaled Q-advantage to logits
			if guidance_scale > 0.0:
				q_vals  = torch.min(q1(state), q2(state))   # (B, n_actions)
				q_adv   = q_vals - q_vals.mean(dim=1, keepdim=True)
				logits  = logits[:, :2] + guidance_scale * q_adv
			else:
				logits  = logits[:, :2]

			# Gumbel-softmax → hard sample
			probs = F.softmax(logits, dim=-1)
			x_t   = torch.multinomial(probs, 1).squeeze(1)  # (B,)

		return x_t

