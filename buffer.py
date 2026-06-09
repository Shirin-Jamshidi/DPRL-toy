import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class OfflineBuffer:
	"""Wraps the static demonstration dataset."""

	def __init__(self, path: str, device: torch.device):
		raw = np.load(path)
		self.states      = torch.tensor(raw["states"],      dtype=torch.float32, device=device)
		self.actions      = torch.tensor(raw["actions"],     dtype=torch.long,    device=device)
		self.rewards      = torch.tensor(raw["rewards"],     dtype=torch.float32, device=device)
		self.next_states  = torch.tensor(raw["next_states"], dtype=torch.float32, device=device)
		# No 'dones' in file — all transitions are mid-episode (reward=1 throughout)
		if "dones" in raw:
			self.dones = torch.tensor(raw["dones"], dtype=torch.float32, device=device)
		else:
			# fallback (just in case old dataset is used)
			self.dones = (self.rewards < 1.0).float()
		self.size         = len(self.states)
		print(f"[OfflineBuffer] Loaded {self.size:,} transitions from {path}")

	def sample(self, batch_size: int) -> dict:
		idx = torch.randint(0, self.size, (batch_size,), device=self.states.device)
		return dict(
			states     = self.states[idx],
			actions    = self.actions[idx],
			rewards    = self.rewards[idx],
			next_states= self.next_states[idx],
			dones      = self.dones[idx],
		)


class OnlineBuffer:
	"""Ring buffer for online experience."""

	def __init__(self, capacity: int, state_dim: int, device: torch.device):
		self.capacity  = capacity
		self.device    = device
		self.ptr       = 0
		self.full      = False

		self.states     = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
		self.actions    = torch.zeros((capacity,),           dtype=torch.long,    device=device)
		self.rewards    = torch.zeros((capacity,),           dtype=torch.float32, device=device)
		self.next_states= torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
		self.dones      = torch.zeros((capacity,),           dtype=torch.float32, device=device)

	def add(self, state, action, reward, next_state, done):
		i = self.ptr
		self.states[i]      = torch.tensor(state,      dtype=torch.float32, device=self.device)
		self.actions[i]     = torch.tensor(action,     dtype=torch.long,    device=self.device)
		self.rewards[i]     = torch.tensor(reward,     dtype=torch.float32, device=self.device)
		self.next_states[i] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
		self.dones[i]       = torch.tensor(done,       dtype=torch.float32, device=self.device)
		self.ptr = (self.ptr + 1) % self.capacity
		if self.ptr == 0:
			self.full = True

	@property
	def size(self) -> int:
		return self.capacity if self.full else self.ptr

	def sample(self, batch_size: int) -> dict:
		idx = torch.randint(0, self.size, (batch_size,), device=self.device)
		return dict(
			states     = self.states[idx],
			actions    = self.actions[idx],
			rewards    = self.rewards[idx],
			next_states= self.next_states[idx],
			dones      = self.dones[idx],
		)



class HyQMixer:
	"""
	Hy-Q hybrid batch sampler.

	At each step draws a batch that is a weighted mixture of offline and
	online data.  The offline ratio β decays linearly from β_start → β_end
	over `anneal_steps` gradient steps, mirroring Algorithm 1 in the Hy-Q
	paper.  Within the offline portion, samples are drawn with probability
	proportional to their |TD-error| computed against the current critics
	(importance weighting for high-value demonstrations).
	"""

	def __init__(
		self,
		offline_buf: OfflineBuffer,
		online_buf:  OnlineBuffer,
		beta_start:  float = 1.0,
		beta_end:    float = 0.25,
		anneal_steps: int  = 50_000,
		td_alpha:    float = 0.6,   # exponent for priority weighting
	):
		self.offline     = offline_buf
		self.online      = online_buf
		self.beta_start  = beta_start
		self.beta_end    = beta_end
		self.anneal_steps= anneal_steps
		self.td_alpha    = td_alpha
		self._step       = 0

		# Uniform priorities at start; updated lazily
		self._offline_priorities = np.ones(offline_buf.size, dtype=np.float32)

	# ------------------------------------------------------------------
	@property
	def beta(self) -> float:
		"""Current offline mixing ratio."""
		frac = min(self._step / max(self.anneal_steps, 1), 1.0)
		return self.beta_start + frac * (self.beta_end - self.beta_start)

	# ------------------------------------------------------------------
	def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
		"""Called by the trainer after each critic update."""
		priorities = (np.abs(td_errors) + 1e-6) ** self.td_alpha
		self._offline_priorities[indices] = priorities

	# ------------------------------------------------------------------
	def sample(self, batch_size: int, q1: nn.Module, q2: nn.Module,
			   gamma: float) -> Tuple[dict, Optional[np.ndarray]]:
		"""
		Returns a mixed batch and the offline indices used (or None).
		The batch always has exactly `batch_size` transitions.
		"""
		self._step += 1
		beta = self.beta

		# How many offline vs online transitions to draw
		n_offline = int(round(beta * batch_size))
		n_online  = batch_size - n_offline

		batches = []
		offline_idx = None

		# ── Offline portion (priority-weighted) ──────────────────────
		if n_offline > 0:
			probs  = self._offline_priorities / self._offline_priorities.sum()
			offline_idx = np.random.choice(self.offline.size, size=n_offline,
											replace=False, p=probs)
			device = self.offline.states.device
			idx_t  = torch.tensor(offline_idx, device=device)
			batch_off = dict(
				states     = self.offline.states[idx_t],
				actions    = self.offline.actions[idx_t],
				rewards    = self.offline.rewards[idx_t],
				next_states= self.offline.next_states[idx_t],
				dones      = self.offline.dones[idx_t],
			)
			batches.append(batch_off)

		# ── Online portion ────────────────────────────────────────────
		if n_online > 0 and self.online.size >= n_online:
			batches.append(self.online.sample(n_online))
		elif n_online > 0:
			# Not enough online data yet — pad with offline
			extra = self.offline.sample(n_online)
			batches.append(extra)

		# ── Concatenate ───────────────────────────────────────────────
		merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
		return merged, offline_idx

