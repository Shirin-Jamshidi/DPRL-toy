import random
import numpy as np
import torch
import argparse


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_config() -> argparse.Namespace:
	p = argparse.ArgumentParser("Diffusion Q-Learning (QVPO + Hy-Q) — CartPole-v1")

	# Environment
	p.add_argument("--env",          default="CartPole-v1")
	p.add_argument("--state_dim",    type=int,   default=4)
	p.add_argument("--n_actions",    type=int,   default=2)
	p.add_argument("--seed",         type=int,   default=42)

	# Demo data
	p.add_argument("--demo_path",    default="draft_cartpole_demo_data.npz")

	# Network architecture
	p.add_argument("--hidden_dim",   type=int,   default=256)
	p.add_argument("--time_emb_dim", type=int,   default=64)
	p.add_argument("--n_diffusion_steps", type=int, default=20)

	# Training
	p.add_argument("--offline_steps",     type=int,   default=20_000)
	p.add_argument("--online_steps",      type=int,   default=100_000)
	p.add_argument("--batch_size",        type=int,   default=256)
	p.add_argument("--gamma",             type=float, default=0.99)
	p.add_argument("--tau",               type=float, default=0.005)
	p.add_argument("--lr_q",              type=float, default=3e-4)
	p.add_argument("--lr_policy",         type=float, default=3e-4)
	p.add_argument("--bc_weight",         type=float, default=0.5)
	p.add_argument("--guidance_scale",    type=float, default=1.0)

	# Hy-Q
	p.add_argument("--hyq_beta_end",      type=float, default=0.25)
	p.add_argument("--hyq_anneal_steps",  type=int,   default=50_000)
	p.add_argument("--hyq_td_alpha",      type=float, default=0.6)
	p.add_argument("--online_capacity",   type=int,   default=200_000)

	# Logging
	p.add_argument("--log_interval",         type=int, default=1_000)
	p.add_argument("--eval_interval_eps",    type=int, default=10)
	p.add_argument("--save_path",            default="checkpoints/diffusion_ql.pt")

	return p.parse_args()

