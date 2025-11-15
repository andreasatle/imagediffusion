#!/usr/bin/env python3
"""
Improved minimal diffusion model on MNIST (epoch-free).

✔ Cosine beta schedule
✔ SiLU + GroupNorm
✔ Sinusoidal timestep embedding + MLP (FiLM add)
✔ Correct posterior variance (beta_tilde) in sampling
✔ Randomized batch sampling + large-batch validation
✔ Checkpointing + sampling every N steps
"""

import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms, utils

#from models import MiniUNet, MiniUNetV2, TinyConv
from models import UNetV3

# -------------------------------
# 1) Schedules
# -------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_hat = f / f[0]
    betas = 1.0 - (alphas_hat[1:] / alphas_hat[:-1])
    return betas.to(torch.float32).clamp(1e-5, 0.999)

T = 500
betas  = cosine_beta_schedule(T)          # [T]
alphas = 1.0 - betas                      # [T]
alpha_hat = torch.cumprod(alphas, dim=0)  # [T]

# Precompute posterior variance β_tilde[t] = ((1 - a_hat[t-1])/(1 - a_hat[t])) * beta[t]
posterior_var = torch.zeros_like(betas)
posterior_var[0] = 1e-20
posterior_var[1:] = (1.0 - alpha_hat[:-1]) / (1.0 - alpha_hat[1:]) * betas[1:]

# -------------------------------
# 3) Forward diffusion (q)
# -------------------------------
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    s1 = torch.sqrt(alpha_hat[t])[:, None, None, None]
    s2 = torch.sqrt(1.0 - alpha_hat[t])[:, None, None, None]
    return s1 * x0 + s2 * noise

# -------------------------------
# 4) Loss (MSE on eps)
# -------------------------------
def diffusion_loss(model, x0):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=x0.device).long()
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, noise)

# -------------------------------
# 5) Reverse process (DDPM with posterior variance)
# -------------------------------
@torch.no_grad()
def p_sample(model, x_t, t_scalar: int):
    device = x_t.device
    t = torch.full((x_t.shape[0],), t_scalar, device=device, dtype=torch.long)

    eps_pred = model(x_t, t)

    alpha_t = alphas[t_scalar].to(device)
    a_hat_t = alpha_hat[t_scalar].to(device)

    # mean: (1/sqrt(alpha_t)) * (x_t - ((1 - alpha_t)/sqrt(1 - a_hat_t)) * eps_pred)
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - a_hat_t)) * eps_pred
    )

    if t_scalar == 0:
        return mean

    # correct posterior variance
    sigma_t = torch.sqrt(posterior_var[t_scalar]).to(device)
    return mean + sigma_t * torch.randn_like(x_t)

@torch.no_grad()
def sample(model, shape):
    device = next(model.parameters()).device
    x_t = torch.randn(shape, device=device)
    for t_scalar in reversed(range(T)):
        x_t = p_sample(model, x_t, t_scalar)
    return x_t.clamp(-1, 1)

def to01(x):
    return (x.clamp(-1, 1) + 1) / 2

# -------------------------------
# 6) Data & simple sampler
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
betas, alphas, alpha_hat, posterior_var = (
    betas.to(device), alphas.to(device), alpha_hat.to(device), posterior_var.to(device)
)

class ScaleToMinus1To1:
    def __call__(self, x): return x * 2 - 1

transform = transforms.Compose([transforms.ToTensor(), ScaleToMinus1To1()])
full_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_size = int(0.9 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

def sample_batch(dataset, batch_size, device):
    idx = torch.randint(0, len(dataset), (batch_size,))
    x = torch.stack([dataset[i][0] for i in idx.tolist()]).to(device)
    return x

# -------------------------------
# 7) Training (no epochs)
# -------------------------------
#model = TinyConv(T).to(device)
model = UNetV3(T).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

total_steps     = 50000
batch_size      = 64
log_interval    = 500
sample_interval = 5000

print(f"Training on {device} for {total_steps} total steps...")

running = 0.0
for step in range(1, total_steps + 1):
    model.train()
    x = sample_batch(train_ds, batch_size, device)
    loss = diffusion_loss(model, x)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    running += loss.item()

    if step % log_interval == 0:
        # validation on large batches (smooth metric)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_bs = batch_size * 8
            for _ in range(5):
                xv = sample_batch(val_ds, val_bs, device)
                val_loss += diffusion_loss(model, xv).item()
            val_loss /= 5.0
        print(f"step {step:05d}: train loss={running/log_interval:.4f} | val loss={val_loss:.4f}")
        running = 0.0

    if step % sample_interval == 0:
        model.eval()
        imgs = sample(model, (16, 1, 28, 28)).cpu()
        utils.save_image(to01(imgs), os.path.join(output_dir, f"samples_step{step}.png"), nrow=4)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_step{step}.pt"))

print("Training complete.")
final = sample(model, (16, 1, 28, 28)).cpu()
utils.save_image(to01(final), os.path.join(output_dir, "samples_final.png"), nrow=4)
print(f"Saved final samples to {os.path.join(output_dir, 'samples_final.png')}")
