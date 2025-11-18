import torch
import math


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Classic DDPM linear beta schedule.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal, 2021).
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_hat = f / f[0]
    betas = 1.0 - (alphas_hat[1:] / alphas_hat[:-1])
    return betas.to(torch.float32).clamp(1e-5, 0.999)


class DiffusionSchedule:
    """
    Wraps the full computation of:
        betas[t]
        alphas[t]
        alpha_hat[t]
        posterior_var[t]

    Example:
        sched = DiffusionSchedule(T=500, schedule="cosine", device=device)
        betas = sched.betas
        alphas = sched.alphas
        alpha_hat = sched.alpha_hat
        posterior_var = sched.posterior_var
    """
    def __init__(self, T=500, schedule="cosine", device="cpu"):
        self.T = T
        self.device = device

        # Choose the schedule
        if schedule == "cosine":
            self.betas = cosine_beta_schedule(T).to(device)
        elif schedule == "linear":
            self.betas = linear_beta_schedule(T).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Compute alphas and cumulative alphas
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

        # Posterior variance β_tilde[t]
        self.posterior_var = torch.zeros_like(self.betas)
        self.posterior_var[0] = 1e-20
        self.posterior_var[1:] = (
            (1.0 - self.alpha_hat[:-1]) / (1.0 - self.alpha_hat[1:]) * self.betas[1:]
        )

    def to(self, device):
        """
        Move all tensors to another device.
        """
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        self.posterior_var = self.posterior_var.to(device)
        return self
