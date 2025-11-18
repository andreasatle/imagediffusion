# training/losses.py

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from diffusion.base import DiffusionModel


def eps_mse_loss(
    model: nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    diffusion: DiffusionModel,
) -> torch.Tensor:
    """
    Standard DDPM training loss:

        E[ || eps - eps_theta(x_t, t, y) ||^2 ]

    where x_t = sqrt(alpha_hat[t]) x0 + sqrt(1 - alpha_hat[t]) eps.
    """
    B = x0.shape[0]
    device = x0.device

    t = diffusion.rand_timesteps(B, device)
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)

    eps_pred = diffusion.predict_eps(model, x_t, t, labels)
    return F.mse_loss(eps_pred, noise)
