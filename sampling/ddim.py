# sampling/ddim.py

from __future__ import annotations
import torch
from torch import nn
from diffusion.base import DiffusionModel


class DDIM(DiffusionModel):
    """
    DDIM sampler (eta = 0): deterministic reverse diffusion.
    Allows subsampled timesteps such as 50, 20, 10, etc.
    """

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_scalar: int,
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        One DDIM reverse step:
            x_{t-1} = sqrt(a_hat_{t-1}) * x0_pred
                      + sqrt(1 - a_hat_{t-1}) * eps_pred
        """

        bsz = x_t.shape[0]
        device = x_t.device

        # Build batch of t values
        t = torch.full((bsz,), t_scalar, device=device, dtype=torch.long)

        # Predict epsilon (supports classifier-free guidance)
        eps_pred = self.guided_eps(model, x_t, t, labels, guidance_scale)

        # Schedules
        alpha_hat = self.schedule.alpha_hat  # cumulative product

        a_hat_t = alpha_hat[t_scalar]

        if t_scalar > 0:
            a_hat_prev = alpha_hat[t_scalar - 1]
        else:
            # Final step: t=0 -> x0
            a_hat_prev = torch.tensor(1.0, device=device)

        # Predict x0 using the forward diffusion algebra
        x0_pred = (
            x_t - torch.sqrt(1.0 - a_hat_t) * eps_pred
        ) / torch.sqrt(a_hat_t)

        # DDIM update (eta = 0)
        x_prev = (
            torch.sqrt(a_hat_prev) * x0_pred
            + torch.sqrt(1.0 - a_hat_prev) * eps_pred
        )

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        steps: int = 50,   # <-- number of DDIM steps
    ) -> torch.Tensor:
        """
        DDIM sampling with a subsampled timestep schedule.
        Example: T=1000, steps=50 → timesteps like [999, 979, 959, ..., 0].
        """

        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)

        # Move label tensor to device
        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.to(device, dtype=torch.long)
            else:
                labels = torch.tensor(labels, device=device, dtype=torch.long)

        # ----- Build subsampled DDIM timestep schedule -----
        # evenly spaced timesteps between T-1 down to 0
        # torch.linspace gives float → convert to long
        ts = (
            torch.linspace(self.T - 1, 0, steps, device=device)
            .long()
            .tolist()
        )

        # ----- Reverse diffusion loop -----
        for t_scalar in ts:
            x_t = self.p_sample(
                model,
                x_t,
                int(t_scalar),
                labels=labels,
                guidance_scale=guidance_scale,
            )

        return x_t.clamp(-1, 1)
