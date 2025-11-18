# sampling/ddpm.py

from __future__ import annotations
import torch
from torch import nn
from diffusion.base import DiffusionModel


class DDPM(DiffusionModel):
    """
    DDPM sampler: reverse diffusion using the original DDPM update rule.
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
        One DDPM reverse step:
            x_{t-1} ~ N(mean, sigma_t^2 I)
        """
        device = x_t.device
        t = torch.full((x_t.shape[0],), t_scalar, device=device, dtype=torch.long)

        eps_pred = self.guided_eps(model, x_t, t, labels, guidance_scale)

        alphas = self.schedule.alphas
        alpha_hat = self.schedule.alpha_hat
        posterior_var = self.schedule.posterior_var

        alpha_t = alphas[t_scalar]
        a_hat_t = alpha_hat[t_scalar]

        # DDPM mean:
        # mean = 1/sqrt(alpha_t) * (x_t - ((1 - alpha_t)/sqrt(1 - a_hat_t)) * eps_pred)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - a_hat_t)) * eps_pred
        )

        if t_scalar == 0:
            return mean

        sigma_t = torch.sqrt(posterior_var[t_scalar])
        return mean + sigma_t * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        labels: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Full DDPM sampling loop starting from x_T ~ N(0, I).
        """
        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)

        label_tensor = None
        if labels is not None:
            if torch.is_tensor(labels):
                label_tensor = labels.to(device, dtype=torch.long)
            else:
                label_tensor = torch.tensor(labels, device=device, dtype=torch.long)

        for t_scalar in reversed(range(self.T)):
            x_t = self.p_sample(
                model,
                x_t,
                t_scalar,
                labels=label_tensor,
                guidance_scale=guidance_scale,
            )

        return x_t.clamp(-1, 1)
