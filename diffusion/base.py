# diffusion/base.py

from __future__ import annotations
import torch
from torch import nn
from schedules.beta_schedules import DiffusionSchedule


class DiffusionModel:
    """
    Generic diffusion model wrapper.

    Holds:
      - schedule (betas, alphas, alpha_hat, etc.)
      - forward diffusion q(x_t | x_0)
      - timestep sampling
      - (optional) classifier-free guidance utilities
    """

    def __init__(self, schedule: DiffusionSchedule, null_label: int | None = None):
        self.schedule = schedule
        self.null_label = null_label

    @property
    def T(self) -> int:
        return self.schedule.T

    # ---------- Forward diffusion (q) ----------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_t = sqrt(alpha_hat[t]) * x0 + sqrt(1 - alpha_hat[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_hat = self.schedule.alpha_hat
        s1 = torch.sqrt(alpha_hat[t])[:, None, None, None]
        s2 = torch.sqrt(1.0 - alpha_hat[t])[:, None, None, None]
        return s1 * x0 + s2 * noise

    def rand_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """
        Sample random timesteps uniformly from {0, ..., T-1}.
        """
        return torch.randint(0, self.T, (batch_size,), device=device)

    # ---------- Epsilon prediction & CFG ----------

    def predict_eps(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Call the underlying neural net ε_θ(x_t, t, y).
        """
        return model(x_t, t, labels)

    def make_uncond_labels(self, batch_size: int, device: torch.device | str) -> torch.Tensor | None:
        """
        Create a 'null' label vector for classifier-free guidance, if configured.
        """
        if self.null_label is None:
            return None
        return torch.full(
            (batch_size,),
            self.null_label,
            device=device,
            dtype=torch.long,
        )

    def guided_eps(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor | None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Classifier-free guidance:

            eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)

        If null_label is None or guidance_scale == 1, falls back to plain conditional.
        """
        # No CFG configured → just return conditional prediction
        if self.null_label is None or guidance_scale == 1.0:
            return self.predict_eps(model, x_t, t, labels)

        # Unconditional labels
        uncond_labels = self.make_uncond_labels(x_t.shape[0], x_t.device)
        eps_uncond = self.predict_eps(model, x_t, t, uncond_labels)

        if labels is None:
            return eps_uncond

        # If all labels are NULL_CLASS (dropped), just use unconditional branch
        if torch.all(labels == self.null_label):
            return eps_uncond

        eps_cond = self.predict_eps(model, x_t, t, labels)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
