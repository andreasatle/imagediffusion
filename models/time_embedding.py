import math
import torch


def sinusoidal_embedding(t, dim=32):
    """Return sinusoidal timestep embedding of size `dim`."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
