import torch.nn as nn

from .time_embedding import sinusoidal_embedding


class TinyConv(nn.Module):
    """Small conv net used as epsilon predictor."""

    def __init__(self, T):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_embedding(t, 32))
        temb = temb[:, :, None, None]
        h = self.conv1(x)
        h = self.act(self.norm1(h + temb))
        h = self.act(self.norm2(self.conv2(h) + temb))
        return self.conv3(h)
