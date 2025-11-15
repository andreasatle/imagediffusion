import torch
import torch.nn as nn

from .time_embedding import sinusoidal_embedding


class MiniUNet(nn.Module):
    """One downsample/up mini U-Net."""

    def __init__(self, T, base_channels=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        self.conv_in = nn.Conv2d(1, base_channels, 3, padding=1)

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )

        self.downsample = nn.Conv2d(
            base_channels * 2, base_channels * 2, 4, stride=2, padding=1
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )

        self.upsample = nn.ConvTranspose2d(
            base_channels * 2, base_channels * 2, 4, stride=2, padding=1
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )

        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, 32)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb[:, :, None, None]

        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x2 = x2 + t_emb[:, : x2.shape[1]]

        x3 = self.downsample(x2)

        mid = self.mid(x3)
        mid = mid + t_emb[:, : mid.shape[1]]

        u = self.upsample(mid)

        u = torch.cat([u, x2], dim=1)

        u = self.up1(u)

        out = self.conv_out(u)
        return out
