import torch
import torch.nn as nn

from .time_embedding import sinusoidal_embedding


class MiniUNetV2(nn.Module):
    """Two-level mini U-Net."""

    def __init__(self, T, base_channels=32):
        super().__init__()

        self.base_channels = base_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels * 4),
            nn.SiLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(4, base_channels * 4),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, stride=2, padding=1
            ),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 4, stride=2, padding=1
            ),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )

        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, t):
        B = x.shape[0]

        temb = self.time_mlp(sinusoidal_embedding(t, 32))

        def add_time(h, ch):
            return h + temb[:, :ch].view(B, ch, 1, 1)

        h1 = self.enc1(x)
        h1 = add_time(h1, self.base_channels)

        h2 = self.down1(h1)
        h2 = add_time(h2, self.base_channels * 2)

        h3 = self.down2(h2)
        h3 = add_time(h3, self.base_channels * 4)

        mid = self.mid(h3)
        mid = add_time(mid, self.base_channels * 4)

        u2 = self.up2(mid)
        u2 = add_time(u2, self.base_channels * 2)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = add_time(u1, self.base_channels)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.dec1(u1)

        out = self.conv_out(u1)
        return out
