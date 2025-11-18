import torch.nn as nn

from .time_embedding import sinusoidal_embedding


class TinyConv(nn.Module):
    """Small conv net used as epsilon predictor."""

    def __init__(self, T):
        super().__init__()
        self.temb_dim = 128
        self.null_class = 10
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.class_emb = nn.Embedding(self.null_class + 1, self.temb_dim)
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t, y=None):
        temb = self.time_mlp(sinusoidal_embedding(t, 32))
        if y is not None:
            y_long = y.long().clamp(0, self.null_class)
            mask = (y_long != self.null_class).float().unsqueeze(1)
            temb = temb + self.class_emb(y_long) * mask
        temb = temb[:, :, None, None]
        h = self.conv1(x)
        h = self.act(self.norm1(h + temb))
        h = self.act(self.norm2(self.conv2(h) + temb))
        return self.conv3(h)
