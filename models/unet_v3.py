import torch
import torch.nn as nn
from .time_embedding import sinusoidal_embedding

# -------------------------------
# 2. ResNet block + UNet v3
# -------------------------------

class ResBlock(nn.Module):
    """
    Standard diffusion-style residual block:
    - Conv -> GN -> SiLU
    - Add time embedding as bias
    - Conv -> GN -> SiLU
    - Residual connection (with 1x1 if channels change)
    """

    def __init__(self, in_ch, out_ch, temb_dim, groups=8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # use min(groups, out_ch) to avoid invalid group count
        g1 = min(groups, out_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)

        self.act = nn.SiLU()

        # project time embedding to out_ch, added after first conv
        self.time_proj = nn.Linear(temb_dim, out_ch)

        # 1x1 conv if we need to change channels in residual path
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        """
        x: [B, in_ch, H, W]
        temb: [B, temb_dim]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # add time embedding as bias
        temb_bias = self.time_proj(temb)  # [B, out_ch]
        h = h + temb_bias[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return self.shortcut(x) + h


class UNetV3(nn.Module):
    """
    A small but "real" diffusion UNet:

    Resolution path:
      28x28 (64 ch) -> 14x14 (128 ch) -> 7x7 (256 ch) -> 14x14 -> 28x28

    Time conditioning:
      t -> sinusoidal(32) -> MLP -> temb_dim (128)
      temb injected into every ResBlock.
    """

    def __init__(self, T, base_channels=64, temb_dim=128):
        super().__init__()
        self.T = T
        self.base_channels = base_channels
        self.temb_dim = temb_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(32, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # ---- Encoder ----
        # initial 28x28 conv
        self.conv_in = nn.Conv2d(1, base_channels, 3, padding=1)

        # 28x28 block(s)
        self.enc_block1_1 = ResBlock(base_channels, base_channels, temb_dim)
        self.enc_block1_2 = ResBlock(base_channels, base_channels, temb_dim)

        # downsample to 14x14
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        # 14x14 block(s)
        self.enc_block2_1 = ResBlock(base_channels * 2, base_channels * 2, temb_dim)
        self.enc_block2_2 = ResBlock(base_channels * 2, base_channels * 2, temb_dim)

        # downsample to 7x7
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        # ---- Bottleneck ----
        self.mid_block1 = ResBlock(base_channels * 4, base_channels * 4, temb_dim)
        self.mid_block2 = ResBlock(base_channels * 4, base_channels * 4, temb_dim)

        # ---- Decoder ----
        # 7x7 -> 14x14
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=4, stride=2, padding=1)

        # after concat with skip from enc_block2: channels = base*4
        self.dec_block2_1 = ResBlock(base_channels * 4, base_channels * 2, temb_dim)
        self.dec_block2_2 = ResBlock(base_channels * 2, base_channels * 2, temb_dim)

        # 14x14 -> 28x28
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=4, stride=2, padding=1)

        # after concat with skip from enc_block1: channels = base*2
        self.dec_block1_1 = ResBlock(base_channels * 2, base_channels, temb_dim)
        self.dec_block1_2 = ResBlock(base_channels, base_channels, temb_dim)

        # final output conv (predict epsilon)
        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, t):
        """
        x: [B, 1, 28, 28]
        t: [B] (int64 or float)
        returns: epsilon prediction, [B, 1, 28, 28]
        """
        # time embedding
        t_emb = sinusoidal_embedding(t, dim=32)         # [B, 32]
        t_emb = self.time_mlp(t_emb)                    # [B, temb_dim]

        # ----- Encoder -----
        # 28x28
        x0 = self.conv_in(x)                            # [B, 64, 28, 28]
        x1 = self.enc_block1_1(x0, t_emb)               # [B, 64, 28, 28]
        x1 = self.enc_block1_2(x1, t_emb)               # [B, 64, 28, 28]
        skip1 = x1

        # 28x28 -> 14x14
        x2 = self.down1(x1)                             # [B, 128, 14, 14]
        x2 = self.enc_block2_1(x2, t_emb)               # [B, 128, 14, 14]
        x2 = self.enc_block2_2(x2, t_emb)               # [B, 128, 14, 14]
        skip2 = x2

        # 14x14 -> 7x7
        x3 = self.down2(x2)                             # [B, 256, 7, 7]

        # ----- Bottleneck -----
        x3 = self.mid_block1(x3, t_emb)                 # [B, 256, 7, 7]
        x3 = self.mid_block2(x3, t_emb)                 # [B, 256, 7, 7]

        # ----- Decoder -----
        # 7x7 -> 14x14
        u2 = self.up2(x3)                               # [B, 128, 14, 14]
        u2 = torch.cat([u2, skip2], dim=1)              # [B, 256, 14, 14]
        u2 = self.dec_block2_1(u2, t_emb)               # [B, 128, 14, 14]
        u2 = self.dec_block2_2(u2, t_emb)               # [B, 128, 14, 14]

        # 14x14 -> 28x28
        u1 = self.up1(u2)                               # [B, 64, 28, 28]
        u1 = torch.cat([u1, skip1], dim=1)              # [B, 128, 28, 28]
        u1 = self.dec_block1_1(u1, t_emb)               # [B, 64, 28, 28]
        u1 = self.dec_block1_2(u1, t_emb)               # [B, 64, 28, 28]

        out = self.conv_out(u1)                         # [B, 1, 28, 28]
        return out

