import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps diffusion timesteps to sinusoidal embeddings.

    Input:
        t: (B,)

    Output:
        (B, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )

        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )

        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):
    """
    Convolutional residual block conditioned on a timestep embedding.
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        self.activation = nn.SiLU()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(self, x, time_embedding):
        residual = self.shortcut(x)

        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        # Inject timestep information
        time_features = self.time_projection(time_embedding)
        time_features = time_features[:, :, None, None]

        h = h + time_features

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)

        # Residual connection
        return self.activation(h + residual)


class UNet(nn.Module):
    """
    U-Net for DDPM noise prediction.

    Args:
        in_channels: Number of image channels.
        out_channels: Number of predicted noise channels.
        base_channels: Number of channels in the first encoder stage.
        time_dim: Dimension of the timestep embedding.

    Inputs:
        x: (B, in_channels, H, W)
        t: (B,) diffusion timesteps.

    Output:
        (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_dim=128,
    ):
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError("base_channels must be divisible by 8.")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Input convolution
        self.input_conv = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=3,
            padding=1,
        )

        # Encoder
        self.enc1 = ResidualBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(
            c1, c2, kernel_size=4, stride=2, padding=1
        )

        self.enc2 = ResidualBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(
            c2, c3, kernel_size=4, stride=2, padding=1
        )

        self.enc3 = ResidualBlock(c3, c3, time_dim)
        self.down3 = nn.Conv2d(
            c3, c4, kernel_size=4, stride=2, padding=1
        )

        # Bottleneck
        self.mid1 = ResidualBlock(c4, c4, time_dim)
        self.mid2 = ResidualBlock(c4, c4, time_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            c4, c3, kernel_size=4, stride=2, padding=1
        )
        self.dec3 = ResidualBlock(c3 * 2, c3, time_dim)

        self.up2 = nn.ConvTranspose2d(
            c3, c2, kernel_size=4, stride=2, padding=1
        )
        self.dec2 = ResidualBlock(c2 * 2, c2, time_dim)

        self.up1 = nn.ConvTranspose2d(
            c2, c1, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = ResidualBlock(c1 * 2, c1, time_dim)

        # Output convolution
        self.output_norm = nn.GroupNorm(8, c1)
        self.output_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, t):
        # Embed timesteps
        time_embedding = self.time_mlp(t)

        # Encoder
        x1 = self.input_conv(x)
        x1 = self.enc1(x1, time_embedding)

        x2 = self.down1(x1)
        x2 = self.enc2(x2, time_embedding)

        x3 = self.down2(x2)
        x3 = self.enc3(x3, time_embedding)

        x4 = self.down3(x3)

        # Bottleneck
        x4 = self.mid1(x4, time_embedding)
        x4 = self.mid2(x4, time_embedding)

        # Decoder: concatenate skip connections
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, time_embedding)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, time_embedding)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, time_embedding)

        x = self.output_norm(x)
        x = F.silu(x)

        return self.output_conv(x)