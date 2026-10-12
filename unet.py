import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (Contracting path)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path (Expanding path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder alignment
        skip_connections = skip_connections[::-1]

        # Upward decoder pass
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d
            skip_connection = skip_connections[idx // 2]

            # Handle potential size mismatches (odd spatial dimensions)
            if x.shape != skip_connection.shape:
                diff_y = skip_connection.size(2) - x.size(2)
                diff_x = skip_connection.size(3) - x.size(3)
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                         diff_y // 2, diff_y - diff_y // 2])

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)  # DoubleConv

        return self.final_conv(x)


# Example usage test
if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    output = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)