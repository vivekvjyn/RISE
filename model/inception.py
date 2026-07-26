import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=9, padding=4)
        self.branch2 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=19, padding=9)
        self.branch3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=39, padding=19)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1) + self.residual(x)


class TransposeInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=9, padding=4)
        self.branch2 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=19, padding=9)
        self.branch3 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=39, padding=19)
        self.residual = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1) + self.residual(x)


class InceptionEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_features):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                InceptionBlock(num_features, embed_dim * 2 ** (depth - 1)),
                nn.ReLU(),
                nn.BatchNorm1d(embed_dim * 2 ** (depth - 1)),
                nn.AvgPool1d(kernel_size=2, stride=2),
            )
        )
        for i in range(depth - 1, 1, -1):
            self.blocks.append(
                nn.Sequential(
                    InceptionBlock(embed_dim * 2 ** i, embed_dim * 2 ** (i - 1)),
                    nn.ReLU(),
                    nn.BatchNorm1d(embed_dim * 2 ** (i - 1)),
                    nn.AvgPool1d(kernel_size=2, stride=2),
                )
            )
        self.blocks.append(InceptionBlock(embed_dim * 2, embed_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class InceptionDecoder(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = embed_dim * (2 ** i)
            out_ch = embed_dim * (2 ** (i + 1))
            self.blocks.append(
                nn.Sequential(TransposeInceptionBlock(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU())
            )
        in_ch = embed_dim * (2 ** (depth - 1))
        self.pitch_head = nn.Sequential(
            nn.Linear(in_ch, in_ch), nn.ReLU(),
            nn.Linear(in_ch, in_ch // 2), nn.ReLU(),
            nn.Linear(in_ch // 2, 1),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2), nn.ReLU(),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x, T):
        for i in range(self.depth - 1):
            x = self.blocks[i](x)
            x = nn.functional.interpolate(x, size=T // (2 ** (self.depth - 2 - i)), mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)
        pitch = self.pitch_head(x)
        mask = torch.sigmoid(self.mask_head(x))
        return torch.cat([pitch, mask], dim=-1).permute(0, 2, 1)
