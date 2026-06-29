import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposeInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=9, padding=4)
        self.branch2 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=19, padding=9)
        self.branch3 = nn.ConvTranspose1d(in_channels, out_channels // 3, kernel_size=39, padding=19)
        self.residual = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1) + self.residual(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, depth, num_features=2):
        super().__init__()
        self.dir = "checkpoints"
        self.depth = depth

        self.blocks = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = embed_dim * (2 ** i)
            out_ch = embed_dim * (2 ** (i + 1))
            self.blocks.append(
                nn.Sequential(
                    TransposeInceptionModule(in_ch, out_ch),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                )
            )

        in_ch = embed_dim * (2 ** (depth - 1))
        self.pitch_head = nn.Sequential(
            nn.Linear(in_ch, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Linear(in_ch // 2, 1),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x, T):
        for i in range(self.depth - 1):
            x = self.blocks[i](x)
            target_T = T // (2 ** (self.depth - 2 - i))
            x = F.interpolate(x, size=target_T, mode="linear", align_corners=False)
        x = x.permute(0, 2, 1)
        pitch = self.pitch_head(x)
        mask = torch.sigmoid(self.mask_head(x))
        out = torch.cat([pitch, mask], dim=-1)
        out = out.permute(0, 2, 1)
        return out

    def load(self, filename, device):
        self.load_state_dict(torch.load(
            os.path.join(self.dir, filename), map_location=device
        ))

    def save(self, filename):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.dir, filename))
