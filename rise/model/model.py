import os
import torch
import torch.nn as nn
from rise.model.inception import Encoder

class Model(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_features=2):
        super().__init__()
        self.encoder = Encoder(embed_dim, depth, num_features)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, input):
        embedding = self.encoder(input)
        embedding = self.gap(embedding).squeeze(-1)
        return self.fully_connected(embedding)

    def save(self, path):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
