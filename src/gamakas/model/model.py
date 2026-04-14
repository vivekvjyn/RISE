import os
import torch
import torch.nn as nn
from gamakas.model.inception import Encoder
from gamakas.model.lora import apply_lora
from gamakas.model.attention import Attention

class Model(nn.Module):
    def __init__(self, embed_dim, num_classes, depth, num_features=2, dropout=0.2):
        super().__init__()
        self.dir = os.path.join("checkpoints")

        self.prec_encoder = Encoder(embed_dim, depth, num_features)
        self.curr_encoder = Encoder(embed_dim, depth, num_features)
        self.succ_encoder = Encoder(embed_dim, depth, num_features)

        self.prec_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.curr_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.succ_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        self.attention = Attention(embed_dim)

        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, prec, curr, succ):
        embedding = self.encode(prec, curr, succ)
        return self.fully_connected(embedding)

    def encode(self, prec, curr, succ):
        prec_encoding = self.prec_encoder(prec)
        curr_encoding = self.curr_encoder(curr)
        succ_encoding = self.succ_encoder(succ)
        prec_encoding, prec_hidden = self.prec_gru(prec_encoding.permute(0, 2, 1))
        curr_encoding, curr_hidden = self.curr_gru(curr_encoding.permute(0, 2, 1))
        succ_encoding, succ_hidden = self.succ_gru(succ_encoding.permute(0, 2, 1))
        prec_context = self.attention(prec_encoding, curr_hidden[-1:])
        curr_context = self.attention(curr_encoding, curr_hidden[-1:])
        succ_context = self.attention(succ_encoding, curr_hidden[-1:])
        embedding = torch.cat([prec_context, curr_context, succ_context], dim=1)
        return embedding

    def save(self, filename):
        os.makedirs(self.dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.dir, filename))

    def load(self, filename, device):
        self.load_state_dict(torch.load(os.path.join(self.dir, filename), map_location=device))

    def apply_lora(self, r=4, alpha=16, dropout=0.0):
        self.prec_encoder = apply_lora(self.prec_encoder, r, alpha, dropout)
        self.curr_encoder = apply_lora(self.curr_encoder, r, alpha, dropout)
        self.succ_encoder = apply_lora(self.succ_encoder, r, alpha, dropout)

    def freeze_encoders(self):
        for param in self.prec_encoder.parameters():
            param.requires_grad = False
        for param in self.curr_encoder.parameters():
            param.requires_grad = False
        for param in self.succ_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        for param in self.prec_encoder.parameters():
            param.requires_grad = True
        for param in self.curr_encoder.parameters():
            param.requires_grad = True
        for param in self.succ_encoder.parameters():
            param.requires_grad = True

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
