import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from .inception import InceptionEncoder
from .attention import Attention


def apply_lora(model, r=8, alpha=16, dropout=0.05):
    targets = [name for name, m in model.named_modules() if isinstance(m, nn.Conv1d)]
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets))


class Model(nn.Module):
    def __init__(self, task, embed_dim, depth, num_features=2, out_dim=16, num_classes=2, dropout=0.2):
        super().__init__()
        self.task = task
        self.encoder = InceptionEncoder(embed_dim, depth, num_features)

        if task == "ssl":
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.projector = nn.Sequential(
                nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                nn.BatchNorm1d(embed_dim), nn.Linear(embed_dim, out_dim),
            )
        elif task in ("classification", "clustering"):
            self.prec_encoder = InceptionEncoder(embed_dim, depth, num_features)
            self.curr_encoder = InceptionEncoder(embed_dim, depth, num_features)
            self.succ_encoder = InceptionEncoder(embed_dim, depth, num_features)
            self.prec_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.curr_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.succ_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.attention = Attention(embed_dim)
            self.head = nn.Sequential(
                nn.BatchNorm1d(embed_dim * 3), nn.Dropout(dropout),
                nn.Linear(embed_dim * 3, embed_dim), nn.ReLU(),
                nn.BatchNorm1d(embed_dim), nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        elif task == "pattern":
            self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, *args):
        if self.task == "ssl":
            return self.projector(self.gap(self.encoder(args[0])).squeeze(-1))
        elif self.task in ("classification", "clustering"):
            return self._forward_triplet(args[0], args[1], args[2])
        elif self.task == "pattern":
            return F.normalize(self.gap(self.encoder(args[0])), p=2, dim=1).squeeze(-1)
        elif self.task == "synth":
            return self.encoder(args[0])

    def _encode_triplet(self, prec, curr, succ):
        def _run(x, enc, gru):
            h, _ = gru(enc(x).permute(0, 2, 1))
            return h
        ph = _run(prec, self.prec_encoder, self.prec_gru)
        ch = _run(curr, self.curr_encoder, self.curr_gru)
        sh = _run(succ, self.succ_encoder, self.succ_gru)
        ch_last = ch[:, -1:]
        return torch.cat([
            self.attention(ph, ch_last),
            self.attention(ch, ch_last),
            self.attention(sh, ch_last),
        ], dim=1)

    def _forward_triplet(self, prec, curr, succ):
        return self.head(self._encode_triplet(prec, curr, succ))

    def encode(self, prec, curr, succ):
        return self._encode_triplet(prec, curr, succ)

    def apply_lora(self, r=4, alpha=16, dropout=0.0):
        self.prec_encoder = apply_lora(self.prec_encoder, r, alpha, dropout)
        self.curr_encoder = apply_lora(self.curr_encoder, r, alpha, dropout)
        self.succ_encoder = apply_lora(self.succ_encoder, r, alpha, dropout)

    def set_encoders_trainable(self, trainable=True):
        for enc in [self.prec_encoder, self.curr_encoder, self.succ_encoder]:
            for p in enc.parameters():
                p.requires_grad = trainable

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
