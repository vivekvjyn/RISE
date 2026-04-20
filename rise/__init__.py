import os
import random
import numpy as np
import torch

from rise.model.model import Model
from rise.model.inception import Encoder

__all__ = ["Model", "Encoder"]

class RISE:
    def __init__(self, device):
        self.model = Model().to(device)
        self.logger = logger

    def __call__(self, data_loader):
        embeddings = self._propagate(data_loader)

        return embeddings

    def _propagate(self, data_loader):
        self.model.eval()

        embeddings = np.array([])
        for i, (prec, curr, succ, target) in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            embedding = self._predict(prec, curr, succ).detach().cpu().numpy()
            embeddings = np.concatenate((embeddings, embedding), axis=0) if embeddings.size else embedding

        return embeddings

    def _predict(self, prec, curr, succ):
        prec_mask = (torch.isnan(prec)).float()
        prec = torch.nan_to_num(prec, nan=0.0)
        prec_input = torch.cat([prec, prec_mask], dim=1)

        curr_mask = (torch.isnan(curr)).float()
        curr = torch.nan_to_num(curr, nan=0.0)
        curr_input = torch.cat([curr, curr_mask], dim=1)

        succ_mask = (torch.isnan(succ)).float()
        succ = torch.nan_to_num(succ, nan=0.0)
        succ_input = torch.cat([succ, succ_mask], dim=1)

        return self.model.encode(prec_input, curr_input, succ_input)

    def _load_weights(self, path):
        model.encoder.load(path, self.model.device())
