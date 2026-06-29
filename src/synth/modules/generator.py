import numpy as np
import torch

class Generator:
    def __init__(self, model, logger, T):
        self.model = model
        self.logger = logger
        self.T = T

    def __call__(self, data_loader):
        outputs = self._propagate(data_loader)
        return outputs

    def _propagate(self, data_loader):
        self.model.eval()
        device = next(self.model.parameters()).device

        outputs = np.array([])
        for i, input in enumerate(data_loader):
            self.logger.pbar(i + 1, len(data_loader))

            x = input[0].to(device) if isinstance(input, (list, tuple)) else input.to(device)
            output = self.model(x, self.T).detach().cpu().numpy()
            outputs = np.concatenate((outputs, output), axis=0) if outputs.size else output

        return outputs
