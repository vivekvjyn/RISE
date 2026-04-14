import numpy as np
from sklearn.metrics import f1_score
from torch.nn import functional as F

class Meter:
    def __init__(self):
        self._loss = 0.0
        self._pred = np.array([], dtype=np.int64)
        self._true = np.array([], dtype=np.int64)

    def __call__(self, logits, targets):
        self._loss += F.cross_entropy(logits, targets).item()
        self._pred = np.concatenate((self._pred, logits.argmax(dim=1).cpu().numpy()))
        self._true = np.concatenate((self._true, targets.cpu().numpy()))

    @property
    def loss(self):
        return self._loss / len(self._true)

    @property
    def f1_score(self):
        return f1_score(self._true, self._pred, average='macro')
