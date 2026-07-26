import numpy as np
import torch
from tsaug import Drift, Resize, TimeWarp

_time_warp = TimeWarp(n_speed_change=5, max_speed_ratio=3)
_drift = Drift(max_drift=0.02)


def augment(batch, proportion=0.1):
    b = batch.clone()
    b[b == 0] = np.nan
    augmented = []

    for sample in b.cpu().numpy():
        augmented.append([_perturb(sample[0], proportion)])

    max_len = max(len(s[0]) for s in b)
    padded = np.zeros((len(augmented), 1, max_len), dtype=np.float32)
    for i, s in enumerate(augmented):
        padded[i, 0, : len(s[0])] = s[0][:max_len]

    return torch.from_numpy(padded).to(batch.device)


def _perturb(sample, proportion):
    idx = np.where((~np.isnan(sample)) & (sample != -4200))[0]
    start = min(idx) if len(idx) else 0
    end = max(idx) if len(idx) else 0

    lead = sample[:start]
    values = np.array(sample[start:end])
    trail = sample[end:]

    if len(values) > 4:
        values = _time_warp.augment(values)
        values = _drift.augment(values)

    resize = Resize(max(1, int(round(len(values) * np.random.uniform(1 - proportion, 1 + proportion)))))
    values = resize.augment(values)

    return np.concatenate([lead, values, trail])
