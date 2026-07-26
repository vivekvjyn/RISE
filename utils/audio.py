import os
import pickle
import numpy as np


def normalize(data, range_min=-4200, range_max=4200):
    return [(sample - range_min) / (range_max - range_min) for sample in data]


def denormalize(data, range_min=-2400, range_max=2400):
    return np.asarray(data) * (range_max - range_min) + range_min


def zero_pad(data):
    max_length = max(len(sample) for sample in data)
    padded = np.zeros((len(data), max_length), dtype=np.float32)
    for i, sample in enumerate(data):
        padded[i, : len(sample)] = sample
    return padded


def load_pitch(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return zero_pad(normalize(data))


def load_pitch_raw(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data, np.array(zero_pad(normalize(data)))


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
