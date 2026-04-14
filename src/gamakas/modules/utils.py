import numpy as np
import pickle

def load_pitch(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    normalized_data = normalize(data)
    padded_data = zero_pad(normalized_data)
    return data, np.array(padded_data)

def normalize(data, range_min=-4200, range_max=4200):
    normalized_data = []
    for sample in data:
        normalized_sample = (sample - range_min) / (range_max - range_min)
        normalized_data.append(normalized_sample)
    return normalized_data

def zero_pad(data):
    max_length = max(len(sample) for sample in data)
    padded_data = []
    for sample in data:
        padded_sample = np.full((max_length,), 0.0, dtype=np.float32)
        padded_sample[:len(sample)] = sample
        padded_data.append(padded_sample)
    return np.array(padded_data)
