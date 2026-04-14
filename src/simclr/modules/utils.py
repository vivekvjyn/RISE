import numpy as np

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
