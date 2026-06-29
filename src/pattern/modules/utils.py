import numpy as np
import pickle

def load_pitch(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    normalized_data = normalize(data)
    return normalized_data

def normalize(data, range_min=-4200, range_max=4200):
    normalized_data = []
    for sample in data:
        normalized_sample = (sample - range_min) / (range_max - range_min)
        normalized_data.append(normalized_sample)
    return normalized_data
