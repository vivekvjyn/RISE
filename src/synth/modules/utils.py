import numpy as np

RANGE_MIN = -2400
RANGE_MAX = 2400


def normalize(data, range_min=RANGE_MIN, range_max=RANGE_MAX):
    normalized_data = []
    for sample in data:
        normalized_sample = (sample - range_min) / (range_max - range_min)
        normalized_data.append(normalized_sample)
    return normalized_data


def denormalize(data, range_min=RANGE_MIN, range_max=RANGE_MAX):
    return np.asarray(data) * (range_max - range_min) + range_min


def zero_pad(data):
    max_length = max(len(sample) for sample in data)
    padded_data = []
    for sample in data:
        padded_sample = np.full((max_length,), 0.0, dtype=np.float32)
        padded_sample[:len(sample)] = sample
        padded_data.append(padded_sample)
    return np.array(padded_data)


def dtw(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m]


def dtw_aligned_distance(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    i, j = n, m
    aligned = []
    while i > 0 and j > 0:
        aligned.append(abs(seq1[i - 1] - seq2[j - 1]))
        prev = min((cost[i - 1, j], 0), (cost[i, j - 1], 1), (cost[i - 1, j - 1], 2), key=lambda x: x[0])
        if prev[1] == 0:
            i -= 1
        elif prev[1] == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return np.mean(aligned)


def harmonic_distance(seq1, seq2):
    return np.mean(np.abs(np.asarray(seq1) - np.asarray(seq2)))


def dtw_normalized(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        aligned.append(abs(seq1[i - 1] - seq2[j - 1]))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif cost[i - 1, j] < cost[i, j - 1]:
            if cost[i - 1, j] < cost[i - 1, j - 1]:
                i -= 1
            else:
                i -= 1
                j -= 1
        elif cost[i, j - 1] < cost[i - 1, j - 1]:
            j -= 1
        else:
            i -= 1
            j -= 1
    return np.mean(aligned)


def dft_oscillation_count(signal):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum[0] = 0
    k = np.argmax(spectrum)
    if k == 0 or k == len(spectrum) - 1:
        return float(k)
    alpha = spectrum[k - 1]
    beta = spectrum[k]
    gamma = spectrum[k + 1]
    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    return float(k + p)


def dft_pitch_position(signal):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < 1:
        return 0.0
    return float(np.median(signal))


def slope_difference(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:n], seq2[:n]
    valid = ~np.isnan(seq1) & ~np.isnan(seq2)
    if valid.sum() < 2:
        return 0.0
    s1, s2 = seq1[valid], seq2[valid]
    slope1 = np.diff(s1)
    slope2 = np.diff(s2)
    return float(np.mean(np.abs(slope1 - slope2)))


def linear_regression(seq, timestep=0.0100006335037896):
    seq = np.asarray(seq, dtype=np.float64).flatten()
    valid = ~np.isnan(seq)
    if valid.sum() < 2:
        return 0.0, 0.0
    x = np.where(valid)[0].astype(np.float64) * timestep
    y = seq[valid]
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1])


def linear_regression_difference(seq1, seq2, timestep=0.0100006335037896):
    slope1, intercept1 = linear_regression(seq1, timestep)
    slope2, intercept2 = linear_regression(seq2, timestep)
    return abs(slope1 - slope2), abs(intercept1 - intercept2)
