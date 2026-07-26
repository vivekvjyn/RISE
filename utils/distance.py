import numpy as np


def _dtw_matrix(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost


def _dtw_align(cost, seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    i, j = n, m
    aligned = []
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
    return aligned


def dtw(seq1, seq2):
    return _dtw_matrix(seq1, seq2)[-1, -1]


def dtw_normalized(seq1, seq2):
    aligned = _dtw_align(_dtw_matrix(seq1, seq2), seq1, seq2)
    return np.mean(aligned)


def dtw_aligned_distance(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n, m = len(seq1), len(seq2)
    cost = _dtw_matrix(seq1, seq2)
    i, j = n, m
    aligned = []
    while i > 0 and j > 0:
        aligned.append(abs(seq1[i - 1] - seq2[j - 1]))
        prev = min(
            (cost[i - 1, j], 0),
            (cost[i, j - 1], 1),
            (cost[i - 1, j - 1], 2),
            key=lambda x: x[0],
        )
        if prev[1] == 0:
            i -= 1
        elif prev[1] == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return np.mean(aligned)


def slope_difference(seq1, seq2):
    seq1 = np.asarray(seq1, dtype=np.float64).flatten()
    seq2 = np.asarray(seq2, dtype=np.float64).flatten()
    n = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:n], seq2[:n]
    valid = ~np.isnan(seq1) & ~np.isnan(seq2)
    if valid.sum() < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(seq1[valid]) - np.diff(seq2[valid]))))


def linear_regression(seq, timestep=0.0100006335037896):
    seq = np.asarray(seq, dtype=np.float64).flatten()
    valid = ~np.isnan(seq)
    if valid.sum() < 2:
        return 0.0, 0.0
    x = np.where(valid)[0].astype(np.float64) * timestep
    coeffs = np.polyfit(x, seq[valid], 1)
    return float(coeffs[0]), float(coeffs[1])


def linear_regression_difference(seq1, seq2, timestep=0.0100006335037896):
    slope1, intercept1 = linear_regression(seq1, timestep)
    slope2, intercept2 = linear_regression(seq2, timestep)
    return abs(slope1 - slope2), abs(intercept1 - intercept2)
