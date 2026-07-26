import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


def smooth_pitch_curve(time_series, pitch_series, smoothing_factor=0.6, min_points=4):
    time_series = np.array(time_series, dtype=float)
    pitch_series = np.array(pitch_series, dtype=float)
    smoothed = np.full_like(pitch_series, np.nan)
    valid_mask = ~pd.isna(time_series) & ~pd.isna(pitch_series)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return smoothed

    chunks = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)

    for chunk in chunks:
        if len(chunk) >= min_points:
            t = time_series[chunk]
            p = pitch_series[chunk]
            p_min, p_max = np.min(p), np.max(p)
            norm = (p - p_min) / (p_max - p_min)
            spline = UnivariateSpline(t, norm, s=smoothing_factor)
            smoothed[chunk] = spline(t) * (p_max - p_min) + p_min
        elif len(chunk) > 1:
            t = time_series[chunk]
            smoothed[chunk] = np.interp(t, t, pitch_series[chunk])

    return smoothed


def interpolate(arr, val, gap, indices=[]):
    s = np.copy(arr)
    indices = set(indices)
    if np.isnan(val):
        is_gap = np.isnan(s)
    else:
        is_gap = s == val
    in_gap = False
    gap_start = None
    gap_ranges = []

    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            gap_start = i
        elif not g and in_gap:
            in_gap = False
            gap_ranges.append((gap_start, i))

    if in_gap:
        gap_ranges.append((gap_start, len(s)))

    for start, end in gap_ranges:
        if end - start > gap:
            continue
        if any(idx in indices for idx in range(start, end)):
            continue
        s[start:end] = np.nan

    return pd.Series(s).interpolate(method="linear").ffill().bfill().values


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
    alpha, beta, gamma = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    return float(k + p)


def dft_pitch_position(signal):
    signal = np.asarray(signal, dtype=np.float64).flatten()
    signal = signal[~np.isnan(signal)]
    if len(signal) < 1:
        return 0.0
    return float(np.median(signal))
