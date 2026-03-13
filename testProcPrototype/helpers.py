import numpy as np
from numpy import pi

def round_odd(x):
    x = int(round(x))
    return x if x % 2 else x + 1

def compmax(x):
    return np.max(x) if len(x) > 1 else 0

def calculate_stats(crosscor_maxima, source_freq, target_freq):
    source_amp = np.mean(np.abs(source_freq) ** 2)
    target_amp = np.mean(np.abs(target_freq) ** 2)
    error = 1 - (np.abs(crosscor_maxima) ** 2) / (source_amp * target_amp)
    phasediff = np.angle(crosscor_maxima)
    return error, phasediff
