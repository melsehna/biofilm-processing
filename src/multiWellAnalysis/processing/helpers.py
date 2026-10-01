import numpy as np

def roundOdd(x):
    x = int(round(x))
    return x if x % 2 else x + 1

def compmax(x):
    return np.max(x) if len(x) > 1 else 0

def calculateStats(crosscorMaxima, sourceFreq, targetFreq):
    sourceAmp = np.mean(np.abs(sourceFreq) ** 2)
    targetAmp = np.mean(np.abs(targetFreq) ** 2)
    error = 1 - (np.abs(crosscorMaxima) ** 2) / (sourceAmp * targetAmp)
    phasediff = np.angle(crosscorMaxima)
    return error, phasediff


def limitThreads():
    """Pool initializer: one compute thread per worker process.

    Parallelism comes from the process pool, so every per-process pool is pure
    oversubscription. OpenCV otherwise sizes its pool to every core on the box
    (56 here), so N workers bursting together demand N x 56 threads and stall
    the whole machine. Thread count does not change OpenCV's results. BLAS/OMP
    are pinned via env vars by the CLI entry points; libx264 via overlay.py.
    """
    import cv2
    cv2.setNumThreads(1)
