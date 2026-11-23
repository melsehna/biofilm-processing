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


_threadLimiter = None


def limitThreads():
    """Pool initializer: one compute thread per worker process.

    Parallelism comes from the process pool, so every per-process pool is pure
    oversubscription. OpenCV, OpenBLAS and OpenMP otherwise size their pools to
    every core on the box (56 here), so N workers bursting together demand
    N x 56 threads and stall the whole machine. With this, N workers use N
    cores. Thread count does not change OpenCV's results. The package
    `__init__` also exports OMP/OPENBLAS/MKL_NUM_THREADS=1 before numpy loads;
    threadpoolctl here caps the BLAS/OpenMP pools even when numpy was loaded
    first (e.g. by a caller that imported numpy before this package). libx264
    is capped via overlay.py.
    """
    global _threadLimiter
    import cv2
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _threadLimiter = threadpool_limits(limits=1)   # kept referenced: applies for the process lifetime
