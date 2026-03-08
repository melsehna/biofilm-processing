import numpy as np
import pandas as pd
import tifffile
import os
import time
from datetime import datetime


def getLogRoot():
    return os.environ.get('MWA_LOG_ROOT', '/mnt/data/logs')


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def loadRawStack(path):
    return tifffile.imread(path)


def loadMaskStack(path):
    npz = np.load(path)
    return npz[npz.files[0]]


def extractFrame(stack, frameIdx):
    return stack[:, :, frameIdx]


def ensureDir(path):
    os.makedirs(path, exist_ok=True)


def appendCsv(path, df):
    writeHeader = not os.path.exists(path)
    df.to_csv(path, mode='a', header=writeHeader, index=False)


class Timer:
    def __init__(self):
        self.startTime = time.time()

    def elapsed(self):
        return time.time() - self.startTime

    def reset(self):
        self.startTime = time.time()


def loadProcessedWells(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f)


def markWellProcessed(path, plateId, wellId):
    with open(path, 'a') as f:
        f.write(f'{plateId},{wellId}\n')


def plateDir(plateId):
    root = getLogRoot()
    path = os.path.join(root, str(plateId))
    ensureDir(path)
    return path


def plateLogPath(plateId, suffix=''):
    suffix = suffix.lstrip('_')
    filename = f'{plateId}_{suffix}.log' if suffix else f'{plateId}.log'
    return os.path.join(plateDir(plateId), filename)


def logPlate(plateId, message, suffix=''):
    path = plateLogPath(plateId, suffix=suffix)
    with open(path, 'a') as f:
        f.write(f'[{timestamp()}] {message}\n')
        f.flush()
        os.fsync(f.fileno())


def logWell(plateId, wellId, msg, suffix):
    suffix = suffix.lstrip('_')
    filename = f'{wellId}.colony_{suffix}.log' if suffix else f'{wellId}.log'
    path = os.path.join(plateDir(plateId), filename)

    with open(path, 'a') as f:
        f.write(f'[{timestamp()}] {msg}\n')


def checkpointDir(plateId):
    path = os.path.join(plateDir(plateId), 'checkpoints')
    ensureDir(path)
    return path


def checkpointPath(plateId, wellId, tag):
    filename = f'{wellId}_{tag}.done'
    return os.path.join(checkpointDir(plateId), filename)


def checkpointExists(plateId, wellId, tag):
    return os.path.exists(checkpointPath(plateId, wellId, tag))


def writeCheckpoint(plateId, wellId, tag, metadata):
    path = checkpointPath(plateId, wellId, tag)
    with open(path, 'w') as f:
        for k, v in metadata.items():
            f.write(f'{k}={v}\n')


def perWellDir(plateId):
    path = os.path.join(plateDir(plateId), 'tmp')
    ensureDir(path)
    return path


def perWellTmpColonyCsv(plateId, wellId):
    return os.path.join(perWellDir(plateId), f'{wellId}.colony.tmp.csv')


def perWellTmpWellCsv(plateId, wellId):
    return os.path.join(perWellDir(plateId), f'{wellId}.well.tmp.csv')