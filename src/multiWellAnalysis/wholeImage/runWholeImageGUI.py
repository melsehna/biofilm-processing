#!/usr/bin/env python3
"""
GUI version of runWholeImage.py

Copy of the whole-image feature extraction adapted for the GUI:
- Saves CSV directly to outdir (no {outDir}/{plateId}/ subdirectory)
- No checkpoint files or log files
- No batch runner / multiprocessing main()
- Single entry point: extractWholeImageFeatures()
"""

import os
import time

import numpy as np
import pandas as pd
import tifffile

from .extractWholeImageFeats import extractFrameFeats


# config

FEAT_VERSION = 'mahotas_v2'


def extractWholeImageFeatures(
    processedPath,
    plateId,
    wellId,
    outDir,
    startFrame=0,
    featureVersion=FEAT_VERSION
):
    """Extract whole-image texture features from the processed stack.

    Parameters
    ----------
    processedPath : str
        Path to the processed TIFF stack.
    plateId, wellId : str
    outDir : str
        Directory for the output CSV (typically processedImages/).
    startFrame : int
    featureVersion : str

    Returns
    -------
    status : str — 'done', 'skipped', or 'error'
    """

    timings = {}
    rows = []

    def tic(k):
        timings[k] = -time.perf_counter()

    def toc(k):
        timings[k] += time.perf_counter()

    os.makedirs(outDir, exist_ok=True)

    outCsv = os.path.join(outDir, f'{wellId}_wholeImageFeatures.csv')

    if not os.path.exists(processedPath):
        return 'skipped'

    try:
        tic('loadStack')

        try:
            procStack = tifffile.memmap(processedPath)
        except Exception:
            procStack = tifffile.imread(processedPath)

        toc('loadStack')

        def ensureThw(stack):
            if stack.ndim == 2:
                stack = stack[np.newaxis, :, :]
            elif stack.ndim == 3:
                if stack.shape[0] <= 64:
                    pass
                elif stack.shape[2] <= 64:
                    stack = np.moveaxis(stack, 2, 0)
                elif stack.shape[1] <= 64:
                    stack = np.moveaxis(stack, 1, 0)
                else:
                    raise ValueError(f'cannot infer time axis for stack {stack.shape}')
            else:
                raise ValueError(f'invalid stack shape {stack.shape}')
            return stack

        procStack = ensureThw(procStack)

        tic('featureExtraction')

        for t in range(startFrame, procStack.shape[0]):

            procFeats = extractFrameFeats(procStack[t])
            procFeats = {f'whole_{k}': v for k, v in procFeats.items()}

            procFeats.update({
                'plateId': plateId,
                'wellId': wellId,
                'frame': t,
                'processedPath': processedPath
            })

            rows.append(procFeats)

        toc('featureExtraction')

        if not rows:
            return 'error'

        tmpCsv = outCsv + '.tmp'
        pd.DataFrame(rows).to_csv(tmpCsv, index=False)
        os.replace(tmpCsv, outCsv)

        elapsed = sum(v for v in timings.values() if v > 0)
        print(f'    whole-image features done in {elapsed:.1f}s ({len(rows)} frames)')

        return 'done'

    except Exception as e:
        print(f'    whole-image features error: {e}')

        if rows:
            tmpCsv = outCsv + '.partial'
            pd.DataFrame(rows).to_csv(tmpCsv, index=False)

        return 'error'
