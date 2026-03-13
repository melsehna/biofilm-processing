# Single-plate test run with separate output directory
# Handles "Bright Field" (space) filenames and magnification filtering.
#
# Usage: python runSinglePlateTest.py <plateDir> [outdir] [mag_suffix]
#
# mag_suffix: _01=4x, _02=10x, _03=20x, _04=40x  (default: _02)
#
# Example:
#   python runSinglePlateTest.py '/mnt/bridgeslab/Good imaging data/...' /tmp/prototype_test_output _02

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
sys.path.append('/home/smellick/ImageLibrary')

import re
import time
import pandas as pd
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from runTrainingPlates import processWell
from helpers import round_odd


def runPlateSingleMag(plateDir, outdir, replicateMap, magSuffix='_02', maxWorkers=24):
    plateName = os.path.basename(plateDir)
    plateOutdir = os.path.join(outdir, plateName)
    os.makedirs(plateOutdir, exist_ok=True)

    tifFiles = glob(os.path.join(plateDir, '*.tif'))

    # Filter to Bright Field images (handles both "Bright Field" and "Bright_Field")
    bfFiles = [f for f in tifFiles if 'Bright Field' in f or 'Bright_Field' in f]

    # Filter to selected magnification
    # Filenames like A10_02_1_1_Bright Field_001.tif — mag suffix is the second underscore-delimited token
    magFiles = []
    for f in bfFiles:
        base = os.path.basename(f)
        # Match well_mag pattern at start: e.g. A10_02
        m = re.match(r'^([A-H]\d+)(_\d+)_', base)
        if m and m.group(2) == magSuffix:
            magFiles.append(f)

    # Group by well
    byWell = defaultdict(list)
    for f in magFiles:
        well = re.match(r'^([A-H]\d+)_', os.path.basename(f)).group(1)
        byWell[well].append(f)

    print(f'Found {len(magFiles)} images across {len(byWell)} wells (mag={magSuffix})')

    if not byWell:
        print('No matching files found!')
        return []

    blockDiameter = round_odd(101)
    fixedThresh = 0.014
    shiftThresh = 50
    dustCorrection = True
    Imin = None
    Imax = None

    results = []

    with ProcessPoolExecutor(max_workers=maxWorkers) as pool:
        futures = [
            pool.submit(
                processWell,
                plateDir,
                outdir,
                well,
                files,
                replicateMap,
                blockDiameter,
                fixedThresh,
                shiftThresh,
                dustCorrection,
                Imin,
                Imax
            )
            for well, files in byWell.items()
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python runSinglePlateTest.py <plateDir> [outdir] [mag_suffix]')
        print('  mag_suffix: _01=4x, _02=10x, _03=20x, _04=40x  (default: _02)')
        sys.exit(1)

    plateDir = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/prototype_test_output'
    magSuffix = sys.argv[3] if len(sys.argv) > 3 else '_02'

    if not os.path.isdir(plateDir):
        print(f'Error: plate directory not found: {plateDir}')
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    replicateMap = dict(
        pd.read_csv('/home/smellick/ImageLibrary/ReplicatePositions.csv')
        .set_index('Header')['Replicate ID']
    )

    plateName = os.path.basename(plateDir)
    print(f'Running plate: {plateName}')
    print(f'Output dir:    {outdir}')
    print(f'Magnification: {magSuffix}')

    t0 = time.perf_counter()

    results = runPlateSingleMag(
        plateDir=plateDir,
        outdir=outdir,
        replicateMap=replicateMap,
        magSuffix=magSuffix,
        maxWorkers=24
    )

    elapsed = time.perf_counter() - t0
    print(f'Plate {plateName} finished in {elapsed/60:.2f} minutes')

    allResults = []
    for res in results:
        if isinstance(res, dict):
            allResults.append(res)
        else:
            allResults.append({
                'plate': plateName,
                'well': res[0],
                'status': res[1]
            })

    df = pd.DataFrame(allResults, columns=['plate', 'well', 'status'])
    summary_path = os.path.join(outdir, 'run_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f'Summary written to {summary_path}')
    print(df['status'].value_counts().to_string())
