# Single-plate test run with separate output directory
# Handles "Bright Field" (space) filenames and magnification filtering.
#
# Usage: python runSinglePlateTest.py <plateDir> [outdir] [mag_suffix]
#
# mag_suffix: _01=4x, _02=4x, _03=10x, _04=20x  (default: _02)
#
# Example:
#   python runSinglePlateTest.py '/mnt/bridgeslab/Good imaging data/...' /tmp/prototype_test_output _02

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import re
import time
import numpy as np
import pandas as pd
import imageio.v3 as iio
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import analysis_main as am
from helpers import round_odd


def processWell(
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
    Imax,
    magSuffix='_02',
    fftStride=6,
    downsample=4
):
    timings = {}

    def tic(key):
        timings[key] = -time.perf_counter()

    def toc(key):
        timings[key] += time.perf_counter()
        log(f'timing.{key} = {timings[key]:.3f} s')

    plateName = os.path.basename(plateDir)
    plateOutdir = os.path.join(outdir, plateName)
    os.makedirs(plateOutdir, exist_ok=True)

    checkpointDir = os.path.join(outdir, 'checkpoints', plateName)
    os.makedirs(checkpointDir, exist_ok=True)

    wellMag = f'{well}{magSuffix}'

    paramHash = (
        f'mag{magSuffix}_'
        f'bd{blockDiameter}_'
        f'fft{fftStride}_'
        f'ds{downsample}_'
        f'ft{fixedThresh:.3f}_'
        f'shift{shiftThresh}_'
        f'dust{int(dustCorrection)}'
    )

    timeseries_path = os.path.join(plateOutdir, f'{wellMag}_timeseries.csv')
    checkpointPath = os.path.join(checkpointDir, f'{well}_{paramHash}.done')
    logPath = os.path.join(plateOutdir, f'{wellMag}.log')

    def now():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(msg):
        with open(logPath, 'a') as f:
            f.write(f'[{now()}] {msg}\n')

    log(f'status=in_progress')
    log(f'start well={well} mag={magSuffix}')

    if os.path.exists(checkpointPath) and os.path.exists(timeseries_path):
        log(f'checkpoint at {checkpointPath}  and timeseries at {timeseries_path} exists for well={well}, skipping')
        return well, 'skipped'

    mutant = replicateMap.get(well)
    if mutant is None or pd.isna(mutant):
        log(f'filtered: no mutant mapping for well={well}')
        return well, 'filtered'

    files = sorted(files, key=am.frame_index_from_filename)
    ntimepoints = len(files)

    img0 = iio.imread(files[0])
    h, w = img0.shape

    tic('load_images')
    stack = np.empty((h, w, ntimepoints), dtype=np.float64)
    am.read_images_inplace(ntimepoints, stack, files)
    toc('load_images')

    log(f'loaded {ntimepoints} frames for well={well}')

    t0 = time.perf_counter()
    t_stage = time.perf_counter()
    log('starting timelapse_processing')

    log(f'params: mag={magSuffix}, blockDiameter={blockDiameter}, fixedThresh={fixedThresh}, '
        f'fftStride={fftStride}, downsample={downsample}')

    try:
        tic('timelapse_processing')
        masks, biomass, odMean = am.timelapse_processing(
            images=stack,
            block_diameter=blockDiameter,
            ntimepoints=ntimepoints,
            shift_thresh=shiftThresh,
            fixed_thresh=fixedThresh,
            dust_correction=dustCorrection,
            outdir=plateOutdir,
            filename=wellMag,
            image_records=None,
            Imin=None,
            Imax=None,
            fftStride=fftStride,
            downsample=downsample
        )
        toc('timelapse_processing')

        elapsed = time.perf_counter() - t0

        tmp_csv = timeseries_path + '.tmp'
        pd.DataFrame({
            'plate': plateName,
            'well': well,
            'mag': magSuffix,
            'mutant': mutant,
            'frame': np.arange(len(biomass), dtype=np.int32),
            'biomass': biomass.astype(np.float32),
            'od_mean': (
                odMean.astype(np.float32)
                if odMean is not None
                else np.full(len(biomass), np.nan, dtype=np.float32)
            )
        }).to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, timeseries_path)

        log(f'wrote timeseries={timeseries_path}')

        tmpPath = checkpointPath + '.tmp'
        with open(tmpPath, 'w') as f:
            f.write(f'well={well}\n')
            f.write(f'mag={magSuffix}\n')
            f.write(f'elapsedSeconds={elapsed:.3f}\n')
            f.write(f'finishedAt={now()}\n')
            f.write(f'fixedThresh={fixedThresh}\n')
            f.write(f'fftStride={fftStride}\n')
            f.write(f'downsample={downsample}\n')
        os.replace(tmpPath, checkpointPath)

        log(f'timelapse_processing finished in {time.perf_counter() - t_stage:.2f}s')
        log(f'checkpoint={checkpointPath}')
        log(f'status=done')

        return {
            'plate': plateName,
            'well': well,
            'mag': magSuffix,
            'status': 'done'
        }

    except Exception as e:
        log(f'error processing well={well}: {e}')
        for k, v in timings.items():
            if v > 0:
                log(f'timing.{k} = {v:.3f} s (partial)')
        return well, 'error'


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
                Imax,
                magSuffix
            )
            for well, files in byWell.items()
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    return results


if __name__ == '__main__':
    import argparse

    DEFAULT_PLATE = (
        '/mnt/bridgeslab/Good imaging data/Multi-phenotype training/'
        '241010_105227_4x_10x_20x_40x_Discontinuous_Drawer1 10-Oct-2024 10-48-54/'
        '241010_105227_Plate 1'
    )

    parser = argparse.ArgumentParser(description='Single-plate prototype test run')
    parser.add_argument('plateDir', nargs='?', default=DEFAULT_PLATE,
                        help='Path to plate directory (default: training plate 1)')
    parser.add_argument('-o', '--outdir', default='/mnt/data/trainingData/tmp/procProtoTest',
                        help='Output directory')
    parser.add_argument('-m', '--mag', default='_03',
                        help='Magnification suffix: _01=4x, _02=4x, _03=10x, _04=20x (default: _02)')
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    plateDir = args.plateDir
    outdir = args.outdir
    magSuffix = args.mag if args.mag.startswith('_') else f'_{args.mag}'

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
    print(f'Workers:       {args.workers}')

    t0 = time.perf_counter()

    results = runPlateSingleMag(
        plateDir=plateDir,
        outdir=outdir,
        replicateMap=replicateMap,
        magSuffix=magSuffix,
        maxWorkers=args.workers
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
                'mag': magSuffix,
                'status': res[1]
            })

    df = pd.DataFrame(allResults, columns=['plate', 'well', 'mag', 'status'])
    summary_path = os.path.join(outdir, f'run_summary{magSuffix}.csv')
    df.to_csv(summary_path, index=False)
    print(f'Summary written to {summary_path}')
    print(df['status'].value_counts().to_string())
