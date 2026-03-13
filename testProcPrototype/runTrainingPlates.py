# NOTE:
# This is a legacy execution script that bypasses the package entrypoint.
# Do NOT import this module elsewhere.

# longterm, this functionality should be replaced with python -m multiWellAnalysis.processing.pipeline


import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


import sys
sys.path.append('/home/smellick/ImageLibrary')

import time
import numpy as np
import imageio.v3 as iio
import importlib
import pandas as pd

from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


import multiWellAnalysis.registration as r
importlib.reload(r)

import multiWellAnalysis.segmentation as s
importlib.reload(s)

import multiWellAnalysis.preprocessing as p
importlib.reload(p)

import multiWellAnalysis.analysis_main as am
importlib.reload(am)

from datetime import datetime


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
    
    paramHash = (
        f'bd{blockDiameter}_'
        f'fft{fftStride}_'
        f'ds{downsample}_'
        f'ft{fixedThresh:.3f}_'
        f'shift{shiftThresh}_'
        f'dust{int(dustCorrection)}'
    )
    
    timeseries_path = os.path.join(plateOutdir, f'{well}_timeseries.csv')

    checkpointPath = os.path.join(checkpointDir, f'{well}_{paramHash}.done')
    
    logPath = os.path.join(plateOutdir, f'{well}.log')
    
    def now():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(msg):
        with open(logPath, 'a') as f:
            f.write(f'[{now()}] {msg}\n')
            
    log(f'status=in_progress')
            
    log(f'start well={well}')

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

    log(f'params: blockDiameter={blockDiameter}, fixedThresh={fixedThresh}, '
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
            filename=well,
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
            'status': 'done'
        }

    except Exception as e:
        log(f'error processing well={well}: {e}')
        for k, v in timings.items():
            if v > 0:
                log(f'timing.{k} = {v:.3f} s (partial)')

        return well, 'error'

def runPlate(
    plateDir,
    outdir,
    replicateMap,
    maxWorkers=min(48, os.cpu_count() // 2)
):
    plateName = os.path.basename(plateDir)
    plateOutdir = os.path.join(outdir, plateName)
    os.makedirs(plateOutdir, exist_ok=True)

    
    tifFiles = glob(os.path.join(plateDir, '*.tif'))

    bfFiles = [
        f for f in tifFiles
        if 'Bright_Field' in f
    ]

    byWell = defaultdict(list)
    for f in bfFiles:
        well = os.path.basename(f).split('_')[0]
        byWell[well].append(f)

    blockDiameter = am.round_odd(101)
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

import pandas as pd
import os

platesRoot = '/home/smellick/ImageLibrary/phenotypr/plates/plates'

outdir = '/mnt/data/trainingData'
os.makedirs(outdir, exist_ok=True)

replicateMap = dict(
    pd.read_csv('/home/smellick/ImageLibrary/ReplicatePositions.csv')
    .set_index('Header')['Replicate ID']
)

def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

allResults = []

for plateDir in sorted(glob(os.path.join(platesRoot, '*Plate*'))):
    plateName = os.path.basename(plateDir)
    print(f'Running plate: {plateName}')
    
    
    plateLog = os.path.join(outdir, f'{plateName}.log')

    with open(plateLog, 'a') as f:
        f.write(f'[{now()}] starting plate {plateName}\n')


    t0 = time.perf_counter()

    plateResults = runPlate(
        plateDir=plateDir,
        outdir=outdir,
        replicateMap=replicateMap,
        maxWorkers=24
    )

    elapsed = time.perf_counter() - t0
    print(f'Plate {plateName} finished in {elapsed/60:.2f} minutes')

    for res in plateResults:
        if isinstance(res, dict):
            allResults.append({
                'plate': res['plate'],
                'well': res['well'],
                'status': 'done'
            })
        else:
            # ('well', 'skipped' | 'filtered' | 'error')
            allResults.append({
                'plate': plateName,
                'well': res[0],
                'status': res[1]
            })

        
    with open(plateLog, 'a') as f:
        f.write(f'[{now()}] finished plate {plateName} in {elapsed/60:.2f} minutes\n')



df = pd.DataFrame(allResults, columns=['plate', 'well', 'status'])
df.to_csv(os.path.join(outdir, 'run_summary.csv'), index=False)

