import pandas as pd
import numpy as np
import imageio.v3 as iio
from datetime import datetime
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


import time

import multiWellAnalysis.processing.analysis_main as am

INDEX_CSV = '/mnt/data/reimaging/index/reimaging_index_annotated.csv'
OUTDIR = '/mnt/data/reimaging/processed'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INDEX_CSV)

# Optional filters
# df = df[df['repPlate'] == 1]
# df = df[df['geneLocus'].notna()]

# helpers

def files_for_well(platePath, well):
    files = glob(os.path.join(platePath, f'{well}_*.tif'))
    try:
        return sorted(files, key=am.frame_index_from_filename)
    except ValueError:
        return sorted(files)



def processWell(
    platePath,
    outdir,
    well,
    files,
    geneLocus,
    blockDiameter,
    fixedThresh,
    shiftThresh,
    dustCorrection,
    Imin,
    Imax,
    fftStride=6,
    downsample=4
):
    plateName = os.path.basename(platePath)

    timings = {}
    
    def now():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(msg):
        with open(logPath, 'a') as f:
            f.write(f'[{now()}] {msg}\n')

    def tic(key):
        timings[key] = -time.perf_counter()

    def toc(key):
        timings[key] += time.perf_counter()
        log(f'timing.{key} = {timings[key]:.3f} s')

    
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
    

            
    log(f'[X] STATUS = in_progress')
            
    log(f'--- start well={well} ---')

    if os.path.exists(checkpointPath) and os.path.exists(timeseries_path):
        log(f'checkpoint at {checkpointPath}  and timeseries at {timeseries_path} exists for well={well}, skipping')
        return {
            'plate': plateName,
            'well': well,
            'status': 'skipped'
        }


    mutant = geneLocus
    if pd.isna(mutant):
        log(f'filtered: no geneLocus for well={well}')
        return {
            'plate': plateName,
            'well': well,
            'status': 'filtered'
        }





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
        log(f'[X] STATUS = done')
        
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

        return {
            'plate': plateName,
            'well': well,
            'status': 'error'
        }

    
def runPlate(df_plate, maxWorkers=None):
    
    if maxWorkers is None:
        maxWorkers = min(48, os.cpu_count() // 2)
    platePath = df_plate['platePath'].iloc[0]
    plateName = os.path.basename(platePath)

    print(f'Running plate: {plateName}')
    
    assert df_plate[['repWell']].duplicated().sum() == 0

    plateOutdir = os.path.join(OUTDIR, plateName)
    os.makedirs(plateOutdir, exist_ok=True)

    plateLog = os.path.join(OUTDIR, f'{plateName}.log')

    def now():
        return time.strftime('%Y-%m-%d %H:%M:%S')

    with open(plateLog, 'a') as f:
        f.write(f'[{now()}] --- Starting plate {plateName} --- \n')

    t0 = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=maxWorkers) as pool:
        futures = []

        for _, row in df_plate.iterrows():
            files = files_for_well(row['platePath'], row['repWell'])

            if not files:
                results.append({
                    'plate': plateName,
                    'well': row['repWell'],
                    'STATUS': 'no_files'
                })
                continue

            futures.append(
                
                pool.submit(
                    processWell,
                    platePath=row['platePath'],
                    outdir=OUTDIR,
                    well=row['repWell'],
                    files=files,
                    geneLocus=row['geneLocus'],
                    blockDiameter=am.round_odd(101),
                    fixedThresh=0.014,
                    shiftThresh=50,
                    dustCorrection=True,
                    Imin=None,
                    Imax=None
                )
            )

        for fut in as_completed(futures):
            results.append(fut.result())

    elapsed = time.perf_counter() - t0

    with open(plateLog, 'a') as f:
        f.write(f'[{now()}] finished plate {plateName} in {elapsed/60:.2f} minutes\n')

    return results

allResults = []

for platePath, df_plate in df.groupby('platePath'):
    plateResults = runPlate(df_plate)

    for res in plateResults:
        if isinstance(res, dict):
            allResults.append(res)
        else:
            allResults.append({
                'plate': plateName,
                'well': res[0],
                'STATUS': res[1]
            })

pd.DataFrame(allResults).to_csv(
    os.path.join(OUTDIR, 'run_summary.csv'),
    index=False
)
