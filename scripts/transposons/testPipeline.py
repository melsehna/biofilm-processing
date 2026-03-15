import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import time
import numpy as np
import imageio.v3 as iio
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from multiWellAnalysis.processing.analysis_main import timelapse_processing, frame_index_from_filename
from multiWellAnalysis.processing.io_utils import read_images_inplace
from multiWellAnalysis.processing.helpers import round_odd

PLATE_DIR = (
    '/mnt/bridgeslab/Good imaging data/Multi-phenotype training/'
    '241010_105227_4x_10x_20x_40x_Discontinuous_Drawer1 10-Oct-2024 10-48-54/'
    '241010_105227_Plate 1'
)
OUTDIR = '/mnt/data/trainingData/tmp/test_fast_branch'
WELL = 'A1'
MAG_SUFFIX = '_03'

os.makedirs(OUTDIR, exist_ok=True)

allTifs = glob(os.path.join(PLATE_DIR, '*.tif'))
wellFiles = sorted(
    [f for f in allTifs
     if os.path.basename(f).startswith(f'{WELL}{MAG_SUFFIX}_')
     and 'Bright Field' in f],
    key=frame_index_from_filename
)

print(f'Well: {WELL}, Mag: {MAG_SUFFIX}')
print(f'Found {len(wellFiles)} frames')
print(f'Output: {OUTDIR}')

img0 = iio.imread(wellFiles[0])
h, w = img0.shape
nframes = len(wellFiles)
print(f'Frame size: {h}x{w}, {nframes} timepoints')

t0 = time.perf_counter()
stack = np.empty((h, w, nframes), dtype=np.float64)
read_images_inplace(nframes, stack, wellFiles)
tLoad = time.perf_counter() - t0
print(f'Load: {tLoad:.2f}s')

t0 = time.perf_counter()
masks, biomass, odMean = timelapse_processing(
    images=stack,
    block_diameter=round_odd(101),
    ntimepoints=nframes,
    shift_thresh=50,
    fixed_thresh=0.014,
    dust_correction=True,
    outdir=OUTDIR,
    filename=f'{WELL}{MAG_SUFFIX}',
    image_records=None,
    Imin=None,
    Imax=None,
    skip_overlay=False,
)
tProc = time.perf_counter() - t0

print(f'Processing: {tProc:.2f}s')
print(f'Total: {tLoad + tProc:.2f}s')
print(f'Biomass range: [{biomass.min():.6f}, {biomass.max():.6f}]')
print(f'Mask coverage (last frame): {masks[..., -1].mean()*100:.1f}%')
print(f'\nOutputs:')
for f in sorted(os.listdir(os.path.join(OUTDIR, 'processedImages'))):
    fpath = os.path.join(OUTDIR, 'processedImages', f)
    sizeMb = os.path.getsize(fpath) / 1e6
    print(f'  {f}  ({sizeMb:.1f} MB)')
