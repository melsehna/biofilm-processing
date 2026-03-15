"""Quick single-well test of the processing pipeline on the test branch.

Runs one well (A1, mag _03) from training plate 1 and writes output to
/mnt/data/trainingData/tmp/test_fast_branch/ so nothing existing is touched.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from multiWellAnalysis.processing.analysis_main import timelapse_processing, frame_index_from_filename
from multiWellAnalysis.processing.io_utils import read_images_inplace
from multiWellAnalysis.processing.helpers import round_odd
from glob import glob

PLATE_DIR = (
    '/mnt/bridgeslab/Good imaging data/Multi-phenotype training/'
    '241010_105227_4x_10x_20x_40x_Discontinuous_Drawer1 10-Oct-2024 10-48-54/'
    '241010_105227_Plate 1'
)
OUTDIR = '/mnt/data/trainingData/tmp/test_fast_branch'
WELL = 'A1'
MAG_SUFFIX = '_03'

os.makedirs(OUTDIR, exist_ok=True)

# Find files for this well + mag
all_tifs = glob(os.path.join(PLATE_DIR, '*.tif'))
well_files = sorted(
    [f for f in all_tifs
     if os.path.basename(f).startswith(f'{WELL}{MAG_SUFFIX}_')
     and 'Bright Field' in f],
    key=frame_index_from_filename
)

print(f'Well: {WELL}, Mag: {MAG_SUFFIX}')
print(f'Found {len(well_files)} frames')
print(f'Output: {OUTDIR}')

import imageio.v3 as iio
img0 = iio.imread(well_files[0])
h, w = img0.shape
nframes = len(well_files)

print(f'Frame size: {h}x{w}, {nframes} timepoints')

# Load stack
t0 = time.perf_counter()
stack = np.empty((h, w, nframes), dtype=np.float64)
read_images_inplace(nframes, stack, well_files)
t_load = time.perf_counter() - t0
print(f'Load: {t_load:.2f}s')

# Process
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
t_proc = time.perf_counter() - t0

print(f'Processing: {t_proc:.2f}s')
print(f'Total: {t_load + t_proc:.2f}s')
print(f'Biomass range: [{biomass.min():.6f}, {biomass.max():.6f}]')
print(f'Mask coverage (last frame): {masks[..., -1].mean()*100:.1f}%')
print(f'\nOutputs:')
for f in sorted(os.listdir(os.path.join(OUTDIR, 'processedImages'))):
    fpath = os.path.join(OUTDIR, 'processedImages', f)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f'  {f}  ({size_mb:.1f} MB)')
