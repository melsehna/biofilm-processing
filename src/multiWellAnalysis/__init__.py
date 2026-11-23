"""
multiWellAnalysis

High-throughput biofilm phenotyping from automated brightfield timelapse microscopy.
"""

# One compute thread per process, set BEFORE the submodule imports below load
# numpy (OpenBLAS/OpenMP) and OpenCV: those size their thread pools to every
# core when first loaded, and environment variables set later are ignored.
# Parallelism comes from the process pools (`--workers` / the GUI worker
# count), so N workers use N cores instead of N x (all cores). setdefault keeps
# any value the caller exported (e.g. OMP_NUM_THREADS=8 for an interactive
# session). Setting these inside a CLI module is too late: importing
# `multiWellAnalysis.cli.run_pipeline` runs this file first.
import os as _os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    _os.environ.setdefault(_var, '1')
del _var

from importlib.metadata import version, PackageNotFoundError
from . import analysis
from . import colony
from . import processing
from . import wholeImage

__all__ = ['analysis', 'colony', 'processing', 'wholeImage']

try:
    __version__ = version('biofilm-processing')
except PackageNotFoundError:
    __version__ = 'unknown'
