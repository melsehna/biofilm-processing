"""
multiWellAnalysis

High-throughput biofilm phenotyping from automated brightfield timelapse microscopy.
"""

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
