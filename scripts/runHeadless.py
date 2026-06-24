#!/usr/bin/env python3
"""Run the full pipeline headless (HPC / no-GUI), from a source checkout.

Thin wrapper around `multiWellAnalysis.cli.run_pipeline:main` for users who run
from a clone without `pip install -e .`. If the package is installed, prefer the
`biofilm-processing-run` console command instead. All args are passed through —
see `python scripts/runHeadless.py --help`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multiWellAnalysis.cli.run_pipeline import main

if __name__ == '__main__':
    sys.exit(main())
