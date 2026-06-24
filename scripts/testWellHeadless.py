#!/usr/bin/env python3
"""Headless single-well parameter check, from a source checkout.

Thin wrapper around `multiWellAnalysis.cli.test_well:main`. If the package is
installed, prefer the `biofilm-processing-test-well` console command. See
`python scripts/testWellHeadless.py --help`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multiWellAnalysis.cli.test_well import main

if __name__ == '__main__':
    sys.exit(main())
