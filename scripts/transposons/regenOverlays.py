#!/usr/bin/env python3
"""Regenerate overlay MP4s for transposon plates.

Thin wrapper around scripts/regenOverlays.py for the transposon dataset.
"""

import subprocess
import sys
import os

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'regenOverlays.py')

PLATE_DIR = '/mnt/data/transposonSet/241106_150118_Plate 1'
WELLS = ['A1_03', 'B5_03', 'D7_03', 'F10_03', 'H12_03']

cmd = [sys.executable, SCRIPT, PLATE_DIR, '--wells'] + WELLS
subprocess.run(cmd, check=True)
