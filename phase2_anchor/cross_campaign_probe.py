#!/usr/bin/env python3
"""Phase 2, part 2 — cross-campaign anchor pressure-test.

The Oct-11/Oct-17 pair drifts only ~1% (same scope, 6 days apart). This probe adds
the 2025 reimaging campaign (different protocol: single-10x BF_YL) to get a pair with
real photometric drift between 2024 training and 2025 reimaging, and asks:

  1. How big is the cross-campaign background (empty-agar) drift?  (= what the anchor
     must remove)
  2. Is the early-frame per-well anchor still well-defined in the 2025 regime?

Background = histogram mode of early (pre-growth) frames; biology-independent.
"""
import os, glob
import numpy as np
import tifffile
from multiWellAnalysis.processing.analysis_main import _toBitDepthScaled

# (label, plateDir, candidate wells) — 10x / _03 / Bright Field
SESSIONS = [
    ("2024 train Oct11",
     "/mnt/bridgeslab/Good imaging data/Multi-phenotype training/241011_183053_4x_10x_20x_40x_Discontinuous_Drawer7 11-Oct-2024 16-30-30/241011_183053_Plate 1",
     ["A1", "B2", "B5", "B9", "C5", "D5"]),
    ("2024 train Oct17",
     "/mnt/bridgeslab/Good imaging data/Multi-phenotype training/241017_183030_4x_10x_20x_40x_Discontinuous_Drawer7 17-Oct-2024 16-08-32/241017_183030_Plate 1",
     ["A1", "B2", "B5", "B9", "C5", "D5"]),
    ("2025 reimg May13",
     "/mnt/bridgeslab/250513_121437_10x_BF_YL_Drawer5 13-May-2025 12-11-04/250513_121437_Plate 1",
     ["A1", "A2", "A3", "B1", "B2", "B5"]),
]
NBINS = 2000
NEARLY = 3  # frames 0..2 averaged for the per-well background estimate


def earlyBgMode(plateDir, well):
    files = sorted(glob.glob(os.path.join(
        plateDir, f"{well}_03_1_1_Bright Field_*.tif")))[:NEARLY]
    if not files:
        return None
    modes = []
    for f in files:
        img = _toBitDepthScaled(tifffile.imread(f))
        hist, edges = np.histogram(img.ravel(), bins=NBINS, range=(0.0, 1.0))
        mi = int(np.argmax(hist))
        modes.append(0.5 * (edges[mi] + edges[mi + 1]))
    return float(np.mean(modes))


def run():
    print(f"{'session':18} {'n':>2} {'bg mode mean':>12} {'within-sess spread':>20}")
    summary = {}
    for label, plateDir, wells in SESSIONS:
        vals = [earlyBgMode(plateDir, w) for w in wells]
        vals = [v for v in vals if v is not None]
        m = float(np.mean(vals))
        spread = float(np.max(vals) - np.min(vals))
        summary[label] = m
        print(f"{label:18} {len(vals):>2} {m:>12.4f} "
              f"{spread:>20.4f}  (wells {[round(v,3) for v in vals]})")

    base = summary["2024 train Oct11"]
    print("\n=== background drift vs 2024 train Oct11 ===")
    for label, m in summary.items():
        print(f"  {label:18} drift={m-base:+.4f}  ({(m-base)/base*100:+.2f}%)")


if __name__ == "__main__":
    run()
