#!/usr/bin/env python
"""Core helpers for building the BioImage Archive deposit tree.

Reads working NAS files (read-only) and writes OME-TIFF copies into a separate
local deposit tree. Never renames/moves/modifies the working files. All writes
are create-only (O_CREAT|O_EXCL); callers pass apply=False for dry-run.

Conversions (all → OME-TIFF, axes TYX, physical pixel size in µm embedded):
  image  : float32 _processed.tif (already T,H,W)         -> float32
  mask   : _masks.npz['masks'] (H,W,T bool)               -> uint8 0/255
  labels : _trackedLabels*.npz['labels'] (H,W,T uint16)   -> uint16
"""
import os, re, glob
import numpy as np
import tifffile
from datetime import datetime


def _create_only(dest):
    """Claim dest atomically; raise FileExistsError if it already exists."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.close(fd)


def _ome_md(px):
    md = {"axes": "TYX"}
    if px:
        md.update(PhysicalSizeX=px, PhysicalSizeXUnit="µm",
                  PhysicalSizeY=px, PhysicalSizeYUnit="µm")
    return md


def omeImage(processedTif, dest, px, compression=None):
    """float32 _processed.tif (T,H,W) -> float32 OME-TIFF (TYX). Returns the array."""
    arr = tifffile.imread(processedTif)          # (T,H,W) float32, already TYX
    arr = np.asarray(arr, dtype=np.float32)
    _create_only(dest)
    tifffile.imwrite(dest, arr, ome=True, metadata=_ome_md(px), compression=compression)
    return arr


def verifyWritten(dest, arr):
    """Verify the written OME-TIFF is bit-identical to the in-memory source array
    (reads only the local dest — no second CIFS read of the source)."""
    b = np.asarray(tifffile.imread(dest), dtype=np.float32)
    return b.shape == arr.shape and np.array_equal(b, arr)


def omeMask(npz, dest, px, compression="zlib"):
    a = np.load(npz)["masks"]                     # (H,W,T) bool
    d = np.transpose(a.astype(np.uint8) * 255, (2, 0, 1))   # -> TYX
    _create_only(dest)
    tifffile.imwrite(dest, d, ome=True, metadata=_ome_md(px), compression=compression)
    return d.shape


def omeLabels(npz, dest, px, compression="zlib"):
    a = np.load(npz)["labels"]                     # (H,W,T) uint16
    d = np.transpose(a.astype(np.uint16), (2, 0, 1))
    _create_only(dest)
    tifffile.imwrite(dest, d, ome=True, metadata=_ome_md(px), compression=compression)
    return d.shape


def roundtripImageOK(processedTif, dest):
    """Verify the deposit image is bit-identical to the source float32."""
    a = np.asarray(tifffile.imread(processedTif), dtype=np.float32)
    b = np.asarray(tifffile.imread(dest), dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a, b)


_TS = re.compile(r"<Date>([^<]+)</Date>.*?<Time>([^<]+)</Time>", re.S)

def imagingCadence(rawPlateDir, well):
    """(nFrames, intervalMin, durationH) from raw Cytation per-frame <Date>/<Time>.

    Cadence is uniform within a dataset, so callers probe once and reuse. well is
    the raw well token, e.g. 'A1_02'. Returns (None,None,None) if unavailable.
    """
    frames = sorted(glob.glob(os.path.join(rawPlateDir, f"{well}_*Bright*Field*_*.tif")))
    if len(frames) < 2:
        return (len(frames) or None, None, None)

    def ts(p):
        with tifffile.TiffFile(p) as tf:
            m = _TS.search(tf.pages[0].description)
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%m/%d/%y %H:%M:%S")

    t0, tN = ts(frames[0]), ts(frames[-1])
    dur_h = (tN - t0).total_seconds() / 3600.0
    interval_min = dur_h * 60.0 / (len(frames) - 1)
    return (len(frames), round(interval_min, 1), round(dur_h, 2))


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitizeToken(s):
    """BIA-safe token: alnum + . _ - only (spaces/other -> _)."""
    return _SAFE.sub("_", s).strip("_")


def writeFileList(rows, columns, path, apply=False):
    """Write a BIA file list as TSV (Files first; one file per line; no blanks)."""
    assert columns[0] == "Files", "first column must be 'Files'"
    if not apply:
        return len(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def cell(v):
        return re.sub(r"[\t\r\n]+", " ", str(v)).strip()
    with open(path, "w", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(cell(r.get(c, "")) for c in columns) + "\n")
    return len(rows)
