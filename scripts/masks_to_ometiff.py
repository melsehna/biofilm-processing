#!/usr/bin/env python
"""Convert working masks/labels (.npz) to OME-TIFF for the BioImage Archive deposit.

Reads _masks.npz (bool H,W,T) and, with --labels, _trackedLabels*.npz (uint16
H,W,T) from a working dataset tree, and writes OME-TIFF (axes T,Y,X) with the
physical pixel size (µm, from the sibling index.csv) embedded in OME-XML, into a
SEPARATE deposit tree. The working .npz files are never modified.

  binary masks    -> uint8 (0/255),  axes TYX, zlib (lossless)
  instance labels -> uint16 (IDs preserved), axes TYX, zlib

Page order (T,Y,X) matches the pipeline's _processed.tif so mask page t aligns
with processed frame t (for the BIA source_image linkage).

Default DRY-RUN. --apply writes, create-only (O_CREAT|O_EXCL — never overwrites).

  python scripts/masks_to_ometiff.py --src-root <nas-dataset> --out-root <deposit> [--labels] [--limit N] [--apply]
"""
import argparse, csv, os, sys, glob
import numpy as np
import tifffile


def pxmap(processed_dir):
    """well -> pxToUm from the sibling index.csv."""
    idx = os.path.join(processed_dir, "index.csv")
    m = {}
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx)):
            try:
                m[r["well"]] = float(r["pxToUm"])
            except (KeyError, ValueError):
                pass
    return m


def collect(src_root, include_labels):
    """List (npz_path, kind) for masks (+ labels)."""
    out = []
    for dp, _, fns in os.walk(src_root):
        if os.path.basename(dp) != "processedImages":
            continue
        for fn in sorted(fns):
            if fn.endswith("_masks.npz"):
                out.append((os.path.join(dp, fn), "mask"))
            elif include_labels and "_trackedLabels" in fn and fn.endswith(".npz"):
                out.append((os.path.join(dp, fn), "label"))
    return out


def out_path(src_root, out_root, npz_path, kind):
    plate = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))
    fn = os.path.basename(npz_path)
    well = fn.split("_masks.npz")[0] if kind == "mask" else fn.split("_trackedLabels")[0]
    suffix = "_mask.ome.tiff" if kind == "mask" else "_labels.ome.tiff"
    return os.path.join(out_root, plate, well + suffix), well


def convert_one(npz_path, kind, px):
    if kind == "mask":
        a = np.load(npz_path)["masks"]                 # (H,W,T) bool
        data = (a.astype(np.uint8)) * 255
    else:
        a = np.load(npz_path)["labels"]                # (H,W,T) uint16
        data = a.astype(np.uint16, copy=False)
    return np.transpose(data, (2, 0, 1))               # -> (T,H,W) == TYX


def write_ometiff(dest, data, px):
    md = {"axes": "TYX"}
    if px:
        md.update(PhysicalSizeX=px, PhysicalSizeXUnit="µm",
                  PhysicalSizeY=px, PhysicalSizeYUnit="µm")
    # atomic create-only: claim the path first, fail if it exists
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.close(fd)
    tifffile.imwrite(dest, data, ome=True, metadata=md, compression="zlib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--labels", action="store_true", help="also convert _trackedLabels")
    ap.add_argument("--limit", type=int, default=None, help="convert only first N (pilot)")
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    a = ap.parse_args()

    items = collect(a.src_root, a.labels)
    if a.limit:
        items = items[:a.limit]
    nmask = sum(1 for _, k in items if k == "mask")
    nlab = sum(1 for _, k in items if k == "label")
    print(f"to convert: {len(items)}  (masks={nmask}, labels={nlab})")

    pxcache = {}
    plan = []
    for npz, kind in items:
        pdir = os.path.dirname(npz)
        if pdir not in pxcache:
            pxcache[pdir] = pxmap(pdir)
        dest, well = out_path(a.src_root, a.out_root, npz, kind)
        px = pxcache[pdir].get(well)
        plan.append((npz, kind, dest, px))

    missing_px = [d for _, _, d, px in plan if not px]
    collisions = [d for _, _, d, _ in plan if os.path.lexists(d)]
    print(f"resolved: {len(plan)}   missing pxToUm: {len(missing_px)}   "
          f"existing dests (would skip): {len(collisions)}")
    for d in collisions[:5]:
        print("  EXISTS:", d)

    if not a.apply:
        print("\n[DRY-RUN] nothing written. Re-run with --apply.")
        return 0

    made = 0
    for npz, kind, dest, px in plan:
        if os.path.lexists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = convert_one(npz, kind, px)
        write_ometiff(dest, data, px)
        made += 1
    print(f"wrote {made} OME-TIFFs to {a.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
