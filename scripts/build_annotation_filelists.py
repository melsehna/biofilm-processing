#!/usr/bin/env python
"""Generate the 2 global annotation file lists (split (A) by type).

Walks the deposit tree's annotations/ subtree and emits:
  _filelists/masks_filelist.tsv   — all _mask.ome.tiff   (uint8 0/255)
  _filelists/labels_filelist.tsv  — all _labels.ome.tiff (uint16 colony IDs)
Each row links to its study image via source_image (derived from the path).
Supersedes any per-component annotation lists (removed here).

  python scripts/build_annotation_filelists.py --apply
"""
import argparse, os, sys, glob
import deposit_lib as L
import deposit_biology as B


def source_image(rel, kind):
    # annotations/masks/<...>/<stem>_mask.ome.tiff -> <...>/<stem>_image.ome.tiff
    p = rel.split(f"annotations/{kind}/", 1)[1]
    suff = "_mask.ome.tiff" if kind == "masks" else "_labels.ome.tiff"
    return p.replace(suff, "_image.ome.tiff")


def collect(root, kind):
    suff = "_mask.ome.tiff" if kind == "masks" else "_labels.ome.tiff"
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "annotations", kind, "**", f"*{suff}"), recursive=True)):
        rel = os.path.relpath(f, root)
        rows.append(dict(
            Files=rel, source_image=source_image(rel, kind),
            AnnotationType=("segmentation mask (binary)" if kind == "masks" else "instance labels (per-colony)"),
            AnnotationMethod="biofilm-processing pipeline",
            PixelValueMeaning=("0 = background, 255 = biofilm" if kind == "masks"
                               else "0 = background, N = colony ID (tracked across frames)")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default="/mnt/data/bia_deposit/Brightfield")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = a.deposit_root
    fldir = os.path.join(root, "_filelists")

    # remove any superseded per-component annotation lists
    if a.apply:
        for old in glob.glob(os.path.join(fldir, "*_masks_filelist.tsv")):
            os.remove(old); print("removed superseded:", os.path.basename(old))

    for kind, out in [("masks", "masks_filelist.tsv"), ("labels", "labels_filelist.tsv")]:
        rows = collect(root, kind)
        # verify every source_image exists
        missing = [r["source_image"] for r in rows if not os.path.exists(os.path.join(root, r["source_image"]))]
        print(f"{kind}: {len(rows)} files   missing source_image: {len(missing)}")
        if missing:
            for m in missing[:5]:
                print("   MISSING:", m)
        if rows:
            L.writeFileList(rows, B.ANNOTATION_COLUMNS, os.path.join(fldir, out), apply=a.apply)
            if a.apply:
                print(f"  wrote {out}")
    if not a.apply:
        print("\n[DRY-RUN] nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
