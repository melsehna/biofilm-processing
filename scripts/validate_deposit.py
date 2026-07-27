#!/usr/bin/env python
"""Pre-upload validation of the BIA deposit tree + file lists.

Checks: every study image has a mask; every annotation source_image resolves;
µm/px + SizeT embedded; BIA filename-rule compliance; file lists well-formed
(Files first col, one file/line, no blanks, no dup paths, all paths exist).

  python scripts/validate_deposit.py [--deposit-root /mnt/data/bia_deposit/Brightfield]
"""
import argparse, csv, os, re, sys, glob
import tifffile

# BIA allowed filename chars: alnum + ! - _ . * ' ( ) and space
BIA_OK = re.compile(r"^[A-Za-z0-9!\-_.*'()/ ]+$")


def read_filelist(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    hdr = lines[0].split("\t")
    rows = [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:] if ln]
    blanks = sum(1 for ln in lines[1:] if not ln)
    return hdr, rows, blanks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default="/mnt/data/bia_deposit/Brightfield")
    a = ap.parse_args()
    root = a.deposit_root
    fails, warns = [], []

    fls = sorted(glob.glob(os.path.join(root, "_filelists", "*.tsv")))
    print(f"file lists: {len(fls)}")
    study_imgs, annot_srcs = set(), []
    for fl in fls:
        hdr, rows, blanks = read_filelist(fl)
        name = os.path.basename(fl)
        if hdr[0] != "Files":
            fails.append(f"{name}: first column is {hdr[0]!r}, must be 'Files'")
        if blanks:
            fails.append(f"{name}: {blanks} blank line(s)")
        paths = [r["Files"] for r in rows]
        if len(paths) != len(set(paths)):
            fails.append(f"{name}: duplicate Files paths")
        for r in rows:
            f = r["Files"]
            if not BIA_OK.match(f):
                fails.append(f"{name}: illegal chars in {f}")
            if not os.path.exists(os.path.join(root, f)):
                fails.append(f"{name}: missing file {f}")
            if "source_image" in r:
                annot_srcs.append((name, r["source_image"]))
            elif f.endswith("_image.ome.tiff"):
                study_imgs.add(f)
        print(f"  {name}: {len(rows)} rows, header {len(hdr)} cols")

    # every annotation's source_image must be a real study image
    for name, src in annot_srcs:
        if not os.path.exists(os.path.join(root, src)):
            fails.append(f"{name}: source_image not found: {src}")
        elif src not in study_imgs:
            warns.append(f"{name}: source_image not in a study file list: {src}")

    # every study image has a matching mask on disk
    for img in sorted(study_imgs):
        mask = "annotations/masks/" + img.replace("_image.ome.tiff", "_mask.ome.tiff")
        if not os.path.exists(os.path.join(root, mask)):
            fails.append(f"image without mask: {img}")

    # spot-check OME metadata on a sample of images
    sample = sorted(glob.glob(os.path.join(root, "**", "*_image.ome.tiff"), recursive=True))[:5]
    for tif in sample:
        with tifffile.TiffFile(tif) as tf:
            ome = tf.ome_metadata or ""
        if 'PhysicalSizeX="' not in ome:
            fails.append(f"no pixel size: {os.path.relpath(tif, root)}")
        if 'SizeT="1"' in ome and 'SizeZ="1"' not in ome:
            warns.append(f"SizeT=1 (single frame?): {os.path.relpath(tif, root)}")

    print(f"\nstudy images: {len(study_imgs)}   annotation rows: {len(annot_srcs)}")
    print(f"FAILURES: {len(fails)}   warnings: {len(warns)}")
    for m in fails[:20]:
        print("  FAIL:", m)
    for m in warns[:10]:
        print("  warn:", m)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
