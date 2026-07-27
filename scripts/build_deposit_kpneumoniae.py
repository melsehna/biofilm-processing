#!/usr/bin/env python
"""Build the K. pneumoniae study component of the BIA deposit (pilot).

Reads working NAS kleb files (read-only), writes float32 image + uint8 mask
OME-TIFFs into the local deposit tree, and emits the study + mask-annotation
file lists. Figure-analyzed scope (wells present in fig5 kleb table).

  dry-run (default):  python scripts/build_deposit_kpneumoniae.py
  spot pilot:         python scripts/build_deposit_kpneumoniae.py --limit 3 --apply
  full component:     python scripts/build_deposit_kpneumoniae.py --apply
"""
import argparse, csv, os, re, sys, glob
import deposit_lib as L
import deposit_biology as B

NAS = "/mnt/phenotyper/Sehna/kleb-data-062926/klebData"
DEFAULT_ROOT = "/mnt/data/bia_deposit/Brightfield"
WELL_RE = re.compile(r"^([A-H]\d{1,2})")


def load_index(pidir):
    idx = os.path.join(pidir, "index.csv")
    rows = {}
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx)):
            rows[r["well"]] = r
    return rows


def load_params(pidir):
    import json
    p = os.path.join(pidir, "run_params.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    pv = d.get("_pipelineVersion", {})
    return dict(blockDiam=d.get("blockDiam", ""), fixedThresh=d.get("fixedThresh", ""),
                Pipeline="biofilm-processing", PipelineVersion=pv.get("version", ""),
                GitCommit=pv.get("gitCommit", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None, help="convert only first N wells (spot pilot)")
    ap.add_argument("--compression", default=None, help="image compression (None=fast, or 'zlib')")
    ap.add_argument("--filelist-only", action="store_true", help="regenerate file list only; no conversion")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    members = B.kleb_membership()
    plate_dirs = sorted(glob.glob(os.path.join(NAS, "*", "*", "processedImages")))
    print(f"NAS kleb processedImages dirs: {len(plate_dirs)}")

    cadence = (None, None, None)
    study_rows, mask_rows = [], []
    made_img = made_mask = skipped = collisions = 0

    for pidir in plate_dirs:
        plate_base = os.path.basename(os.path.dirname(pidir))
        drawer, iso = B.KLEB_ISOLATE.get(plate_base, ("", ""))
        clean_plate = f"Kp_{drawer}_{L.sanitizeToken(iso)}"
        idx = load_index(pidir)
        prov = load_params(pidir)
        raw_dir = next(iter(idx.values()), {}).get("plate_path", "") if idx else ""

        for tif in sorted(glob.glob(os.path.join(pidir, "*_processed.tif"))):
            well_full = os.path.basename(tif).replace("_processed.tif", "")   # e.g. A1_02
            m = WELL_RE.match(well_full)
            well = m.group(1) if m else well_full
            key = (re.sub(r"[\s_]+", "", plate_base).lower(), well)
            bio = members.get(key)
            if bio is None:
                continue                      # not figure-analyzed
            if a.limit and (made_img >= a.limit):
                break
            geno = bio["Genotype"]; gtok = L.sanitizeToken(geno)
            row = idx.get(well_full, {})
            px = float(row["pxToUm"]) if row.get("pxToUm") else None
            obj = row.get("objective", "4"); objtok = f"{obj}x"
            masknpz = os.path.join(pidir, f"{well_full}_masks.npz")

            if cadence == (None, None, None) and raw_dir:
                cadence = L.imagingCadence(raw_dir, well_full)

            img_rel = f"Kpneumoniae/{clean_plate}/{well}_{objtok}_{gtok}_image.ome.tiff"
            msk_rel = f"annotations/masks/Kpneumoniae/{clean_plate}/{well}_{objtok}_{gtok}_mask.ome.tiff"
            img_dest = os.path.join(a.deposit_root, img_rel)
            msk_dest = os.path.join(a.deposit_root, msk_rel)

            frames = None
            if a.apply and not a.filelist_only and not os.path.lexists(img_dest):
                arr = L.omeImage(tif, img_dest, px, compression=a.compression); frames = arr.shape[0]
                if not L.verifyWritten(img_dest, arr):
                    print(f"  !! round-trip FAIL {img_rel}", file=sys.stderr); return 3
                L.omeMask(masknpz, msk_dest, px); made_mask += 1
                made_img += 1

            nF, iv, dur = cadence
            study_rows.append(dict(
                Files=img_rel, Plate=clean_plate, Well=well, Field=1,
                Objective=objtok, PixelSize_um=px, Frames=(frames or nF or ""),
                FrameInterval=(f"{iv} min" if iv else ""), TotalDuration=(f"{dur} h" if dur else ""),
                OriginalPlate=plate_base, OriginalWell=well_full,
                **B.kleb_row(plate_base, well, geno), **B.ACQUISITION, **prov))
            mask_rows.append(dict(
                Files=msk_rel, source_image=img_rel,
                AnnotationType="segmentation mask (binary)", AnnotationMethod="biofilm-processing pipeline",
                PixelValueMeaning="0 = background, 255 = biofilm"))

    print(f"\nfigure-analyzed wells: {len(study_rows)}   images written: {made_img}   "
          f"masks written: {made_mask}   collisions: {collisions}")
    print("cadence (frames, interval, duration):", cadence)

    fl_dir = os.path.join(a.deposit_root, "_filelists")
    sp = os.path.join(fl_dir, "kpneumoniae_study_filelist.tsv")
    mp = os.path.join(fl_dir, "kpneumoniae_masks_filelist.tsv")
    L.writeFileList(study_rows, B.KLEB_STUDY_COLUMNS, sp, apply=a.apply)
    L.writeFileList(mask_rows, B.ANNOTATION_COLUMNS, mp, apply=a.apply)

    if not a.apply:
        print("\n[DRY-RUN] nothing written. Sample study row:")
        if study_rows:
            for c in B.KLEB_STUDY_COLUMNS:
                print(f"   {c:16} = {study_rows[0].get(c,'')}")
        print("Re-run with --apply (optionally --limit N).")
    else:
        print(f"\nwrote file lists:\n  {sp}\n  {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
