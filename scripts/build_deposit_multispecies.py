#!/usr/bin/env python
"""Build the Multispecies study component of the BIA deposit.

210 wells (100% LB, 10x = mag _04), 8 species (per-well) -> species subdirs.
Image + mask OME-TIFFs (masks-only dataset; no instance labels). Study file list
only; global annotation list from build_annotation_filelists.py.

  dry-run:        python scripts/build_deposit_multispecies.py
  full:           python scripts/build_deposit_multispecies.py --apply
"""
import argparse, csv, os, re, sys, glob, json
import deposit_lib as L
import deposit_biology as B

NAS = "/mnt/phenotyper/Sehna/multispecies-data-062926"
DEFAULT_ROOT = "/mnt/data/bia_deposit/Brightfield"
MAG = "_04"                      # 10x
WELL_RE = re.compile(r"^([A-H]\d{1,2})")


def load_index(pidir):
    idx = os.path.join(pidir, "index.csv"); out = {}
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx)):
            out[r["well"]] = r
    return out


def load_params(pidir):
    p = os.path.join(pidir, "run_params.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    # multispecies used magParams; _04 override may hold blockDiam/fixedThresh
    mp = (d.get("magParams") or {}).get(MAG, {})
    pv = d.get("_pipelineVersion", {})
    return dict(blockDiam=mp.get("blockDiam", d.get("blockDiam", "")),
                fixedThresh=mp.get("fixedThresh", d.get("fixedThresh", "")),
                Pipeline="biofilm-processing", PipelineVersion=pv.get("version", ""),
                GitCommit=pv.get("gitCommit", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compression", default=None)
    ap.add_argument("--filelist-only", action="store_true", help="regenerate file list only; no conversion")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    members = B.ms_membership()
    pidirs = sorted(glob.glob(os.path.join(NAS, "*", "processedImages")))
    cadence = (None, None, None); study_rows = []; made = 0

    for pidir in pidirs:
        plate_base = os.path.basename(os.path.dirname(pidir))
        clean_plate = L.sanitizeToken(plate_base)
        idx = load_index(pidir); prov = load_params(pidir)
        raw_dir = next(iter(idx.values()), {}).get("plate_path", "") if idx else ""
        for tif in sorted(glob.glob(os.path.join(pidir, f"*{MAG}_processed.tif"))):
            well_full = os.path.basename(tif).replace("_processed.tif", "")
            if not well_full.endswith(MAG):
                continue
            m = WELL_RE.match(well_full); well = m.group(1) if m else well_full
            key = (re.sub(r"[\s_]+", "", plate_base).lower(), well)
            bio = members.get(key)
            if bio is None:
                continue
            if a.limit and made >= a.limit:
                break
            row = idx.get(well_full, {})
            px = float(row["pxToUm"]) if row.get("pxToUm") else None
            obj = row.get("objective", "10"); objtok = f"{obj}x"
            spdir = B.sanitize_species(bio["species"])
            masknpz = os.path.join(pidir, f"{well_full}_masks.npz")
            if cadence == (None, None, None) and raw_dir:
                cadence = L.imagingCadence(raw_dir, well_full)

            stem = f"{well}_{objtok}"
            img_rel = f"Multispecies/{spdir}/{clean_plate}/{stem}_image.ome.tiff"
            msk_rel = f"annotations/masks/Multispecies/{spdir}/{clean_plate}/{stem}_mask.ome.tiff"
            img_dest = os.path.join(a.deposit_root, img_rel)

            frames = None
            if a.apply and not a.filelist_only and not os.path.lexists(img_dest):
                arr = L.omeImage(tif, img_dest, px, compression=a.compression); frames = arr.shape[0]
                if not L.verifyWritten(img_dest, arr):
                    print(f"  !! round-trip FAIL {img_rel}", file=sys.stderr); return 3
                L.omeMask(masknpz, os.path.join(a.deposit_root, msk_rel), px)
            made += 1
            nF, iv, dur = cadence
            study_rows.append(dict(
                Files=img_rel, Plate=clean_plate, Well=well, Field=1,
                Objective=objtok, PixelSize_um=px, Frames=(frames or nF or ""),
                FrameInterval=(f"{iv} min" if iv else ""), TotalDuration=(f"{dur} h" if dur else ""),
                OriginalPlate=plate_base, OriginalWell=well_full,
                **B.ms_row(bio), **B.ACQUISITION, **prov))

    print(f"Multispecies wells: {len(study_rows)}   written: {made if a.apply else 0}   cadence={cadence}")
    sp = os.path.join(a.deposit_root, "_filelists", "multispecies_study_filelist.tsv")
    L.writeFileList(study_rows, B.MS_STUDY_COLUMNS, sp, apply=a.apply)
    if not a.apply and study_rows:
        print("[DRY-RUN] sample:", {c: study_rows[0].get(c) for c in ["Files", "Species", "PixelSize_um", "Frames"]})
    elif a.apply:
        print(f"wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
