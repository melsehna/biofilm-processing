#!/usr/bin/env python
"""Build the V. cholerae study component of the BIA deposit.

One study component, 4 perturbation subdirs from 4 source datasets:
  knownMutant   <- training  (fig1)   nested drawer/plate, mag _03
  transposon    <- reimaging (fig3)   mag _02   (+ gene locus/function/group)
  cleanDeletion <- cluster   (fig4)   mag _02
  compound      <- cluster   (fig5)   mag _02
Emits float32 image + uint8 mask + uint16 instance-label OME-TIFFs. Figure-analyzed
scope (well present in the perturbation's table). Study file list only; the 2 global
annotation file lists come from build_annotation_filelists.py.

  dry-run:            python scripts/build_deposit_vcholerae.py
  one perturbation:   python scripts/build_deposit_vcholerae.py --only transposon --apply
  full component:     python scripts/build_deposit_vcholerae.py --apply
"""
import argparse, csv, os, re, sys, glob, json
import deposit_lib as L
import deposit_biology as B

DEFAULT_ROOT = "/mnt/data/bia_deposit/Brightfield"
WELL_RE = re.compile(r"^([A-H]\d{1,2})")

SOURCES = [
    dict(pert="knownMutant",   nas="/mnt/phenotyper/Sehna/multiphenotype-data-061426",     mag="_03", membership=B.vc_known_membership),
    dict(pert="transposon",    nas="/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26",   mag="_02", membership=B.vc_transposon_membership),
    dict(pert="cleanDeletion", nas="/mnt/phenotyper/Sehna/cluster-data-062726",            mag="_02", membership=B.vc_cleandel_membership),
    dict(pert="compound",      nas="/mnt/phenotyper/Sehna/cluster-data-062726",            mag="_02", membership=B.vc_compound_membership),
]


def load_index(pidir):
    idx = os.path.join(pidir, "index.csv"); out = {}
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx)):
            out[r["well"]] = r
    return out


def load_params(pidir):
    p = os.path.join(pidir, "run_params.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    pv = d.get("_pipelineVersion", {})
    return dict(blockDiam=d.get("blockDiam", ""), fixedThresh=d.get("fixedThresh", ""),
                Pipeline="biofilm-processing", PipelineVersion=pv.get("version", ""),
                GitCommit=pv.get("gitCommit", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default=DEFAULT_ROOT)
    ap.add_argument("--only", choices=[s["pert"] for s in SOURCES], help="build one perturbation")
    ap.add_argument("--limit", type=int, default=None, help="first N wells per perturbation (pilot)")
    ap.add_argument("--compression", default=None)
    ap.add_argument("--filelist-only", action="store_true", help="regenerate file list only; no image conversion")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    study_rows = []
    totals = {}
    for src in SOURCES:
        if a.only and src["pert"] != a.only:
            continue
        pert, mag = src["pert"], src["mag"]
        members = src["membership"]()
        pidirs = sorted(glob.glob(os.path.join(src["nas"], "**", "processedImages"), recursive=True))
        cadence = (None, None, None)
        made = 0
        for pidir in pidirs:
            plate_base = os.path.basename(os.path.dirname(pidir))
            clean_plate = L.sanitizeToken(plate_base)
            idx = load_index(pidir); prov = load_params(pidir)
            raw_dir = next(iter(idx.values()), {}).get("plate_path", "") if idx else ""
            for tif in sorted(glob.glob(os.path.join(pidir, f"*{mag}_processed.tif"))):
                well_full = os.path.basename(tif).replace("_processed.tif", "")
                if not well_full.endswith(mag):
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
                masknpz = os.path.join(pidir, f"{well_full}_masks.npz")
                lblnpz = glob.glob(os.path.join(pidir, f"{well_full}_trackedLabels*.npz"))
                if cadence == (None, None, None) and raw_dir:
                    cadence = L.imagingCadence(raw_dir, well_full)

                vc = B.vc_row(pert, bio); tok = L.sanitizeToken(vc.pop("_token"))
                stem = f"{well}_{objtok}_{tok}"
                img_rel = f"Vcholerae/{pert}/{clean_plate}/{stem}_image.ome.tiff"
                msk_rel = f"annotations/masks/Vcholerae/{pert}/{clean_plate}/{stem}_mask.ome.tiff"
                lbl_rel = f"annotations/labels/Vcholerae/{pert}/{clean_plate}/{stem}_labels.ome.tiff"
                img_dest = os.path.join(a.deposit_root, img_rel)

                frames = None
                if a.apply and not a.filelist_only and not os.path.lexists(img_dest):
                    arr = L.omeImage(tif, img_dest, px, compression=a.compression); frames = arr.shape[0]
                    if not L.verifyWritten(img_dest, arr):
                        print(f"  !! round-trip FAIL {img_rel}", file=sys.stderr); return 3
                    L.omeMask(masknpz, os.path.join(a.deposit_root, msk_rel), px)
                    if lblnpz:
                        L.omeLabels(lblnpz[0], os.path.join(a.deposit_root, lbl_rel), px)
                made += 1
                nF, iv, dur = cadence
                study_rows.append(dict(
                    Files=img_rel, Plate=clean_plate, Well=well, Field=1,
                    Objective=objtok, PixelSize_um=px, Frames=(frames or nF or ""),
                    FrameInterval=(f"{iv} min" if iv else ""), TotalDuration=(f"{dur} h" if dur else ""),
                    OriginalPlate=plate_base, OriginalWell=well_full,
                    **vc, **B.ACQUISITION, **prov))
        totals[pert] = made if a.apply else sum(1 for r in study_rows if f"/{pert}/" in r["Files"])
        print(f"  {pert}: {'wrote' if a.apply else 'planned'} {totals[pert]} wells   cadence={cadence}")

    print(f"\nV. cholerae total wells: {len(study_rows)}")
    sp = os.path.join(a.deposit_root, "_filelists", "vcholerae_study_filelist.tsv")
    L.writeFileList(study_rows, B.VC_STUDY_COLUMNS, sp, apply=a.apply)
    if not a.apply:
        print("[DRY-RUN] nothing written. Sample row:")
        if study_rows:
            for c in B.VC_STUDY_COLUMNS[:16]:
                print(f"   {c:18} = {study_rows[0].get(c,'')}")
    else:
        print(f"wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
