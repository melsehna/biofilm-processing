#!/usr/bin/env python
"""One-shot: rename transposon deposit files to the new "<name>_<locus>" token.

Only affects named genes (locus-only / WT tokens are unchanged). Renames the
existing image/mask/labels OME-TIFFs in place on local disk (no reconversion),
old token -> new token. Dry-run by default; --apply to rename.

Run AFTER updating the transposon token logic in deposit_biology.vc_row, then
regenerate file lists (vcholerae --filelist-only + build_annotation_filelists).
"""
import argparse, os, re, sys
import deposit_lib as L
import deposit_biology as B

REIMAGING = "/mnt/phenotyper/Sehna/multiphenotype-data-06-25-26/reimagingData"
OBJTOK = "10x"


def norm(p):
    return re.sub(r"[\s_]+", "", p).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit-root", default="/mnt/data/bia_deposit/Brightfield")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = a.deposit_root

    # map norm(plate_base) -> sanitized clean_plate (deposit dir name), via a fast listdir
    rev = {}
    for pb in os.listdir(REIMAGING):
        if os.path.isdir(os.path.join(REIMAGING, pb, "processedImages")):
            rev[norm(pb)] = L.sanitizeToken(pb)

    members = B.vc_transposon_membership()
    renamed = missing = unchanged = 0
    examples = []
    for (nrm, well), bio in members.items():
        clean = rev.get(nrm)
        if not clean:
            continue
        old_tok = L.sanitizeToken(bio["mutant"])
        new_tok = L.sanitizeToken(B.vc_row("transposon", dict(bio))["_token"])
        if old_tok == new_tok:
            unchanged += 1
            continue
        for kind, sub in [("image", ""), ("mask", "annotations/masks/"), ("labels", "annotations/labels/")]:
            old = os.path.join(root, f"{sub}Vcholerae/transposon/{clean}/{well}_{OBJTOK}_{old_tok}_{kind}.ome.tiff")
            new = os.path.join(root, f"{sub}Vcholerae/transposon/{clean}/{well}_{OBJTOK}_{new_tok}_{kind}.ome.tiff")
            if not os.path.exists(old):
                missing += 1
                continue
            if os.path.lexists(new):
                print(f"  !! target exists, skipping: {new}", file=sys.stderr)
                continue
            if a.apply:
                os.rename(old, new)
            renamed += 1
            if len(examples) < 5:
                examples.append((os.path.basename(old), os.path.basename(new)))

    print(f"named-gene wells changed: token differs; unchanged (locus-only/WT): {unchanged}")
    print(f"files {'renamed' if a.apply else 'to rename'}: {renamed}   missing old files: {missing}")
    for o, n in examples:
        print(f"   {o}  ->  {n}")
    if not a.apply:
        print("\n[DRY-RUN] nothing renamed. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
