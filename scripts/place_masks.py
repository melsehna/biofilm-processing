#!/usr/bin/env python
"""Create-only placement of regenerated masks/labels onto the NAS.

Used after a regen_masks_*.sh run: it lifts ONLY the _masks.npz (and, with
--labels, _trackedLabels*.npz) out of the local regen scratch tree and creates
them next to the existing _processed.tif in the matching NAS processedImages/
dir. It never writes anything else.

SAFETY CONTRACT (enforced structurally, not by promise):
  * Only files whose basename ends in _masks.npz / matches _trackedLabels*.npz
    are ever considered as destinations. No other filename is referenced.
  * Every destination is created with O_CREAT|O_EXCL -> if it already exists the
    copy is refused, never overwritten.
  * This module contains NO destructive calls: no remove/unlink/rmtree/truncate,
    no open(...,'w'), no rename/replace/move over any path. Only exclusive-create
    + byte-write of new files (grep this file to verify).
  * A before/after manifest of every pre-existing file in each touched
    processedImages/ dir is compared; any change to a pre-existing file aborts.

Default mode is DRY-RUN (writes nothing). Pass --apply to actually create files.

  python scripts/place_masks.py --src-root <local-regen> --nas-root <nas-dataset>
  python scripts/place_masks.py --src-root ... --nas-root ... --labels --apply
"""
import argparse
import os
import sys

MASK_SUFFIX = "_masks.npz"
def is_target(name, include_labels):
    if name.endswith(MASK_SUFFIX):
        return True
    if include_labels and "_trackedLabels" in name and name.endswith(".npz"):
        return True
    return False

def find_src(src_root, include_labels):
    out = []
    for dp, _, fns in os.walk(src_root):
        if os.path.basename(dp) != "processedImages":
            continue
        for fn in fns:
            if is_target(fn, include_labels):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def index_nas_plates(nas_root):
    """Map plate-dir basename -> its processedImages path(s)."""
    m = {}
    for dp, _, _ in os.walk(nas_root):
        if os.path.basename(dp) == "processedImages":
            plate = os.path.basename(os.path.dirname(dp))
            m.setdefault(plate, []).append(dp)
    return m

def manifest(dirpath):
    d = {}
    for fn in os.listdir(dirpath):
        p = os.path.join(dirpath, fn)
        if os.path.isfile(p):
            st = os.stat(p); d[fn] = (st.st_size, st.st_mtime_ns)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="local scratch root of regenerated output")
    ap.add_argument("--nas-root", required=True, help="NAS dataset root to place into")
    ap.add_argument("--labels", action="store_true", help="also place _trackedLabels*.npz")
    ap.add_argument("--apply", action="store_true", help="actually create files (default: dry-run)")
    a = ap.parse_args()

    srcs = find_src(a.src_root, a.labels)
    nas = index_nas_plates(a.nas_root)
    print(f"source files found: {len(srcs)}  (labels={'yes' if a.labels else 'no'})")

    plan = []            # (src, dest)
    unresolved = []
    for s in srcs:
        plate = os.path.basename(os.path.dirname(os.path.dirname(s)))
        cands = nas.get(plate, [])
        if len(cands) != 1:
            unresolved.append((s, plate, len(cands))); continue
        plan.append((s, os.path.join(cands[0], os.path.basename(s))))

    collisions = [d for _, d in plan if os.path.lexists(d)]
    print(f"resolved: {len(plan)}   unresolved: {len(unresolved)}   "
          f"would-create: {len(plan)-len(collisions)}   COLLISIONS(existing): {len(collisions)}")
    for s, plate, n in unresolved[:10]:
        print(f"  UNRESOLVED ({n} NAS matches for plate {plate!r}): {os.path.basename(s)}")
    for d in collisions[:10]:
        print(f"  WOULD-COLLIDE (exists, will be SKIPPED not overwritten): {d}")

    if not a.apply:
        tot = sum(os.path.getsize(s) for s, _ in plan)
        print(f"\n[DRY-RUN] nothing written. {len(plan)-len(collisions)} files ({tot/1e6:.1f} MB) would be created.")
        print("Re-run with --apply to create them.")
        return 0

    if unresolved:
        print(f"\nABORT: {len(unresolved)} source files did not resolve to a unique NAS dir. "
              f"Fix mapping first.", file=sys.stderr)
        return 2
    if collisions:
        print(f"\nABORT: {len(collisions)} destinations already exist. "
              f"Refusing to touch an existing file. Resolve first.", file=sys.stderr)
        return 2

    # before-manifest of every touched dir
    touched = sorted({os.path.dirname(d) for _, d in plan})
    before = {t: manifest(t) for t in touched}

    import numpy as np
    created = 0
    for s, d in plan:
        with open(s, "rb") as fh:
            data = fh.read()
        fd = os.open(d, os.O_WRONLY | os.O_CREAT | os.O_EXCL)  # atomic no-clobber
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        with np.load(d) as _z:          # verify it reads back as a valid npz
            _ = list(_z.keys())
        created += 1
    print(f"created {created} new files.")

    # after-manifest: pre-existing files must be byte-identical
    violations = []
    for t in touched:
        aft = manifest(t)
        for fn, meta in before[t].items():
            if fn not in aft or aft[fn] != meta:
                violations.append(os.path.join(t, fn))
    if violations:
        print("!!! INTEGRITY VIOLATION: pre-existing files changed:", file=sys.stderr)
        for v in violations:
            print("   ", v, file=sys.stderr)
        return 3
    print("audit OK: every pre-existing file unchanged; only new files added.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
