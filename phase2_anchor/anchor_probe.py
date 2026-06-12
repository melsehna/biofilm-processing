#!/usr/bin/env python3
"""Phase 2 anchor pressure-test (ISSUES.md, Remediation Plan).

Throwaway probe — NOT pipeline code. Measures whether an in-silico background
(empty-agar) level is a stable photometric anchor:

  1. Stability across the timelapse: does the background estimate stay flat as the
     biofilm grows and confluences? (decides per-frame vs per-well/early-frame anchor)
  2. Cross-session drift: how far apart are the background levels of the SAME well
     imaged Oct-11 vs Oct-17? (quantifies the illumination/exposure drift the anchor
     must remove — background is biology-independent, so any gap is pure drift)
  3. Estimator comparison: histogram mode vs high percentiles (p90/p95/p99).

Brightfield here is bright agar (~full scale) with darker colonies, so "background"
is the bright peak. Registration is skipped: a few-px shift does not move a
histogram, and the background estimators are shift-invariant.
"""
import os, glob, csv
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiWellAnalysis.processing.analysis_main import _toBitDepthScaled

ROOT = "/mnt/bridgeslab/Good imaging data/Multi-phenotype training"
SESSIONS = {
    "Oct11": f"{ROOT}/241011_183053_4x_10x_20x_40x_Discontinuous_Drawer7 11-Oct-2024 16-30-30/241011_183053_Plate 1",
    "Oct17": f"{ROOT}/241017_183030_4x_10x_20x_40x_Discontinuous_Drawer7 17-Oct-2024 16-08-32/241017_183030_Plate 1",
}
WELLS = ["B5", "B9"]
SUFFIXES = {"_03": "10x", "_04": "20x"}
OUTDIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUTDIR, exist_ok=True)

NBINS = 2000  # histogram resolution on [0,1] for mode estimation


def loadStack(plateDir, well, suffix):
    pat = os.path.join(plateDir, f"{well}{suffix}_1_1_Bright Field_*.tif")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    frames = [tifffile.imread(f) for f in files]
    stack = np.stack(frames, axis=-1)          # (H, W, T)
    return _toBitDepthScaled(stack)            # float32 in [0,1]


def frameStats(img):
    """Background-level candidates + diagnostics for one frame (values in [0,1])."""
    flat = img.ravel()
    # histogram mode (peak bin center) — robust background estimate
    hist, edges = np.histogram(flat, bins=NBINS, range=(0.0, 1.0))
    mi = int(np.argmax(hist))
    mode = 0.5 * (edges[mi] + edges[mi + 1])
    # fraction of pixels within +/-1% of the mode = "how much background is left"
    bgFrac = float(np.mean(np.abs(flat - mode) <= 0.01))
    return {
        "mode": float(mode),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "median": float(np.median(flat)),
        "mean": float(flat.mean()),          # the BAD anchor — should track biology
        "bgFrac": bgFrac,
    }


def run():
    rows = []
    for session, plateDir in SESSIONS.items():
        for well in WELLS:
            for suffix, mag in SUFFIXES.items():
                stack = loadStack(plateDir, well, suffix)
                T = stack.shape[-1]
                for t in range(T):
                    s = frameStats(stack[..., t])
                    s.update(session=session, well=well, mag=mag, frame=t)
                    rows.append(s)
                print(f"{session} {well} {mag}: {T} frames done")

    # CSV
    cols = ["session", "well", "mag", "frame",
            "mode", "p90", "p95", "p99", "median", "mean", "bgFrac"]
    csvPath = os.path.join(OUTDIR, "anchor_stats.csv")
    with open(csvPath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", csvPath)

    # Plots: one row per (well, mag); columns = stability, cross-session mode, bgFrac
    estimators = ["mode", "p90", "p95", "p99", "mean"]
    for mag in SUFFIXES.values():
        for well in WELLS:
            fig, ax = plt.subplots(1, 3, figsize=(16, 4))
            for session, color in (("Oct11", "tab:blue"), ("Oct17", "tab:red")):
                sub = [r for r in rows if r["well"] == well
                       and r["mag"] == mag and r["session"] == session]
                sub.sort(key=lambda r: r["frame"])
                fr = [r["frame"] for r in sub]
                # panel 1: estimator stability across timelapse (this session)
                for est in estimators:
                    ls = "--" if est == "mean" else "-"
                    ax[0].plot(fr, [r[est] for r in sub], ls, label=f"{session}:{est}",
                               alpha=0.8)
                # panel 2: mode (chosen anchor) both sessions overlaid
                ax[1].plot(fr, [r["mode"] for r in sub], "-o", ms=3,
                           color=color, label=session)
                # panel 3: background fraction over time
                ax[2].plot(fr, [r["bgFrac"] for r in sub], "-",
                           color=color, label=session)
            ax[0].set_title(f"{well} {mag}: estimator stability")
            ax[0].set_xlabel("frame"); ax[0].set_ylabel("level [0,1]")
            ax[0].legend(fontsize=6, ncol=2)
            ax[1].set_title(f"{well} {mag}: mode anchor, cross-session")
            ax[1].set_xlabel("frame"); ax[1].set_ylabel("mode [0,1]")
            ax[1].legend(fontsize=8)
            ax[2].set_title(f"{well} {mag}: background pixel fraction")
            ax[2].set_xlabel("frame"); ax[2].set_ylabel("frac within 1% of mode")
            ax[2].legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(OUTDIR, f"anchor_{well}_{mag}.png")
            fig.savefig(p, dpi=110); plt.close(fig)
            print("wrote", p)

    # Cross-session drift summary (background level, biology-independent)
    print("\n=== cross-session background drift (mode, mean over frames) ===")
    for well in WELLS:
        for mag in SUFFIXES.values():
            vals = {}
            for session in SESSIONS:
                sub = [r["mode"] for r in rows if r["well"] == well
                       and r["mag"] == mag and r["session"] == session]
                vals[session] = float(np.mean(sub))
            drift = vals["Oct17"] - vals["Oct11"]
            rel = drift / vals["Oct11"] * 100
            print(f"  {well} {mag}: Oct11={vals['Oct11']:.4f}  "
                  f"Oct17={vals['Oct17']:.4f}  drift={drift:+.4f} ({rel:+.2f}%)")


if __name__ == "__main__":
    run()
