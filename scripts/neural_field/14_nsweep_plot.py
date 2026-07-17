"""Phase 4.2 — the MRI data-scaling curve: where does N (VC solutions) saturate?

Free experiment: the 256-electrode dataset's electrodes are i.i.d. random, so a prefix of
n_elec IS a valid smaller draw. No new FEM. The referee (frozen 27-config MRI benchmark) is
held out by construction at every N, so the curve is apples-to-apples.

The question this settles: the cluster plan assumed "scale N to 2048 and the rankings may
invert". If the curve is already flat at N=64, there is nothing at 2048 to find, and the
cylinder's target drama was never about N at all -- it was about WHICH points we sampled.

Run: PYTHONPATH=src python scripts/neural_field/14_nsweep_plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
NS = (32, 64, 128, 256)


def load(tag):
    f = OUT / f"scoremri_cyl_mlp_raw_mriN{tag}.npz"
    return np.load(f) if f.exists() else None


def summarise(d):
    r, amp = d["r"], d["amp"]
    w = amp / amp.sum()
    det = amp >= 1.0
    return dict(ampw=float((w * r).sum()), med=float(np.median(r)),
                dmin=float(r[det].min()), dmed=float(np.median(r[det])),
                p2p=float(np.median(d["p2p_ratio"][det])), npass=int((r >= 0.94).sum()),
                n=len(r))


def main():
    rows = [(n, summarise(d)) for n in NS if (d := load(n)) is not None]
    if not rows:
        print("no N-sweep scores found — run the sweep first")
        return

    print(f"{'N':>5}{'amp-wtd r':>11}{'median r':>10}{'≥0.94':>8}"
          f"{'det min r':>11}{'det p2p':>9}")
    print("-" * 54)
    for n, s in rows:
        print(f"{n:5d}{s['ampw']:+11.3f}{s['med']:+10.3f}{s['npass']:>4d}/{s['n']:<3d}"
              f"{s['dmin']:+11.3f}{s['p2p']:9.2f}")

    N = np.array([n for n, _ in rows])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, key, ttl, lo in [
        (ax[0], "ampw", "amplitude-weighted MUAP r", 0.9),
        (ax[1], "dmin", "worst detectable (>1µV) MUAP r", 0.9),
        (ax[2], "p2p", "detectable p2p ratio (1.0 = exact)", None),
    ]:
        y = np.array([s[key] for _, s in rows])
        a.plot(N, y, "o-", lw=2, ms=7, color="#1f77b4")
        a.set_xscale("log", base=2); a.set_xticks(N); a.set_xticklabels(N)
        a.set_xlabel("N (VC solutions / electrodes)"); a.set_title(ttl, fontsize=10)
        a.grid(alpha=0.3)
        if key == "p2p":
            a.axhline(1.0, color="k", ls="--", lw=1, label="exact")
            a.legend(fontsize=8)
        else:
            a.axhline(0.94, color="r", ls="--", lw=1, label="Gate 4")
            a.set_ylim(lo, 1.005); a.legend(fontsize=8, loc="lower right")
    fig.suptitle("MRI data-scaling: does N matter once the sampling is right?", fontsize=11)
    fig.tight_layout()
    p = OUT / "nsweep_mri.png"
    fig.savefig(p, dpi=140)
    print(f"\nwrote {p}")

    a0, a1 = rows[0][1]["ampw"], rows[-1][1]["ampw"]
    print(f"\nN={rows[0][0]} -> N={rows[-1][0]}: amp-wtd r {a0:+.3f} -> {a1:+.3f} "
          f"(Δ {a1-a0:+.4f})")
    print("If Δ is within noise, the 2048-electrode cluster race has nothing to find.")


if __name__ == "__main__":
    main()
