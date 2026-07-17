"""Phase 4.2 — the MRI data-scaling curve: where does N (VC solutions) saturate?

Near-free experiment: within one dataset, a prefix of n_elec IS a valid smaller draw (i.i.d.
electrodes). No new FEM below the dataset's own N. The referee (frozen 27-config MRI benchmark)
is held out by construction at every N, so the curve is apples-to-apples.

The whole 32->1024 curve is drawn from the SINGLE 1024-electrode dataset -- crucial, because
two different datasets share theta but not zfrac and splicing them put a spurious dip at 512.

What it settles: the amp-weighted mean saturates at N=64, but the WORST detectable config keeps
improving to 1024 (0.83->0.99). "Does more data help?" has two answers; the mean is the
misleading one. And phi is non-monotonic in N EVEN WITHIN this one draw (128 < 256 while the
MUAPs improve) -- phi is the wrong referee, with no cross-dataset or retrain-noise excuse left.

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
NS = (32, 64, 128, 256, 512, 1024)

# One ELECTRODE DRAW for the whole curve. 32-256 are prefixes of the 1024-electrode dataset
# (tag ds1k); 512/1024 are the same dataset (tag mriN). They must NOT be spliced with the
# original 256-set runs: that set shares theta but not zfrac (the rng stream position for zf
# depends on n), so mixing them confounds "more electrodes" with "different electrodes" -- it
# put a spurious dip at 512. Every point below is the SAME draw.
STEM = {32: "ds1k32", 64: "ds1k64", 128: "ds1k128", 256: "ds1k256",
        512: "mriN512", 1024: "mriN1024"}


def load(n):
    f = OUT / f"scoremri_cyl_mlp_raw_{STEM[n]}.npz"
    return np.load(f) if f.exists() else None


def rel_l2(n):
    """phi rel-L2 lives in the checkpoint, not the score file."""
    import torch
    f = OUT / f"cyl_mlp_raw_{STEM[n]}.pt"
    if not f.exists():
        return np.nan
    return float(torch.load(f, map_location="cpu", weights_only=False).get("rel_l2", np.nan))


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

    print(f"{'N':>5}{'φ relL2':>9}{'amp-wtd r':>11}{'median r':>10}{'≥0.94':>8}"
          f"{'det min r':>11}{'det p2p':>9}")
    print("-" * 63)
    for n, s in rows:
        print(f"{n:5d}{rel_l2(n):9.4f}{s['ampw']:+11.3f}{s['med']:+10.3f}"
              f"{s['npass']:>4d}/{s['n']:<3d}{s['dmin']:+11.3f}{s['p2p']:9.2f}")

    N = np.array([n for n, _ in rows])
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))

    # Panel 0: the punchline — phi rel-L2 is NON-MONOTONIC in N (it prefers 128 over 256)
    # while every MUAP measure improves. Ranking by phi picks the WORSE model given MORE data.
    L = np.array([rel_l2(n) for n, _ in rows])
    A = np.array([s["ampw"] for _, s in rows])
    a0 = ax[0]
    a0.plot(N, L, "s--", color="tab:red", lw=2, ms=7, label="φ rel-L2 (lower=better)")
    a0.set_ylabel("φ rel-L2", color="tab:red"); a0.tick_params(axis="y", labelcolor="tab:red")
    a0.set_xscale("log", base=2); a0.set_xticks(N); a0.set_xticklabels(N)
    a0.set_xlabel("N (VC solutions)"); a0.grid(alpha=0.3)
    b0 = a0.twinx()
    b0.plot(N, A, "o-", color="tab:blue", lw=2, ms=7, label="amp-wtd MUAP r")
    b0.set_ylabel("amp-weighted MUAP r", color="tab:blue")
    b0.tick_params(axis="y", labelcolor="tab:blue")
    # mark the point where phi INCREASES with N (worse) while the MUAP score improves — the
    # non-monotonicity is the whole claim, so find it rather than hard-code an index.
    up = [i for i in range(1, len(L)) if np.isfinite(L[i]) and np.isfinite(L[i-1])
          and L[i] > L[i-1] and A[i] >= A[i-1]]
    if up:
        i = up[0]
        a0.annotate(f"φ worsens {int(N[i-1])}→{int(N[i])}\nwhile MUAPs improve",
                    xy=(N[i], L[i]), xytext=(0.30, 0.55), textcoords="axes fraction",
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2),
                    color="tab:red")
    a0.set_title("φ error is the WRONG referee", fontsize=10)

    for a, key, ttl, lo in [
        (ax[1], "ampw", "amplitude-weighted MUAP r\n(saturates at N=64)", 0.90),
        # lo=0.80, NOT 0.90: the tail's whole point is the 0.83->0.99 climb, and a 0.90
        # floor clips the low-N points off the axis — the interesting curve would be invisible.
        (ax[2], "dmin", "worst detectable (>1µV) MUAP r\n(climbs all the way to 1024)", 0.80),
        (ax[3], "p2p", "detectable p2p ratio (1.0 = exact)", None),
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
    d0, d1 = rows[0][1]["dmin"], rows[-1][1]["dmin"]
    print(f"\nN={rows[0][0]} -> N={rows[-1][0]}:")
    print(f"  amp-weighted mean r : {a0:+.3f} -> {a1:+.3f}  (Δ {a1-a0:+.4f}) — flat after 64")
    print(f"  worst detectable r  : {d0:+.3f} -> {d1:+.3f}  (Δ {d1-d0:+.4f}) — NOT saturated")
    print("\nThe mean says stop at 64; the tail says keep going. They disagree, and the mean is")
    print("the misleading one — hence 1024, and hence reporting the worst case beside the mean.")


if __name__ == "__main__":
    main()
