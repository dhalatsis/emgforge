"""Cross-tier comparison for the spatial refactor.

Reads both tiers' ``data.npz`` (cylindrical = controlled/trusted reference, mri =
real Neurodec target), builds side-by-side comparison figures, prints a unified
metrics table comparing the two engines across both tiers.

Run:
    python \
        muap_generator/spatial_refactor/compare_tiers.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CYL = HERE / "datasets/cylindrical/data.npz"
MRI = HERE / "datasets/mri/data.npz"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def _norm(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


def align_score(t_ref, ref, t_spat, spat, max_lag_ms=12.0, dt=0.05):
    """Align spat to ref on a fine common grid via a cross-correlation lag
    search (robust, method-uniform across tiers). Returns (best_shift_ms,
    r_align). Positive shift means spat is moved later in time."""
    rn, sn = _norm(ref), _norm(spat)
    # coarse pre-alignment: match dominant extrema (bridges the centred-Fourier
    # vs physical-time axis gap in the cylindrical tier, ~46 ms)
    coarse = t_ref[np.argmax(np.abs(rn))] - t_spat[np.argmax(np.abs(sn))]
    # fine xcorr refinement around the coarse shift, on a common fine grid
    grid = np.arange(t_ref.min(), t_ref.max(), dt)
    rg = np.interp(grid, t_ref, rn, left=0.0, right=0.0)
    best_r, best_shift = -2.0, coarse
    for d in np.arange(-max_lag_ms, max_lag_ms + dt, dt):
        shift = coarse + d
        sg = np.interp(grid, t_spat + shift, sn, left=0.0, right=0.0)
        if sg.std() < 1e-12:
            continue
        r = pearsonr(rg, sg)[0]
        if r > best_r:
            best_r, best_shift = r, shift
    return best_shift, best_r


def lobe_metrics(t, m, win=25):
    """pos-before / pos-after the negative trough, on the normalised waveform."""
    mn = _norm(m)
    ti = int(np.argmin(mn))
    before = mn[max(0, ti - win):ti].max() if ti > 0 else 0.0
    after = mn[ti:ti + win].max()
    return t[ti], float(before), float(after)


def main():
    cyl = np.load(CYL, allow_pickle=True)
    mri = np.load(MRI, allow_pickle=True)
    cases = [str(c) for c in cyl["case_names"]]

    # ---- cylindrical metrics ----
    cyl_rows = []
    for n in cases:
        t_ref, ref = cyl[f"{n}__ref_t_ms"], cyl[f"{n}__ref_sfap"]
        t_sp, sp = cyl[f"{n}__spat_t_ms"], cyl[f"{n}__spat_sfap"]
        _, r = align_score(t_ref, ref, t_sp, sp)
        cyl_rows.append((n, float(cyl[f"{n}__L1"]), float(cyl[f"{n}__L2"]),
                         float(cyl[f"{n}__v"]), r))

    # ---- mri metrics ----
    pol = int(mri["selected_polarity"])
    muap = mri["muap_m1"] if pol == -1 else mri["muap_p1"]
    t_m = mri["t_ms"]
    neuro = mri["neurodec_muap"]; t_n = mri["t_ms_neuro"]; scale = float(mri["scale_factor"])
    lo, hi = max(t_m.min(), t_n.min()), min(t_m.max(), t_n.max())
    msk = (t_m >= lo) & (t_m <= hi)
    ni = np.interp(t_m[msk], t_n, neuro)
    r_raw = pearsonr(muap[msk], ni)[0]
    _, r_align = align_score(t_n, neuro, t_m, muap)
    tr_n, pb_n, pa_n = lobe_metrics(t_n, neuro)
    tr_s, pb_s, pa_s = lobe_metrics(t_m, muap)

    # ================= FIGURE 1: side-by-side tiers =================
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.24)

    # Row A — cylindrical (controlled, vs trusted analytical reference)
    for col, n in enumerate(["baseline_d14_sym", "pm_asym_d14"]):
        ax = fig.add_subplot(gs[0, col])
        t_ref, ref = cyl[f"{n}__ref_t_ms"], cyl[f"{n}__ref_sfap"]
        t_sp, sp = cyl[f"{n}__spat_t_ms"], cyl[f"{n}__spat_sfap"]
        shift, r = align_score(t_ref, ref, t_sp, sp)
        ax.plot(t_ref, _norm(ref), "k-", lw=2, label="analytical ref (trusted)")
        ax.plot(t_sp + shift, _norm(sp), "tab:blue", lw=1.4, label="spatial (aligned)")
        ax.axhline(0, color="k", lw=0.3)
        ax.set_xlim(t_ref[np.argmax(np.abs(ref))] - 18, t_ref[np.argmax(np.abs(ref))] + 18)
        L1 = float(cyl[f"{n}__L1"]); L2 = float(cyl[f"{n}__L2"])
        ax.set_title(f"CYL · {n}\nL1={L1:.0f} L2={L2:.0f}  ·  r_align={r:+.2f}", fontsize=10)
        ax.set_xlabel("t (ms, ref frame)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("normalised")

    ax = fig.add_subplot(gs[0, 2])
    names = [r[0].replace("_d14", "").replace("_sym", "") for r in cyl_rows]
    rs = [r[4] for r in cyl_rows]
    ax.barh(names, rs, color="tab:blue", alpha=0.8)
    for i, v in enumerate(rs):
        ax.text(v + 0.01, i, f"{v:+.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.axvline(0.99, color="gray", ls=":", label="perfect=0.99")
    ax.set_title("CYL · r_align vs trusted reference\n(5 cases)", fontsize=10)
    ax.set_xlabel("peak-aligned Pearson r"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    # Row B — MRI (real Neurodec target)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_n, _norm(neuro), "k-", lw=2, label="Neurodec (truth)")
    ax.plot(t_m, _norm(muap), "tab:red", lw=1.4, label=f"spatial (pol {pol:+d})")
    ax.axhline(0, color="k", lw=0.3); ax.set_xlim(0, 45)
    ax.set_title(f"MRI · spatial MUAP vs Neurodec\nr_raw={r_raw:+.2f}  r_align={r_align:+.2f}", fontsize=10)
    ax.set_xlabel("t (ms, physical)"); ax.set_ylabel("normalised"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_n, _norm(neuro), "k-", lw=2, label="Neurodec")
    ax.plot(t_m, _norm(muap), "tab:red", lw=1.4, label="spatial")
    ax.axvline(tr_n, color="gray", ls=":", lw=1, label=f"trough {tr_n:.1f} ms")
    ax.axhline(0, color="k", lw=0.3); ax.set_xlim(10, 30)
    ax.set_title("MRI · EOF zoom (trailing-lobe check)", fontsize=10)
    ax.set_xlabel("t (ms)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    labels = ["r (raw)", "pos-before", "pos-after"]
    spatial_v = [r_raw, pb_s, pa_s]
    neuro_v = [1.0, pb_n, pa_n]
    fourier_v = [0.737, 0.21, 0.06]
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w, neuro_v, w, label="Neurodec", color="k")
    ax.bar(x, spatial_v, w, label="spatial", color="tab:red")
    ax.bar(x + w, fourier_v, w, label="Fourier", color="tab:green", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_title("MRI · scorecard (vs Fourier pipeline)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Spatial refactor — cylindrical (controlled) vs MRI (Neurodec target)",
                 fontsize=13, y=0.97)
    fig.savefig(FIG / "compare_tiers.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {FIG/'compare_tiers.png'}")

    # ================= print unified table =================
    print("\n" + "=" * 78)
    print("CYLINDRICAL TIER — spatial vs TRUSTED analytical reference")
    print("=" * 78)
    print(f"{'case':20s} {'L1':>4} {'L2':>5} {'v':>4}  {'r_align':>8}")
    for n, L1, L2, v, r in cyl_rows:
        print(f"{n:20s} {L1:>4.0f} {L2:>5.0f} {v:>4.1f}  {r:>+8.3f}")
    print(f"  mean r_align = {np.mean([r[4] for r in cyl_rows]):+.3f}")

    print("\n" + "=" * 78)
    print("MRI TIER — spatial vs Neurodec (real target)")
    print("=" * 78)
    print(f"{'metric':22s} {'Neurodec':>9} {'spatial':>9} {'Fourier':>9}")
    print(f"{'Pearson r (raw)':22s} {1.000:>9.3f} {r_raw:>+9.3f} {0.737:>+9.3f}")
    print(f"{'Pearson r (aligned)':22s} {1.000:>9.3f} {r_align:>+9.3f} {'n/a':>9}")
    print(f"{'trough (ms)':22s} {tr_n:>9.1f} {tr_s:>9.1f} {18.6:>9.1f}")
    print(f"{'pos-before-trough':22s} {pb_n:>+9.3f} {pb_s:>+9.3f} {0.21:>+9.3f}")
    print(f"{'pos-after-trough':22s} {pa_n:>+9.3f} {pa_s:>+9.3f} {0.06:>+9.3f}")

    return dict(cyl_rows=cyl_rows, r_raw=r_raw, r_align=r_align,
                tr_n=tr_n, tr_s=tr_s, pb_n=pb_n, pb_s=pb_s, pa_n=pa_n, pa_s=pa_s)


if __name__ == "__main__":
    main()
