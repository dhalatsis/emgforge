"""Thorough head-to-head: SPATIAL vs FOURIER MUAP method, across both tiers.

Runs *both* pipelines on *identical* inputs and scores them with one shared set
of metrics, so the cylindrical (controlled) and MRI (Neurodec) tiers are directly
comparable.

  - Fourier  : emgforge.synthesis.fourier  (pare / radon / spe2; canonical winning
               conventions posz=0, polarity=-1, phi taper + Butterworth c0.03 o2)
  - Spatial  : emgforge.synthesis.spatial_refactor (CSD@phi; csd_derivative=2,
               upsample_factor=2)

Tier 1 (cylindrical, SFAP level): single fibre per case. Fourier ≈ ground truth
here (the analytical pipeline is validated r=0.997 vs MATLAB), so this measures
how closely the spatial engine reproduces the trusted Fourier method.

Tier 2 (MRI, MUAP level): sum 100 PM WR-FCU fibres. Both methods are scored
against the Neurodec ground truth, and against each other.

Outputs: figures/svf_*.png + prints all comparison tables.
Run: python \
        emgforge.synthesis/spatial_refactor/compare_spatial_vs_fourier.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.synthesis.fourier import (build_fourier_grids, build_spe2_iap_spectrum,
                                     radon_section, build_time_vector_ms,
                                     compute_C_from_phi_z)
from emgforge.synthesis.preprocessing import resample_centered_line, smooth_butterworth
from emgforge.synthesis.spatial_refactor import compute_sfap_spatial, SpatialConfig

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
FS, W = 2048.0, 256


# ---------------------------------------------------------------- methods
def _proc_phi(phi):
    p = phi.copy(); rN, lN = 10, 5
    p[-rN:] *= 0.5 * (1 + np.cos(np.pi * np.arange(rN) / rN))
    p[:lN] *= (0.5 * (1 + np.cos(np.pi * np.arange(lN) / lN)))[::-1]
    return smooth_butterworth(p, cutoff_norm=0.03, order=2)


def fourier_sfap(phi, dz, Lp, Ld, v, posz=0.0, polarity=-1):
    """Production Fourier SFAP with the canonical winning conventions.
    Returns (t_ms centred, sfap)."""
    zs = v * 1000.0 / FS
    pu = resample_centered_line(_proc_phi(phi), delta_s_mm=dz, w_out=W, delta_s_out_mm=zs)
    Ck, _ = compute_C_from_phi_z(pu, delta_s_mm=zs, apply_z_window=False)
    g = build_fourier_grids(w=W, fsamp=FS, v=v)
    ka, kb, kzt, kt = g['kalpha'], g['kbeta'], g['kz_t'], g['kt']
    sp, _ = build_spe2_iap_spectrum(w=W, fsamp=FS, v=v)
    pare = (np.exp(-1j*Lp/2*ka)*Lp*np.sinc(Lp/2*ka/np.pi)
            - np.exp(1j*Ld/2*kb)*Ld*np.sinc(Ld/2*kb/np.pi))
    Em = pare*np.exp(1j*kzt*posz)*(np.ones((W, 1))@Ck.reshape(1, -1))
    E1 = (1.0/v)*(np.ones((W, 1))@sp.reshape(1, -1)).T*Em*(1j*kzt)
    s = np.flip(radon_section(kt.T/(2*np.pi), kzt.T/(2*np.pi), E1.T, 0.0))
    t = build_time_vector_ms(w=W, fsamp=FS) - W/FS*1000.0/2
    return t, polarity*np.flip(s)


def spatial_sfap(phi, dz, Lp, Ld, v, polarity=-1, t_start_ms=0.0, posz_mm=0.0):
    # t_start_ms<0 adds pre-roll baseline before the NMJ fire — used for the
    # cylinder (NMJ under electrode → complex would otherwise pin to t=0).
    # posz_mm = NMJ position in the φ-array (MRI tier passes the iz_norm offset).
    cfg = SpatialConfig(v=v, polarity=polarity, fsamp=FS, w=W,
                        csd_derivative=2, upsample_factor=2, t_start_ms=t_start_ms,
                        fiber_window="boxcar")
    t, s, _ = compute_sfap_spatial(phi, dz, Lp, Ld, posz_mm=posz_mm, config=cfg)
    return t, s


# ---------------------------------------------------------------- metrics
def _norm(x):
    m = np.abs(x).max()
    return x / m if m > 0 else x


def align_score(t_ref, ref, t_b, b, max_lag_ms=12.0, dt=0.05):
    """Coarse-extremum + fine xcorr alignment. Returns (shift_ms, r_align)."""
    rn, bn = _norm(ref), _norm(b)
    coarse = t_ref[np.argmax(np.abs(rn))] - t_b[np.argmax(np.abs(bn))]
    grid = np.arange(t_ref.min(), t_ref.max(), dt)
    rg = np.interp(grid, t_ref, rn, left=0, right=0)
    best_r, best_s = -2.0, coarse
    for d in np.arange(-max_lag_ms, max_lag_ms + dt, dt):
        sg = np.interp(grid, t_b + coarse + d, bn, left=0, right=0)
        if sg.std() < 1e-12:
            continue
        r = pearsonr(rg, sg)[0]
        if r > best_r:
            best_r, best_s = r, coarse + d
    return best_s, best_r


def nrmse_aligned(t_ref, ref, t_b, b):
    shift, _ = align_score(t_ref, ref, t_b, b)
    rn = _norm(ref)
    bi = np.interp(t_ref, t_b + shift, _norm(b), left=0, right=0)
    return float(np.sqrt(np.mean((rn - bi)**2)))


def lobe_metrics(t, m, win_ms=12.0):
    mn = _norm(m); ti = int(np.argmin(mn))
    dt = t[1] - t[0]; win = max(int(win_ms / dt), 1)
    before = mn[max(0, ti-win):ti].max() if ti > 0 else 0.0
    after = mn[ti:ti+win].max()
    return float(t[ti]), float(before), float(after)


def jaggedness(m):
    return float(np.mean(np.abs(np.diff(m, 2))) / m.ptp()) if m.ptp() > 0 else 0.0


def raw_r_vs(t, m, t_g, g):
    lo, hi = max(t.min(), t_g.min()), min(t.max(), t_g.max())
    msk = (t >= lo) & (t <= hi)
    gi = np.interp(t[msk], t_g, g)
    return float(pearsonr(m[msk], gi)[0])


# ================================================================ TIER 1
def tier_cylinder():
    cyl = np.load(HERE / "datasets/cylindrical/data.npz", allow_pickle=True)
    cases = [str(c) for c in cyl["case_names"]]
    rows = []
    fig, axes = plt.subplots(1, len(cases), figsize=(4*len(cases), 4), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), cases):
        phi = cyl[f"{n}__phi"]; dz = float(cyl[f"{n}__dz_mm"])
        Lp, Ld, v = float(cyl[f"{n}__L1"]), float(cyl[f"{n}__L2"]), float(cyl[f"{n}__v"])
        tf, sf = fourier_sfap(phi, dz, Lp, Ld, v, polarity=1)   # treat Fourier as ref/truth
        ts, ss = spatial_sfap(phi, dz, Lp, Ld, v, polarity=1, t_start_ms=-12.0)  # pre-roll
        # sign-match spatial to fourier for a clean shape comparison
        _, r_try = align_score(tf, sf, ts, ss)
        if r_try < 0:
            ss = -ss
        shift, r = align_score(tf, sf, ts, ss)
        nrm = nrmse_aligned(tf, sf, ts, ss)
        rows.append((n, Lp, Ld, v, r, nrm))
        ax.plot(tf, _norm(sf), "k-", lw=2, label="Fourier (≈truth)")
        ax.plot(ts + shift, _norm(ss), "tab:blue", lw=1.4, label="spatial")
        c = tf[np.argmax(np.abs(sf))]
        ax.set_xlim(c-16, c+16); ax.axhline(0, color="k", lw=0.3)
        ax.set_title(f"{n}\nr={r:+.2f} nrmse={nrm:.2f}", fontsize=9)
        ax.set_xlabel("t (ms, aligned)"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8); np.atleast_1d(axes)[0].set_ylabel("normalised")
    fig.suptitle("TIER 1 cylinder — spatial vs Fourier (Fourier ≈ ground truth, r=0.997 vs MATLAB)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"svf_cylinder.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return rows


# ================================================================ TIER 2
def tier_mri():
    F = np.load(HERE.parent/"datasets/pm/fibres.npz")
    gt = np.load(HERE.parent/"datasets/pm/ground_truth.npz")
    neuro, t_n = gt["neurodec_muap"], gt["t_ms"]
    N = F["phi_raw"].shape[0]
    n_phi = F["phi_raw"].shape[1]
    iz = F["iz_norm"]
    fou = np.zeros(W); spa = np.zeros(W); t_f = t_s = None
    for i in range(N):
        phi = F["phi_raw"][i].astype(float); dz = float(F["dz_mm"][i])
        Lp, Ld, v = float(F["L_prox_mm"][i]), float(F["L_dist_mm"][i]), float(F["v_m_per_s"][i])
        t_f, sf = fourier_sfap(phi, dz, Lp, Ld, v, polarity=-1); fou += sf
        posz = (float(iz[i]) * (n_phi - 1) - n_phi // 2) * dz
        t_s, ss = spatial_sfap(phi, dz, Lp, Ld, v, polarity=-1, posz_mm=posz); spa += ss

    def vs_truth(t, m):
        r_raw = raw_r_vs(t, m, t_n, neuro)
        _, r_al = align_score(t_n, neuro, t, m)
        nrm = nrmse_aligned(t_n, neuro, t, m)
        tr, pb, pa = lobe_metrics(t, m)
        # amplitude ratio: ptp on common window vs Neurodec×1.24
        lo, hi = max(t.min(), t_n.min()), min(t.max(), t_n.max()); msk = (t >= lo)&(t <= hi)
        amp = m[msk].ptp()/(neuro*float(gt["scale_factor_for_100_vs_full"])).ptp()
        return dict(r_raw=r_raw, r_align=r_al, nrmse=nrm, trough=tr, pb=pb, pa=pa,
                    amp=amp, jag=jaggedness(m))
    mf = vs_truth(t_f, fou); ms = vs_truth(t_s, spa)
    # method-vs-method
    _, r_methods = align_score(t_f, fou, t_s, spa)
    tr_n, pb_n, pa_n = lobe_metrics(t_n, neuro)

    # ---- figure: full + zoom + method-vs-method ----
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(t_n, _norm(neuro), "k-", lw=2.2, label="Neurodec (truth)")
    ax[0].plot(t_f, _norm(fou), "tab:green", lw=1.4, label=f"Fourier (r={mf['r_raw']:+.2f})")
    ax[0].plot(t_s, _norm(spa), "tab:red", lw=1.4, label=f"spatial (r={ms['r_raw']:+.2f})")
    ax[0].set_xlim(0, 45); ax[0].axhline(0, color="k", lw=0.3); ax[0].legend(fontsize=9)
    ax[0].set_title("MRI MUAP — both methods vs Neurodec"); ax[0].set_xlabel("t (ms)")
    ax[0].set_ylabel("normalised"); ax[0].grid(alpha=0.3)

    ax[1].plot(t_n, _norm(neuro), "k-", lw=2.2, label="Neurodec")
    ax[1].plot(t_f, _norm(fou), "tab:green", lw=1.4, label="Fourier")
    ax[1].plot(t_s, _norm(spa), "tab:red", lw=1.4, label="spatial")
    ax[1].axvline(tr_n, color="gray", ls=":", lw=1)
    ax[1].set_xlim(10, 30); ax[1].axhline(0, color="k", lw=0.3); ax[1].legend(fontsize=9)
    ax[1].set_title("EOF zoom — trailing-lobe check"); ax[1].set_xlabel("t (ms)"); ax[1].grid(alpha=0.3)

    labels = ["r(raw)", "r(align)", "pos-after"]
    fv = [mf["r_raw"], mf["r_align"], mf["pa"]]
    sv = [ms["r_raw"], ms["r_align"], ms["pa"]]
    nv = [1.0, 1.0, pa_n]
    x = np.arange(len(labels)); w = 0.26
    ax[2].bar(x-w, nv, w, color="k", label="Neurodec")
    ax[2].bar(x, sv, w, color="tab:red", label="spatial")
    ax[2].bar(x+w, fv, w, color="tab:green", label="Fourier")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].axhline(0, color="k", lw=0.3)
    ax[2].set_title("scorecard vs Neurodec"); ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"TIER 2 MRI — spatial vs Fourier vs Neurodec  (method-vs-method r={r_methods:+.2f})",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG/"svf_mri.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return mf, ms, r_methods, (tr_n, pb_n, pa_n)


def main():
    print("="*86)
    print("TIER 1 — CYLINDER (SFAP): spatial vs Fourier (Fourier ≈ truth, r=0.997 vs MATLAB)")
    print("="*86)
    cyl = tier_cylinder()
    print(f"{'case':20s} {'L1':>4} {'L2':>5} {'v':>4}  {'r(spat~four)':>12} {'nrmse':>7}")
    for n, Lp, Ld, v, r, nrm in cyl:
        print(f"{n:20s} {Lp:>4.0f} {Ld:>5.0f} {v:>4.1f}  {r:>+12.3f} {nrm:>7.3f}")
    print(f"  mean r = {np.mean([r[4] for r in cyl]):+.3f}   mean nrmse = {np.mean([r[5] for r in cyl]):.3f}")

    print("\n" + "="*86)
    print("TIER 2 — MRI (MUAP): spatial vs Fourier, both vs Neurodec")
    print("="*86)
    mf, ms, rmm, (tr_n, pb_n, pa_n) = tier_mri()
    hdr = f"{'metric':18s} {'Neurodec':>9} {'spatial':>9} {'Fourier':>9}   winner"
    print(hdr); print("-"*len(hdr))
    def line(name, key, nv, hi_good=True, fmt="%+9.3f"):
        s, f = ms[key], mf[key]
        win = "spatial" if (s > f) == hi_good else "Fourier"
        if abs(s-f) < 1e-6: win = "tie"
        print(f"{name:18s} {nv:>9} {fmt%s:>9} {fmt%f:>9}   {win}")
    line("Pearson r (raw)", "r_raw", "1.000", True)
    line("Pearson r (align)", "r_align", "1.000", True)
    line("NRMSE (aligned)", "nrmse", "0.000", False)
    line("pos-after-trough", "pa", f"{pa_n:+.3f}", True)   # closer-to-truth ≈ higher here
    print(f"{'pos-before-trough':18s} {pb_n:>+9.3f} {ms['pb']:>+9.3f} {mf['pb']:>+9.3f}")
    print(f"{'trough (ms)':18s} {tr_n:>9.1f} {ms['trough']:>9.1f} {mf['trough']:>9.1f}   (Neuro {tr_n:.1f})")
    print(f"{'amplitude ratio':18s} {1.0:>9.3f} {ms['amp']:>+9.3f} {mf['amp']:>+9.3f}")
    print(f"{'jaggedness':18s} {'-':>9} {ms['jag']:>9.3f} {mf['jag']:>9.3f}   (lower=smoother)")
    print(f"\nmethod-vs-method (spatial vs Fourier), peak-aligned r = {rmm:+.3f}")
    print(f"\n✓ figures: {FIG}/svf_cylinder.png , {FIG}/svf_mri.png")
    return cyl, mf, ms, rmm, (tr_n, pb_n, pa_n)


if __name__ == "__main__":
    main()
