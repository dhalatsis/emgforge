#!/usr/bin/env python
"""Waveforms behind the mesh-convergence study, and the 3-monopole model on the MRI forearm.

    P=/home/dc23/miniconda3/envs/fenicsx-env/bin/python
    $P scripts/validation/mesh_convergence_waveforms.py            # reads the level caches only

Reads the per-level caches of ``mesh_convergence.py`` (no finite-element solve) and writes

* ``_results/validation/mesh_convergence/waveforms.json`` — the actual φ(z), 3-monopole fit,
  residual, φ″ and direct-recipe SFAP waveforms (cylinder vs the analytical oracle; MRI
  forearm across the five mesh levels) plus the forearm summary (i)–(iv), see ``STRUCTURE``;
* ``_results/validation/mesh_convergence/waveforms_preview.png`` — a sanity-check figure.

Cylinder (σ_s = 5 mm; depths 7 / 10 / 13 / 20 mm; meshes 0.42 / 0.3 / 0.2; plus σ_s = 1 mm at
10 mm on 0.3 / 0.2): φ is DC-freed (mean of the 12 edge samples of the RAW φ, the same constant
subtracted from the fit) and normalised to the FEM centre value, the analytical φ to its own
centre value, so the shapes overlay; φ″ of the three unit-peak φ's is divided by max|φ″_ana| on
the core; the SFAPs (direct recipe on the un-normalised φ, then scaled by the same centre
values) are divided by p2p of the analytical SFAP — so p2p(sfap_fit) IS the FEM/analytical
amplitude ratio at matched φ peak.

Forearm: centre electrode (index 12) of the 5×5 grid; six bed fibres at increasing depth
below the electrode (closest approach ≈ 4 / 6 / 8 / 10 / 12.5 / 16 mm; the four shallow ones
< 1.5 mm off the inward ray, the two deep ones 6 / 12.6 mm lateral — the FCU is only ~11 mm thick
under the electrode); φ in mV, φ″ in mV/mm², SFAP in µV (single fibre, NMJ at ``IZ_FRAC_FCU`` of the
arc length without the pool's per-unit jitter, ``production_config`` time base).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EMGFORGE_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "_results/validation/mesh_convergence"

import cyl_fem                                   # noqa: E402
import harness as H                              # noqa: E402
import mesh_convergence as MC                    # noqa: E402
from emgforge.mri.pipeline import IZ_FRAC_FCU, production_config   # noqa: E402
from emgforge.synthesis.engines.spatial import compute_sfap_spatial  # noqa: E402
from emgforge.synthesis.preprocessing import denoise_field_n         # noqa: E402

CYL_DEPTHS = [7.0, 10.0, 13.0, 20.0]
CYL_MESHES = [0.42, 0.3, 0.2]
CYL_S1 = [(10.0, 0.3), (10.0, 0.2)]
FA_LEVELS = [0.05, 0.03, 0.02, 0.015, 0.012]
FA_TARGET_DEPTHS = [4.0, 6.0, 8.0, 10.0, 13.0, 16.0]
E0 = 12                                          # centre electrode of the 5×5 grid
HALF_Z = 80.0                                    # stored φ / φ″ window (mm)
CORE = 60.0                                      # correlation core (mm)
T_CYL = (-5.0, 35.0)                             # stored SFAP window (ms)
T_FA = (-5.0, 45.0)
WORKERS = 4
VARIANTS = ["none", "mono3", "mono5", "mono7", "bw03"]

STRUCTURE = """
cyl[depth]         depth in {7,10,13,20} (mm below the skin). cyl[depth]["ana"] holds what every mesh shares:
   z              (mm) along the fibre, electrode at z = 0, |z| ≤ 77.5 mm (159 samples)
   phi_ana        analytical (Farina 2004) φ, DC-free (edge mean), / its centre value
   dd_ana         its second z-derivative, / max|dd_ana| on |z| ≤ 60 (the normaliser of every dd_* at this depth)
   t              (ms) physical time, NMJ fires at 0 (fibre 60 + 60 mm, NMJ at −20 mm), −5 … 35 ms
   sfap_ana       direct recipe on the analytical φ, normalised to p2p = 1
cyl[depth][mesh]   mesh in {0.42,0.3,0.2} (σ_s = 5 mm) and, at depth 10, {0.3_s1, 0.2_s1} (σ_s = 1 mm). Each:
   phi_raw        FEM φ, DC-free (edge mean of the raw φ), / its centre value       — on cyl[depth]["ana"]["z"]
   phi_fit        the 3-monopole fit, same constant and scale
   resid          phi_raw − phi_fit (same units)
   dd_raw, dd_fit second z-derivatives of the unit-peak φ's, / max|dd_ana|
   sfap_raw       direct recipe with denoise="none" on the raw FEM φ, / (FEM centre value · p2p(sfap_ana / ana centre value)) — on ["ana"]["t"]
   sfap_fit       direct recipe (production: 3-monopole fit) on the raw FEM φ, same normalisation
   phi_centre_fem the centre value used (V); phi_centre_ana is under ["ana"]
   scale_ls_fem_over_ana            least-squares φ_FEM / φ_ana on the core (the study's 'scale')
   p2p_fit_over_ana, p2p_raw_over_ana, p2p_raw_over_fit   SFAP amplitude ratios (as stored)
   r_phi_ana, r_dd_fit_ana, r_dd_raw_ana, r_sfap_fit_ana, r_sfap_raw_ana   correlations on the core / the SFAP window
   resid_rms_rel_peak               rms(phi_raw − phi_fit) / |centre value|
cyl_summary        r of the fit residual between successive cylinder meshes (0.6→0.42→0.3→0.24→0.2) and vs 0.2,
                   per depth, σ_s = 5 and 1 mm; a residual that is the same on every mesh is model misfit.
mri[fibre][level]  fibre = "0".."5" (shallow → deep), level in {0.05,0.03,0.02,0.015,0.012}. Each:
   phi_raw, phi_fit (mV), resid (µV)   along the fibre, |z| ≤ 80 mm (z, depth_mm, fibre_index, lateral_mm, len1_mm,
                              len2_mm, posz_mm, dz_mm, t, lambda_mm are under mri[fibre]["meta"], shared by the levels)
   dd_raw, dd_fit            (µV/mm²)
   sfap_raw (denoise none) / sfap_fit (production, 3 monopoles)   (µV) on mri[fibre]["meta"]["t"], −5 … 45 ms
   sfap_mono5 / sfap_mono7 / sfap_bw03   (µV) at the current (0.03) and finest (0.012) level only
   p2p_uV[variant]           SFAP p2p of every variant at every level
   resid_rms_rel_p2p         rms(resid) / p2p(phi_raw)
   resid_spectrum            |Hann-tapered rfft(resid)| / p2p(phi_raw) on ["meta"]["lambda_mm"], at 0.03 and 0.012
mri_summary
   levels, variants, fibres (index, depth_mm, lateral_mm)
   i_resid        per fibre: rms_rel_p2p[level], r_succ[pair], r_vs_finest[level]
   ii_ddfit       per fibre: r_succ[pair], r_vs_finest[level] (fitted φ″ on |z| ≤ 60); r_succ_raw for contrast
   iii_sfap       per fibre per variant: p2p_ratio_succ[pair], r_succ[pair], p2p_vs_finest[level], r_vs_finest[level];
                  shape_cost per fibre per variant: r and p2p ratio vs mono3 on the finest mesh;
                  bed: the same over all 637 bed fibres at the centre electrode — median / p90 |p2p change| and
                  median / p10 r per pair and per variant, and the shape cost medians
   iv_spectrum    per fibre per level: peak wavelength (< 40 mm) and band power fractions of the residual;
                  bed_mean[level]: mean amplitude spectrum over the bed on lambda_mm
   bed_scatter    depth_mm (closest approach to the centre electrode), rms_rel_p2p_e0.03, rms_rel_p2p_e0.012 and
                  r_resid_0.03_vs_0.012 for all 637 fibres; depth_bins: medians per 2 mm bin
"""


# --------------------------------------------------------------------------- helpers
def rl(a, sig=4):
    """Array → list of floats rounded to ``sig`` significant digits."""
    return [None if not np.isfinite(x) else float(f"{x:.{sig}g}") for x in np.asarray(a, float).ravel()]


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-300 or b.std() < 1e-300:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dd(p, dz):
    return np.gradient(np.gradient(p, dz), dz)


def edge_mean(p, n=12):
    return float(np.mean(np.r_[p[:n], p[-n:]]))


def spectrum(res, dz):
    """Hann-tapered amplitude spectrum of a residual along z → (lambda_mm, amp)."""
    e = np.asarray(res, float)
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fq = np.fft.rfftfreq(len(e), dz)
    return 1.0 / fq[1:], E[1:]


def band_fracs(lam, amp):
    P = amp ** 2
    bands = {"gt40": lam > 40, "15to40": (lam > 15) & (lam <= 40), "6to15": (lam > 6) & (lam <= 15), "lt6": lam <= 6}
    return {k: float(P[m].sum() / P.sum()) for k, m in bands.items()}


# --------------------------------------------------------------------------- part A: cylinder
def part_cyl():
    DZ = H.V * 1000.0 / H.FS
    Wn = H.W
    z = (np.arange(Wn) - Wn // 2) * DZ
    keep = np.abs(z) <= 77.5                 # 159 samples at dz = 0.977 mm (≤ 160 over ±80 mm)
    core = np.abs(z) <= CORE
    ana = MC.Ana()
    cfg_fit, cfg_none = H.golden_cfg(), H.golden_cfg(denoise="none")
    levels = {}
    for f in sorted(OUT.glob("cyl_f*.npz")):
        d = dict(np.load(f))
        if str(d["variant"]) == "":
            levels[float(d["factor"])] = d
    lv0 = next(iter(levels.values()))
    z_abs, ze, radii, thetas = lv0["z_abs"], float(lv0["ze"]), lv0["radii"], lv0["thetas"]
    i_th0 = int(np.argmin(np.abs(thetas)))

    def fem_phi(d, s, dep):
        i = int(np.argmin(np.abs(radii - (40.0 - dep))))
        return cyl_fem.window(d[f"phi_s{s:g}"][i, i_th0], z_abs, ze, Wn, DZ)

    out = {}
    resid_store = {}                                   # (depth, sigma, factor) → residual (for cyl_summary)
    for dep in CYL_DEPTHS:
        pa = ana.phi(40.0 - dep)
        a = pa - edge_mean(pa)
        pka = float(a[Wn // 2])
        a_n = a / pka
        dd_a = dd(a_n, DZ)
        nrm = float(np.abs(dd_a[core]).max())
        t_a, s_a = H.sfap(pa, DZ, 60, 60, -20.0, cfg_fit)
        tm = (t_a >= T_CYL[0] - 1e-9) & (t_a <= T_CYL[1] + 1e-9)
        s_a_n = s_a / pka
        P = float(np.ptp(s_a_n))
        out[f"{dep:g}"] = dict(ana=dict(z=rl(z[keep]), phi_ana=rl(a_n[keep]), dd_ana=rl(dd_a[keep] / nrm, 3),
                                        t=rl(t_a[tm]), sfap_ana=rl(s_a_n[tm] / P, 3), phi_centre_ana=pka))
        runs = [(m, 5.0) for m in CYL_MESHES] + [(m, 1.0) for (dp, m) in CYL_S1 if dp == dep]
        for factor, s in runs:
            d = levels[factor]
            pf = fem_phi(d, s, dep)
            pf_m = denoise_field_n(pf, DZ, n=3)
            c0 = edge_mean(pf)
            f_, fit_ = pf - c0, pf_m - c0
            pk = float(f_[Wn // 2])
            f_n, fit_n, res_n = f_ / pk, fit_ / pk, (pf - pf_m) / pk
            dd_r, dd_f = dd(f_n, DZ), dd(fit_n, DZ)
            _, s_f = H.sfap(pf, DZ, 60, 60, -20.0, cfg_fit)
            _, s_r = H.sfap(pf, DZ, 60, 60, -20.0, cfg_none)
            s_f_n, s_r_n = s_f / pk / P, s_r / pk / P
            scale = float(np.dot(a[core], f_[core]) / np.dot(a[core], a[core]))
            key = f"{factor:g}" + ("" if s == 5.0 else "_s1")
            out[f"{dep:g}"][key] = dict(
                phi_raw=rl(f_n[keep]), phi_fit=rl(fit_n[keep]), resid=rl(res_n[keep], 3),
                dd_raw=rl(dd_r[keep] / nrm, 3), dd_fit=rl(dd_f[keep] / nrm, 3),
                sfap_raw=rl(s_r_n[tm], 3), sfap_fit=rl(s_f_n[tm], 3),
                phi_centre_fem=pk, scale_ls_fem_over_ana=scale,
                p2p_fit_over_ana=float(np.ptp(s_f_n)), p2p_raw_over_ana=float(np.ptp(s_r_n)),
                p2p_raw_over_fit=float(np.ptp(s_r) / np.ptp(s_f)),
                r_phi_ana=corr(a_n[core], f_n[core]), r_dd_fit_ana=corr(dd_a[core], dd_f[core]),
                r_dd_raw_ana=corr(dd_a[core], dd_r[core]), r_sfap_fit_ana=corr(s_a_n, s_f_n), r_sfap_raw_ana=corr(s_a_n, s_r_n),
                resid_rms_rel_peak=float(np.sqrt(np.mean(res_n ** 2))), sigma_s=s, factor=factor, cells=int(d["cells"]))
            print(f"  [cyl] depth {dep:g} mesh {key}: p2p fit/ana {np.ptp(s_f_n):.3f}, raw/ana {np.ptp(s_r_n):.3f}, "
                  f"r(φ″ fit) {corr(dd_a[core], dd_f[core]):.3f}, r(sfap fit) {corr(s_a_n, s_f_n):.3f}", flush=True)
        # residuals on every global level (for the cross-mesh residual correlation)
        for s in (5.0, 1.0):
            for factor, d in levels.items():
                pf = fem_phi(d, s, dep)
                pf_m = denoise_field_n(pf, DZ, n=3)
                resid_store[(dep, s, factor)] = (pf - pf_m) / float((pf - edge_mean(pf))[Wn // 2])
    ana.save()
    # cyl_summary: residual correlation between successive meshes and vs the finest
    facs = sorted(levels, reverse=True)
    summ = dict(levels=facs, depths=CYL_DEPTHS, pairs=[f"{facs[k]:g}->{facs[k + 1]:g}" for k in range(len(facs) - 1)], per_sigma={})
    for s in (5.0, 1.0):
        rec = dict(r_resid_succ={}, r_resid_vs_finest={}, resid_rms_rel_peak={})
        for dep in CYL_DEPTHS:
            rr = [resid_store[(dep, s, f)] for f in facs]
            rec["r_resid_succ"][f"{dep:g}"] = [corr(rr[k][core], rr[k + 1][core]) for k in range(len(facs) - 1)]
            rec["r_resid_vs_finest"][f"{dep:g}"] = [corr(rr[k][core], rr[-1][core]) for k in range(len(facs))]
            rec["resid_rms_rel_peak"][f"{dep:g}"] = [float(np.sqrt(np.mean(r ** 2))) for r in rr]
        summ["per_sigma"][f"s{s:g}"] = rec
    return out, summ


# --------------------------------------------------------------------------- part B: forearm
_CTX: dict = {}


def fa_sfap(phi, dz, L, cfg):
    """Single-fibre SFAP with the pipeline's geometry (NMJ at IZ_FRAC_FCU of the arc, no jitter)."""
    Lp = IZ_FRAC_FCU * L
    Ld = L - Lp
    t, s, _ = compute_sfap_spatial(np.asarray(phi, float), float(dz), Lp, Ld, (Lp - Ld) / 2.0, cfg)
    return np.asarray(t, float), np.asarray(s, float)


def _variant_cfgs():
    base = production_config()
    return {"none": replace(base, denoise="none"), "mono3": base,
            "mono5": replace(base, denoise_n_poles=5), "mono7": replace(base, denoise_n_poles=7),
            "bw03": replace(base, denoise="butterworth", butterworth_cutoff=0.03, butterworth_order=2)}


def _bed_task(i):
    """All variants' SFAP p2p and waveform for bed fibre i on every level (centre electrode)."""
    phis, dz, L, cfgs = _CTX["phis"], _CTX["dz"], _CTX["L"], _CTX["cfgs"]
    out = {}
    for lv, phi in phis.items():
        for name, cfg in cfgs.items():
            _, s = fa_sfap(phi[i], dz[i], L[i], cfg)
            out[(lv, name)] = s
    return i, out


def part_forearm():
    common = np.load(OUT / "forearm_common.npz")
    bed = common["bed_paths"].astype(float)                       # (637, 200, 3)
    arc_dz, L_fib = common["arc_dz"], common["L_fib"]
    elec = common["elec_ref"].reshape(-1, 3)[E0]
    cxy = common["limb_centre_xy"]
    n_fib, n_z = bed.shape[:2]
    # depth = closest approach to the centre electrode; lateral = offset of that point from the electrode's inward ray
    dist = np.linalg.norm(bed - elec, axis=2)
    k_min = dist.argmin(axis=1)
    depth = dist[np.arange(n_fib), k_min]
    inward = np.array([cxy[0] - elec[0], cxy[1] - elec[1], 0.0]); inward /= np.linalg.norm(inward)
    v = bed[np.arange(n_fib), k_min] - elec
    along = v @ inward
    lateral = np.linalg.norm(v - along[:, None] * inward, axis=1)
    # the FCU is only ~11 mm thick along the inward ray under the centre electrode, so the two deepest
    # targets are met by fibres whose closest approach is that deep but sits laterally off the ray
    # (6 / 12.6 mm): within ±0.6 mm of the target depth take the fibre with the smallest lateral offset
    fibres = []
    for tgt in FA_TARGET_DEPTHS:
        cand = np.where(np.abs(depth - tgt) < 0.6)[0]
        fibres.append(int(cand[np.argmin(lateral[cand])]))
    print(f"  [fa] fibres {fibres}: depth {np.round(depth[fibres], 2)}, lateral {np.round(lateral[fibres], 2)}, "
          f"k_min {k_min[fibres]}", flush=True)

    levels = {}
    for lv in FA_LEVELS:
        d = np.load(OUT / f"forearm_e{lv:g}.npz")
        levels[lv] = dict(phi=d["phi"][E0], phi_cond=d["phi_cond"][E0], cells=int(d["cells"]))
    cfgs = _variant_cfgs()
    pairs = [(FA_LEVELS[k], FA_LEVELS[k + 1]) for k in range(len(FA_LEVELS) - 1)]
    pair_keys = [f"{a:g}->{b:g}" for a, b in pairs]
    fin = FA_LEVELS[-1]

    # ---- the six fibres: waveforms + (i)–(iv) ------------------------------------------------
    mri, summ_i, summ_ii, summ_iii, summ_iv = {}, {}, {}, {}, {}
    for fi, i in enumerate(fibres):
        dz, L = float(arc_dz[i]), float(L_fib[i])
        z = (np.arange(n_z) - n_z // 2) * dz
        keep, core = np.abs(z) <= HALF_Z, np.abs(z) <= CORE
        lam, _ = spectrum(np.zeros(n_z), dz)
        Lp = IZ_FRAC_FCU * L
        meta = dict(fibre_index=int(i), depth_mm=float(depth[i]), lateral_mm=float(lateral[i]), k_min=int(k_min[i]),
                    dz_mm=dz, len1_mm=Lp, len2_mm=L - Lp, posz_mm=(2 * Lp - L) / 2.0, z=rl(z[keep]), lambda_mm=rl(lam))
        per = {}
        raw, fit, res, ddf, ddr, sf = {}, {}, {}, {}, {}, {}
        for lv in FA_LEVELS:
            phi, phic = levels[lv]["phi"][i], levels[lv]["phi_cond"][i]
            assert np.array_equal(phic, denoise_field_n(phi, dz, n=3)) or np.allclose(phic, denoise_field_n(phi, dz, n=3), rtol=0, atol=1e-15)
            raw[lv], fit[lv], res[lv] = phi, phic, phi - phic
            ddr[lv], ddf[lv] = dd(phi, dz), dd(phic, dz)
            sf[lv] = {}
            for name, cfg in cfgs.items():
                t, s = fa_sfap(phi, dz, L, cfg)
                sf[lv][name] = s
            tm = (t >= T_FA[0] - 1e-9) & (t <= T_FA[1] + 1e-9)
            meta.setdefault("t", rl(t[tm]))
            lam_i, amp = spectrum(res[lv], dz)
            p2p = float(np.ptp(phi))
            sub = lam_i <= 40
            per[f"{lv:g}"] = dict(
                phi_raw=rl(phi[keep] * 1e3), phi_fit=rl(phic[keep] * 1e3), resid=rl(res[lv][keep] * 1e6, 3),
                dd_raw=rl(ddr[lv][keep] * 1e6, 3), dd_fit=rl(ddf[lv][keep] * 1e6, 3),
                sfap_raw=rl(sf[lv]["none"][tm] * 1e6, 3), sfap_fit=rl(sf[lv]["mono3"][tm] * 1e6, 3),
                resid_rms_rel_p2p=float(np.sqrt(np.mean(res[lv] ** 2)) / p2p), phi_p2p_mV=p2p * 1e3,
                cells=levels[lv]["cells"], p2p_uV={name: float(np.ptp(sf[lv][name]) * 1e6) for name in VARIANTS})
            if lv in (0.03, fin):
                per[f"{lv:g}"].update(sfap_mono5=rl(sf[lv]["mono5"][tm] * 1e6, 3), sfap_mono7=rl(sf[lv]["mono7"][tm] * 1e6, 3),
                                      sfap_bw03=rl(sf[lv]["bw03"][tm] * 1e6, 3), resid_spectrum=rl(amp / p2p, 3))
            summ_iv.setdefault(f"{fi}", {})[f"{lv:g}"] = dict(peak_lambda_sub40_mm=float(lam_i[sub][np.argmax(amp[sub])]),
                                                           band_frac=band_fracs(lam_i, amp))
        mri[f"{fi}"] = dict(meta=meta, **per)
        summ_i[f"{fi}"] = dict(rms_rel_p2p=[per[f"{lv:g}"]["resid_rms_rel_p2p"] for lv in FA_LEVELS],
                               r_succ=[corr(res[a][core], res[b][core]) for a, b in pairs],
                               r_vs_finest=[corr(res[lv][core], res[fin][core]) for lv in FA_LEVELS],
                               r_succ_full=[corr(res[a], res[b]) for a, b in pairs])
        summ_ii[f"{fi}"] = dict(r_succ=[corr(ddf[a][core], ddf[b][core]) for a, b in pairs],
                                r_vs_finest=[corr(ddf[lv][core], ddf[fin][core]) for lv in FA_LEVELS],
                                r_succ_raw=[corr(ddr[a][core], ddr[b][core]) for a, b in pairs],
                                r_phi_succ=[corr(raw[a][core], raw[b][core]) for a, b in pairs])
        summ_iii[f"{fi}"] = {}
        for name in VARIANTS:
            summ_iii[f"{fi}"][name] = dict(
                p2p_ratio_succ=[float(np.ptp(sf[a][name]) / np.ptp(sf[b][name])) for a, b in pairs],
                r_succ=[corr(sf[a][name], sf[b][name]) for a, b in pairs],
                p2p_vs_finest=[float(np.ptp(sf[lv][name]) / np.ptp(sf[fin][name])) for lv in FA_LEVELS],
                r_vs_finest=[corr(sf[lv][name], sf[fin][name]) for lv in FA_LEVELS],
                shape_cost_r_vs_mono3_finest=corr(sf[fin][name], sf[fin]["mono3"]),
                shape_cost_p2p_vs_mono3_finest=float(np.ptp(sf[fin][name]) / np.ptp(sf[fin]["mono3"])))
        print(f"  [fa] fibre {fi} (#{i}, {depth[i]:.1f} mm): resid rms/p2p {np.round(summ_i[f'{fi}']['rms_rel_p2p'], 4)}, "
              f"r(resid succ) {np.round(summ_i[f'{fi}']['r_succ'], 2)}, r(φ″fit succ) {np.round(summ_ii[f'{fi}']['r_succ'], 3)}", flush=True)
        for name in VARIANTS:
            v_ = summ_iii[f"{fi}"][name]
            print(f"        {name:6s} p2p succ {np.round(v_['p2p_ratio_succ'], 3)} r succ {np.round(v_['r_succ'], 3)} "
                  f"shape r {v_['shape_cost_r_vs_mono3_finest']:.4f} p2p/mono3 {v_['shape_cost_p2p_vs_mono3_finest']:.3f}", flush=True)

    # ---- (iv) bed-mean residual spectrum per level, (i) whole-bed scatter ----------------------
    lam_common = np.linspace(2.1, 100.0, 120)
    bed_mean = {}
    for lv in FA_LEVELS:
        acc = np.zeros(len(lam_common))
        for i in range(n_fib):
            lam_i, amp = spectrum(levels[lv]["phi"][i] - levels[lv]["phi_cond"][i], float(arc_dz[i]))
            acc += np.interp(lam_common, lam_i[::-1], amp[::-1]) / np.ptp(levels[lv]["phi"][i])
        bed_mean[f"{lv:g}"] = rl(acc / n_fib, 3)
    summ_iv["bed_mean"] = dict(lambda_mm=rl(lam_common), amp=bed_mean)
    cur, fine = levels[0.03], levels[fin]
    res_cur, res_fin = cur["phi"] - cur["phi_cond"], fine["phi"] - fine["phi_cond"]
    rms_cur = np.sqrt(np.mean(res_cur ** 2, axis=1)) / np.ptp(cur["phi"], axis=1)
    rms_fin = np.sqrt(np.mean(res_fin ** 2, axis=1)) / np.ptp(fine["phi"], axis=1)
    r_cf = np.array([corr(res_cur[i], res_fin[i]) for i in range(n_fib)])
    r_cf_core = np.array([corr(res_cur[i][np.abs((np.arange(n_z) - n_z // 2) * arc_dz[i]) <= CORE],
                               res_fin[i][np.abs((np.arange(n_z) - n_z // 2) * arc_dz[i]) <= CORE]) for i in range(n_fib)])
    edges = np.arange(3.0, 18.01, 2.0)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (depth >= lo) & (depth < hi)
        if m.sum():
            bins.append(dict(lo=lo, hi=hi, n=int(m.sum()), rms_0_03=float(np.median(rms_cur[m])), rms_0_012=float(np.median(rms_fin[m])),
                             r_resid=float(np.median(r_cf[m])), r_resid_core=float(np.median(r_cf_core[m]))))
    scatter = dict(depth_mm=rl(depth, 3), lateral_mm=rl(lateral, 3), rms_rel_p2p_e0_03=rl(rms_cur, 3), rms_rel_p2p_e0_012=rl(rms_fin, 3),
                   r_resid_0_03_vs_0_012=rl(r_cf, 3), depth_bins=bins,
                   corr_rms_depth=dict(e0_03=corr(np.log(depth), np.log(rms_cur)), e0_012=corr(np.log(depth), np.log(rms_fin))))
    print(f"  [fa] bed: r(resid 0.03 vs 0.012) median {np.median(r_cf):.3f} (core {np.median(r_cf_core):.3f}); "
          f"rms/p2p median {np.median(rms_cur):.4f} → {np.median(rms_fin):.4f}", flush=True)

    # ---- (iii) over the whole bed: every variant, every level, centre electrode -----------------
    t0 = time.time()
    _CTX.update(phis={lv: levels[lv]["phi"] for lv in FA_LEVELS}, dz=arc_dz, L=L_fib, cfgs=cfgs)
    P2P = {(lv, nm): np.zeros(n_fib) for lv in FA_LEVELS for nm in VARIANTS}
    Rsucc = {(pk, nm): np.zeros(n_fib) for pk in pair_keys for nm in VARIANTS}
    Rfin = {(lv, nm): np.zeros(n_fib) for lv in FA_LEVELS for nm in VARIANTS}
    Rshape = {nm: np.zeros(n_fib) for nm in VARIANTS}
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for k, (i, sf_i) in enumerate(pool.imap_unordered(_bed_task, range(n_fib), chunksize=8)):
            for lv in FA_LEVELS:
                for nm in VARIANTS:
                    P2P[(lv, nm)][i] = np.ptp(sf_i[(lv, nm)])
                    Rfin[(lv, nm)][i] = corr(sf_i[(lv, nm)], sf_i[(fin, nm)])
            for (a, b), pk in zip(pairs, pair_keys):
                for nm in VARIANTS:
                    Rsucc[(pk, nm)][i] = corr(sf_i[(a, nm)], sf_i[(b, nm)])
            for nm in VARIANTS:
                Rshape[nm][i] = corr(sf_i[(fin, nm)], sf_i[(fin, "mono3")])
            if (k + 1) % 100 == 0:
                print(f"  [fa] bed SFAPs {k + 1}/{n_fib} ({time.time() - t0:.0f} s)", flush=True)
    bed_iii = dict(pairs=pair_keys, per_variant={})
    for nm in VARIANTS:
        rec = dict(succ={}, vs_finest={}, shape_cost={})
        for (a, b), pk in zip(pairs, pair_keys):
            ch = np.abs(P2P[(a, nm)] / P2P[(b, nm)] - 1) * 100
            rec["succ"][pk] = dict(p2p_change_pct_median=float(np.median(ch)), p2p_change_pct_p90=float(np.percentile(ch, 90)),
                                   p2p_change_pct_max=float(ch.max()), r_median=float(np.median(Rsucc[(pk, nm)])),
                                   r_p10=float(np.percentile(Rsucc[(pk, nm)], 10)), r_min=float(Rsucc[(pk, nm)].min()))
        for lv in FA_LEVELS[:-1]:
            ch = np.abs(P2P[(lv, nm)] / P2P[(fin, nm)] - 1) * 100
            rec["vs_finest"][f"{lv:g}"] = dict(p2p_change_pct_median=float(np.median(ch)), p2p_change_pct_p90=float(np.percentile(ch, 90)),
                                              r_median=float(np.median(Rfin[(lv, nm)])), r_p10=float(np.percentile(Rfin[(lv, nm)], 10)))
        rat = P2P[(fin, nm)] / P2P[(fin, "mono3")]
        rec["shape_cost"] = dict(r_vs_mono3_median=float(np.median(Rshape[nm])), r_vs_mono3_p10=float(np.percentile(Rshape[nm], 10)),
                                 r_vs_mono3_min=float(Rshape[nm].min()), p2p_over_mono3_median=float(np.median(rat)),
                                 p2p_over_mono3_p10=float(np.percentile(rat, 10)), p2p_over_mono3_p90=float(np.percentile(rat, 90)))
        bed_iii["per_variant"][nm] = rec
        print(f"  [fa] bed {nm:6s}: |Δp2p| median per pair {[round(rec['succ'][pk]['p2p_change_pct_median'], 1) for pk in pair_keys]} %, "
              f"r median {[round(rec['succ'][pk]['r_median'], 4) for pk in pair_keys]}; shape r vs mono3 {rec['shape_cost']['r_vs_mono3_median']:.4f}", flush=True)
    # depth dependence of the mesh stability (current → finest) per variant, in the same 2 mm bins
    bed_iii["vs_finest_by_depth"] = {}
    for nm in VARIANTS:
        ch = np.abs(P2P[(0.03, nm)] / P2P[(fin, nm)] - 1) * 100
        bed_iii["vs_finest_by_depth"][nm] = [dict(lo=lo, hi=hi, p2p_change_pct_median=float(np.median(ch[(depth >= lo) & (depth < hi)])),
                                                  r_median=float(np.median(Rfin[(0.03, nm)][(depth >= lo) & (depth < hi)])))
                                             for lo, hi in zip(edges[:-1], edges[1:]) if ((depth >= lo) & (depth < hi)).sum()]
    summ_iii["bed"] = bed_iii
    summary = dict(levels=FA_LEVELS, pairs=pair_keys, variants=VARIANTS, centre_electrode=E0,
                   fibres=[dict(key=f"{fi}", index=int(i), depth_mm=float(depth[i]), lateral_mm=float(lateral[i])) for fi, i in enumerate(fibres)],
                   cells={f"{lv:g}": levels[lv]["cells"] for lv in FA_LEVELS},
                   i_resid=summ_i, ii_ddfit=summ_ii, iii_sfap=summ_iii, iv_spectrum=summ_iv, bed_scatter=scatter)
    return mri, summary


# --------------------------------------------------------------------------- preview figure
def preview(store):
    import matplotlib; matplotlib.use("Agg")           # noqa: E702
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 0.9})
    cyl, mri, ms = store["cyl"], store["mri"], store["mri_summary"]
    mesh_col = {"0.42": "#8fb3e0", "0.3": "#2f6fba", "0.2": "#0b2d5b", "0.3_s1": "#e08a3c", "0.2_s1": "#8a4a12"}
    lv_col = dict(zip([f"{lv:g}" for lv in FA_LEVELS], ["#c7d3e3", "#8fb3e0", "#4f8ccf", "#1f5fa8", "#0b2d5b"]))
    var_col = dict(zip(VARIANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))   # fixed categorical order
    n_rows = len(CYL_DEPTHS) + len(mri) + 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 2.3 * n_rows))
    for r, dep in enumerate(CYL_DEPTHS):
        row = cyl[f"{dep:g}"]
        ax = axes[r]
        an = row["ana"]; zz, tt = an["z"], an["t"]
        ax[0].plot(zz, an["phi_ana"], "k", lw=1.4, label="analytical")
        ax[2].plot(zz, an["dd_ana"], "k", lw=1.4, label="analytical")
        ax[3].plot(tt, an["sfap_ana"], "k", lw=1.4, label="analytical")
        for key, e in row.items():
            if key == "ana":
                continue
            c = mesh_col[key]
            ax[0].plot(zz, e["phi_raw"], color=c, label=f"FEM {key} raw")
            ax[1].plot(zz, np.array(e["resid"]) * 1e3, color=c, label=f"{key}")
            ax[2].plot(zz, e["dd_fit"], color=c, label=f"{key} fit (r={e['r_dd_fit_ana']:.2f})")
            if key == "0.3":
                ax[2].plot(zz, e["dd_raw"], color="0.6", lw=0.5, label="0.3 raw", zorder=0)
                ax[3].plot(tt, e["sfap_raw"], color="0.6", lw=0.6, label=f"0.3 raw (p2p {e['p2p_raw_over_ana']:.2f})", zorder=0)
            ax[3].plot(tt, e["sfap_fit"], color=c, label=f"{key} fit (p2p {e['p2p_fit_over_ana']:.2f}, r={e['r_sfap_fit_ana']:.3f})")
        ax[0].set_title(f"cylinder {dep:g} mm: φ / centre value"); ax[1].set_title("residual φ − fit (×1e-3 of centre)")
        ax[2].set_title("φ″ / max|φ″_ana|"); ax[3].set_title("SFAP / p2p(analytical)")
        for a_ in ax[:3]:
            a_.set_xlabel("z (mm)")
        ax[3].set_xlabel("t (ms)")
        for a_ in ax:
            a_.legend(loc="best")
    for j, (fk, fr) in enumerate(mri.items()):
        r = len(CYL_DEPTHS) + j
        ax = axes[r]
        meta = fr["meta"]
        for lv in FA_LEVELS:
            e = fr[f"{lv:g}"]; c = lv_col[f"{lv:g}"]
            ax[0].plot(meta["z"], e["phi_raw"], color=c, label=f"{lv:g}")
            ax[1].plot(meta["z"], e["resid"], color=c, label=f"{lv:g} (rms/p2p {e['resid_rms_rel_p2p']:.4f})")
            ax[2].plot(meta["z"], e["dd_fit"], color=c, label=f"{lv:g} fit")
            if lv == 0.03:
                ax[2].plot(meta["z"], e["dd_raw"], color="0.6", lw=0.5, label="0.03 raw", zorder=0)
                ax[3].plot(meta["t"], e["sfap_raw"], color="0.6", lw=0.6, label="0.03 none", zorder=0)
            ax[3].plot(meta["t"], e["sfap_fit"], color=c, label=f"{lv:g} mono3 ({e['p2p_uV']['mono3']:.1f} µV)")
        ax[0].set_title(f"forearm fibre #{meta['fibre_index']} @ {meta['depth_mm']:.1f} mm (lat {meta['lateral_mm']:.1f}): φ (mV)")
        ax[1].set_title("residual φ − fit (µV)"); ax[2].set_title("φ″ (µV/mm²)"); ax[3].set_title("SFAP (µV), single fibre")
        for a_ in ax[:3]:
            a_.set_xlabel("z (mm)")
        ax[3].set_xlabel("t (ms)")
        for a_ in ax:
            a_.legend(loc="best")
    # summary rows
    r = len(CYL_DEPTHS) + len(mri)
    ax = axes[r]
    sc = ms["bed_scatter"]
    ax[0].scatter(sc["depth_mm"], sc["rms_rel_p2p_e0_03"], s=4, color="#4f8ccf", label="0.03")
    ax[0].scatter(sc["depth_mm"], sc["rms_rel_p2p_e0_012"], s=4, color="#0b2d5b", label="0.012")
    ax[0].set(xlabel="fibre depth below centre electrode (mm)", ylabel="fit residual rms / p2p(φ)", yscale="log", title="whole bed (637 fibres)")
    ax[0].legend()
    ax[1].scatter(sc["depth_mm"], sc["r_resid_0_03_vs_0_012"], s=4, color="#0b2d5b")
    ax[1].set(xlabel="fibre depth (mm)", ylabel="r(residual 0.03 vs 0.012)", ylim=(-1, 1), title="is the residual the same on both meshes?")
    bm = ms["iv_spectrum"]["bed_mean"]
    for lv in FA_LEVELS:
        ax[2].plot(bm["lambda_mm"], bm["amp"][f"{lv:g}"], color=lv_col[f"{lv:g}"], label=f"{lv:g}")
    ax[2].set(xscale="log", yscale="log", xlabel="wavelength (mm)", ylabel="|spectrum| / p2p φ", title="bed-mean residual spectrum"); ax[2].legend()
    bed = ms["iii_sfap"]["bed"]
    x = np.arange(len(bed["pairs"]))
    for k, nm in enumerate(VARIANTS):
        ax[3].plot(x, [bed["per_variant"][nm]["succ"][pk]["p2p_change_pct_median"] for pk in bed["pairs"]], "o-", color=var_col[nm], label=nm)
    ax[3].set(xticks=x, xticklabels=bed["pairs"], ylabel="median |Δp2p| (%) over the bed", title="SFAP mesh stability per conditioning", yscale="log"); ax[3].legend()
    ax = axes[r + 1]
    for k, nm in enumerate(VARIANTS):
        ax[0].plot(x, [bed["per_variant"][nm]["succ"][pk]["r_median"] for pk in bed["pairs"]], "o-", color=var_col[nm], label=nm)
    ax[0].set(xticks=x, xticklabels=bed["pairs"], ylabel="median r (successive)", title="SFAP waveform stability"); ax[0].legend()
    deps = [f["depth_mm"] for f in ms["fibres"]]
    for nm in VARIANTS:
        ax[1].plot(deps, [ms["iii_sfap"][fk][nm]["shape_cost_r_vs_mono3_finest"] for fk in mri], "o-", color=var_col[nm], label=nm)
    ax[1].set(xlabel="fibre depth (mm)", ylabel="r vs mono3 on 0.012", title="shape cost (6 fibres)"); ax[1].legend()
    for fk in mri:
        ax[2].plot(x, ms["i_resid"][fk]["r_succ"], "o-", label=f"{ms['fibres'][int(fk)]['depth_mm']:.1f} mm")
    ax[2].set(xticks=x, xticklabels=bed["pairs"], ylabel="r(residual) successive", ylim=(-1, 1), title="residual structure across meshes"); ax[2].legend()
    for fk in mri:
        ax[3].plot(x, ms["ii_ddfit"][fk]["r_succ"], "o-", label=f"{ms['fibres'][int(fk)]['depth_mm']:.1f} mm")
    ax[3].set(xticks=x, xticklabels=bed["pairs"], ylabel="r(φ″ fitted) successive", title="conditioned φ″ convergence"); ax[3].legend()
    fig.tight_layout()
    fig.savefig(OUT / "waveforms_preview.png", dpi=110)
    plt.close(fig)
    print("wrote", OUT / "waveforms_preview.png", flush=True)


def main():
    t0 = time.time()
    cyl, cyl_summary = part_cyl()
    print(f"[cyl] done in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    mri, mri_summary = part_forearm()
    print(f"[fa] done in {time.time() - t0:.0f} s", flush=True)
    store = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), structure=STRUCTURE.strip().splitlines(),
                 cyl=cyl, cyl_summary=cyl_summary, mri=mri, mri_summary=mri_summary)
    path = OUT / "waveforms.json"
    path.write_text(json.dumps(MC.jsonable(store), separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)", flush=True)
    preview(store)


if __name__ == "__main__":
    main()
