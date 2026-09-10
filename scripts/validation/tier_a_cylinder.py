"""Tier A — the cylinder must reproduce: first principles → analytical → FEM → pipeline.

  A0  engine vs FIRST PRINCIPLES  (closed-form φ, infinite anisotropic medium; no
      volume-conductor model in the loop — a pure test of the SFAP integral)
  A1  FEM lead field φ(z) vs the Farina-2004 analytical cylinder φ(z)
  A2  full pipeline (FEM φ → golden recipe) vs the analytical-φ pipeline and vs the
      Farina generator: waveform, EOF timing, propagation / CV
  A3  volume-conductor symmetries the cylinder must obey: rotation, z-translation,
      reciprocity (interior points)

Run:  python scripts/validation/tier_a_cylinder.py      (needs cyl_fem cache; ~1 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import *                                    # noqa: F401,F403
import cyl_fem

T = "A"
DZ = V * 1000.0 / FS                                     # 0.977 mm (analytical grid)
cache = cyl_fem.build_cache()
Z_ABS, RADII, THETAS, ZE = cache["z_abs"], cache["radii"], cache["thetas"], float(cache["ze"])
FIG = {}

print("=" * 74); print("Tier A — cylinder: analytical vs pipeline"); print("=" * 74, "\n")

# =========================================================================== A0
# --- A0.1 spatial engine reproduces the line-source integral exactly ---------
z = (np.arange(W) - W // 2) * DZ
phi0 = phi_inf(z)
geoms = [(0.0, 60.0, 60.0), (-20.0, 40.0, 80.0), (-30.0, 30.0, 90.0)]
rs, lags, amps, wf = [], [], [], []
for posz, l1, l2 in geoms:
    t_r, s_r_ = sfap_first_principles(posz, l1, l2)
    t_s, s_s = sfap(phi0, DZ, l1, l2, posz, golden_cfg(fiber_window="boxcar", denoise="none"))
    r, lag, amp = best_r(t_r, s_r_, t_s, s_s)
    rs.append(r); lags.append(lag); amps.append(amp); wf.append((t_r, s_r_, t_s, s_s))
FIG["a0"] = wf
check(T, "A0.1 spatial engine ≡ first-principles line-source integral (shape, timing)",
      min(rs) >= 0.999 and max(abs(l) for l in lags) <= 0.1,
      f"r = {np.round(rs, 4).tolist()}, lag(ms) = {np.round(lags, 2).tolist()} over NMJ offsets 0/−20/−30 mm",
      "r ≥ 0.999 and |lag| ≤ 0.1 ms: same integral, so only discretisation can differ",
      principle="SFAP = ∫ i_m(z,t)·φ(z) dz with i_m = σ_in·π·a²·∂²Vm/∂z² (core-conductor line source)",
      refs="Rosenfalck 1969; Andreassen & Rosenfalck 1981; Dimitrov & Dimitrova 1998")

# --- A0.2 amplitude constant ---------------------------------------------------
amp_v = {}
for v_ in (2.0, 4.0):
    t_r, s_r_ = sfap_first_principles(0.0, 60, 60, v=v_)
    t_s, s_s = sfap(phi0, DZ, 60, 60, 0.0, golden_cfg(fiber_window="boxcar", denoise="none", v=v_))
    amp_v[v_] = best_r(t_r, s_r_, t_s, s_s)[2]
check(T, "A0.2 spatial engine amplitude constant equals first principles",
      all(abs(a - 1.0) < 0.05 for a in amp_v.values()),
      f"engine/reference amplitude ratio = {amp_v[2.0]:.3f} at v=2, {amp_v[4.0]:.3f} at v=4  (= 1/v)",
      "ratio 1.00 ± 5 %, independent of v",
      principle="for a fixed spatial IAP the extracellular potential does not depend on CV; "
                "`sfap = (CSD @ φ)·dz/v` divides by v once too often (CSD is already ∂²Vm/∂z²)",
      refs="Plonsey & Barr, Bioelectricity ch. 8; engines/spatial.py:239", known=True)

# --- A0.3 Fourier engine vs first principles (+ mirrored-IAP diagnosis) --------
rf, lf, rf_m = [], [], []
for posz, l1, l2 in geoms:
    t_r, s_r_ = sfap_first_principles(posz, l1, l2)
    t_m, s_m = sfap_first_principles(posz, l1, l2, mirror=True)
    t_f, s_f = fourier_physical(phi0, DZ, l1, l2, posz)
    r, lag, _ = best_r(t_r, s_r_, t_f, s_f, max_lag_ms=15)
    rm, _, _ = best_r(t_m, s_m, t_f, s_f, max_lag_ms=15)
    rf.append(r); lf.append(lag); rf_m.append(rm)
FIG["a0f"] = (t_r, s_r_, t_m, s_m, t_f, s_f)
check(T, "A0.3 Fourier engine vs first principles (signed r, lag)",
      min(rf) >= 0.99,
      f"signed r = {np.round(rf, 3).tolist()} at lag {np.round(lf, 2).tolist()} ms;  "
      f"vs a spatially MIRRORED IAP: r = {np.round(rf_m, 3).tolist()}",
      "r ≥ +0.99 (same integral, same sign, no lag)",
      principle="the Fourier (Farina 2001) engine is the repo's shape oracle (r=0.997 vs MATLAB); "
                "if it matches the mirrored IAP better, its IAP orientation/sign conventions are "
                "the source of the pinned engine disagreements (anti-phase, ~2 ms lag)",
      refs="GOLDEN_METHOD.md §6, §8; Farina & Merletti 2001", known=True)

# --- A0.4 engine invariances -----------------------------------------------------
t_a, s_a = sfap(phi0, DZ, 60, 60, -20.0, golden_cfg(fsamp=4096.0))
t_b, s_b = sfap(phi0, DZ, 60, 60, -20.0, golden_cfg(fsamp=2048.0))
r_fs = float(np.corrcoef(np.interp(t_b, t_a, s_a), s_b)[0, 1]); a_fs = p2p(s_b) / p2p(s_a)
zf = np.arange(-125, 125.01, 0.5); phi_f = phi_inf(zf)
t_c, s_c = sfap(phi_f, 0.5, 60, 60, -20.0)
r_dz = float(np.corrcoef(s_a, s_c)[0, 1]); a_dz = p2p(s_c) / p2p(s_a)
check(T, "A0.4a sampling invariance: fsamp 4096→2048 and dz 0.98→0.5 mm leave the SFAP unchanged",
      r_fs > 0.995 and abs(a_fs - 1) < 0.02 and r_dz > 0.995 and abs(a_dz - 1) < 0.02,
      f"fsamp: r={r_fs:.5f} p2p ratio={a_fs:.4f};  dz: r={r_dz:.5f} p2p ratio={a_dz:.4f}",
      "r > 0.995, amplitude within 2 % (discretisation-converged)", principle="a converged discretisation")

sh = 20.0                                                 # translate the electrode +20 mm
# long fibre (tendons ≥ 80 mm away) so the non-moving EOFs do not bias the lobe timing
t_1, s_1 = sfap(phi0, DZ, 100, 100, -20.0); t_2, s_2 = sfap(shift_phi(phi0, DZ, sh), DZ, 100, 100, -20.0)
lag_tr = t_2[np.argmin(s_2)] - t_1[np.argmin(s_1)]
check(T, "A0.4b translation: moving the electrode Δz along the fibre delays the propagating lobe by Δz/v",
      abs(lag_tr - sh / V) <= 1000.0 / FS + 1e-9,
      f"Δz={sh:g} mm → main-lobe delay {lag_tr:.2f} ms (expect {sh / V:.2f})", "|Δ| ≤ one sample (0.24 ms)",
      principle="propagation at v in a z-invariant conductor; the EOFs stay put (they are checked in A2.3)")

# --- A0.5 the source carries no net current at any instant (monopole-free) ---------
from emgforge.synthesis.engines.spatial import build_csd_matrix
net = {}
for win in ("boxcar", "one_sided"):
    cfg = golden_cfg(fiber_window=win)
    tt = np.arange(W) / FS * 1000.0 - 10.0
    csd = build_csd_matrix(z, tt, -20.0, 40.0, 80.0, DZ, cfg)
    net[win] = float(np.max(np.abs(csd.sum(1))) / np.max(np.abs(csd).sum(1)))
check(T, "A0.5 the CSD source is monopole-free: ∫ i_m(z,t) dz = 0 at every instant",
      max(net.values()) < 1e-6,
      f"max_t |Σ_z i_m| / Σ_z |i_m| = {net['boxcar']:.1e} (boxcar), {net['one_sided']:.1e} (one_sided)",
      "< 1e-6: generation and end-of-fibre terms exactly balance the propagating tripoles",
      principle="a fibre injects no net current; the GEN/EOF terms are the unique monopole-free completion",
      refs="Petersen 2016; Merletti et al. 1999 (tripole sums to zero); Kleinpenning et al. 1990")

t_1, s_1 = sfap(phi0, DZ, 60, 60, -10.0); t_2, s_2 = sfap(phi0, DZ, 60, 60, +10.0)
from emgforge.synthesis.fibres import FibreBed
from emgforge.synthesis.api import field_to_muap
res = field_to_muap(np.vstack([phi0, phi0]), FibreBed.from_arrays(DZ, 60, 60, [-10.0, 10.0]), golden_cfg())
lin_err = float(np.abs(res.muap - (s_1 + s_2)).max() / p2p(s_1 + s_2))
check(T, "A0.4c superposition: a two-fibre MUAP is the sum of its SFAPs",
      lin_err < 1e-9, f"max|MUAP − (SFAP₁+SFAP₂)| / p2p = {lin_err:.1e}", "< 1e-9 (linear volume conductor)")

t_3, s_3 = sfap(phi0, DZ, 60, 60, -20.0, golden_cfg(v=3.0)); t_4, s_4 = sfap(phi0, DZ, 60, 60, -20.0)
d3, d4 = duration_ms(t_3, s_3), duration_ms(t_4, s_4); f3, f4 = mnf(s_3, FS), mnf(s_4, FS)
check(T, "A0.4d CV scaling: v 4→3 stretches the SFAP by 4/3 and scales its spectrum by 3/4",
      abs(d3 / d4 - 4 / 3) < 0.08 and abs(f3 / f4 - 0.75) < 0.08,
      f"duration ratio {d3 / d4:.3f} (expect 1.333), MNF ratio {f3 / f4:.3f} (expect 0.750)",
      "both within 8 %", principle="time–space scaling of a travelling wave: MNF ∝ CV",
      refs="Lindström & Magnusson 1977; Stulen & De Luca 1981")

# =========================================================================== A1
def fem_phi(r, th, centre=ZE, key="phi"):
    i, j = int(np.argmin(np.abs(RADII - r))), int(np.argmin(np.abs(THETAS - th)))
    return cyl_fem.window(cache[key][i, j], Z_ABS, centre, W, DZ)


def dc_free(p, n_edge=12):
    return p - np.mean(np.r_[p[:n_edge], p[-n_edge:]])


core = np.abs(z) <= 60.0                                 # the fibre extent that matters
depth_r, depth_fw, ratio_pk = [], [], []
for r in (33.0, 30.0, 27.0, 25.0, 20.0):
    pa, _ = ana_phi(r); pf = fem_phi(r, 0.0)
    a, f = dc_free(pa), dc_free(pf)
    depth_r.append(float(np.corrcoef(a[core], f[core])[0, 1]))
    depth_fw.append((fwhm(z, a), fwhm(z, f)))
    ratio_pk.append(f[W // 2] / a[W // 2])
ratio_pk = np.array(ratio_pk); fw_err = [abs(ff / fa - 1) for fa, ff in depth_fw]
FIG["a1"] = [(r, dc_free(ana_phi(r)[0]), dc_free(fem_phi(r, 0.0))) for r in (33.0, 27.0, 20.0)]
check(T, "A1.1 FEM φ(z) reproduces the analytical cylinder φ(z) along the fibre (5 depths)",
      min(depth_r) >= 0.99 and max(fw_err) <= 0.20,
      f"r = {np.round(depth_r, 4).tolist()}; FWHM_z ana/FEM (mm) = "
      f"{[f'{a:.1f}/{f:.1f}' for a, f in depth_fw]} at r = 33/30/27/25/20 mm",
      "r ≥ 0.99; FWHM within 20 % (electrode models differ: Ø10 mm disk vs σ=5 mm Gaussian blob, "
      "so the shallowest fibres see different electrode averaging)",
      principle="reciprocity: the electrode's field sampled along the fibre IS the lead field",
      refs="Farina et al. 2004 (multilayer cylinder); Lowery et al. 2002 (FEM); Maksymenko et al. 2023 (NMSE 3–5 % FEM vs Farina)")
# the SFAP integrates φ'' (dual form), so compare the denoised second derivative on the core
from emgforge.synthesis.preprocessing import denoise_field_n
def dd(p):
    return np.gradient(np.gradient(p, DZ), DZ)
dd_r = {}
for r in (33.0, 27.0, 20.0):
    f = dd(denoise_field_n(fem_phi(r, 0.0), DZ, n=3))
    dd_r[r] = [float(np.corrcoef(dd(ana_phi(r, dim1=d)[0])[core], f[core])[0, 1]) for d in (2.5, 5.0, 7.5, 10.0)]
best = {r: max(v) for r, v in dd_r.items()}
check(T, "A1.1b … and its second derivative φ''(z) — the kernel the SFAP actually integrates",
      min(best.values()) >= 0.98,
      "r(φ'') vs analytical electrode radius 2.5/5/7.5/10 mm: " +
      "; ".join(f"r={r:g}: {np.round(v, 3).tolist()}" for r, v in dd_r.items()),
      "r ≥ 0.98 at the best-matched electrode radius (the FEM electrode is a σ=5 mm Gaussian blob "
      "clipped by the skin — the analytical disk radius that matches it tells its effective size)",
      principle="SFAP = ∫ Vm·win·φ'' dz (integration by parts) — φ'' is what must agree")

# depth law at the SFAP level (DC-free): is the FEM/analytical ratio one constant? And is any
# drift the electrode model, the monopole fit, or the engine?
def drift(dim1=5.0, cfg=None, route="spatial"):
    ra = []
    for r in (33.0, 30.0, 27.0, 25.0, 20.0):
        pa, _ = ana_phi(r, dim1=dim1); pf = fem_phi(r, 0.0)
        if route == "fourier":
            ra.append(p2p(fourier_physical(pf, DZ, 60, 60, -20.0)[1]) / p2p(fourier_physical(pa, DZ, 60, 60, -20.0)[1]))
        else:
            ra.append(p2p(sfap(pf, DZ, 60, 60, -20.0, cfg)[1]) / p2p(sfap(pa, DZ, 60, 60, -20.0, cfg)[1]))
    ra = np.array(ra); return float(ra.max() / ra.min()), ra / ra[0]
d_gold = drift(5.0)
d_none = drift(5.0, golden_cfg(denoise="none"))
d_four = drift(5.0, route="fourier")
d_p5, d_p7 = drift(5.0, golden_cfg(denoise_n_poles=5))[0], drift(5.0, golden_cfg(denoise_n_poles=7))[0]
# is the erratic amplitude just Nyquist-scale ripple? measure p2p after a 500 Hz low-pass
from scipy.signal import butter, filtfilt
_b, _a = butter(4, 500.0 / (FS / 2))
def drift_lp(cfg):
    ra = []
    for r in (33.0, 30.0, 27.0, 25.0, 20.0):
        sa = filtfilt(_b, _a, sfap(ana_phi(r)[0], DZ, 60, 60, -20.0, cfg)[1])
        sf = filtfilt(_b, _a, sfap(fem_phi(r, 0.0), DZ, 60, 60, -20.0, cfg)[1])
        ra.append(p2p(sf) / p2p(sa))
    ra = np.array(ra); return float(ra.max() / ra.min()), ra / ra[0]
lp_gold, lp_none = drift_lp(golden_cfg()), drift_lp(golden_cfg(denoise="none"))
# hypothesis: the σ=5 mm Gaussian "electrode" reaches into the muscle → slower depth decay.
# Repeat the Fourier-route and golden drifts with a σ=1 mm source.
phi_s1 = cyl_fem.sigma1_lines()
def fem_phi_s1(r):
    return cyl_fem.window(phi_s1[int(np.argmin(np.abs(RADII - r))), 0], Z_ABS, ZE, W, DZ)
def drift_s1(route):
    ra = []
    for r in (33.0, 30.0, 27.0, 25.0, 20.0):
        pa, pf = ana_phi(r)[0], fem_phi_s1(r)
        if route == "fourier":
            ra.append(p2p(fourier_physical(pf, DZ, 60, 60, -20.0)[1]) / p2p(fourier_physical(pa, DZ, 60, 60, -20.0)[1]))
        else:
            ra.append(p2p(filtfilt(_b, _a, sfap(pf, DZ, 60, 60, -20.0)[1])) / p2p(filtfilt(_b, _a, sfap(pa, DZ, 60, 60, -20.0)[1])))
    ra = np.array(ra); return float(ra.max() / ra.min()), ra / ra[0]
s1_four, s1_gold = drift_s1("fourier"), drift_s1("golden")
# and the spatial engine with the Fourier route's own φ smoothing (Butterworth 0.03 cyc/sample ≈ 32 mm)
d_butter = drift(5.0, golden_cfg(denoise="butterworth"))
FIG["a12"] = (d_four[1], s1_four[1], lp_gold[1], s1_gold[1])
check(T, "A1.2 FEM/analytical SFAP amplitude ratio is one constant across depth (same depth law)",
      d_butter[0] <= 1.35 and s1_four[0] <= 1.5,
      f"p2p ratio FEM/ana, r=33→20 mm, rel. to r=33 — golden: {np.round(d_gold[1], 2).tolist()} "
      f"(spread {d_gold[0]:.2f}×); no denoise: {np.round(d_none[1], 2).tolist()} ({d_none[0]:.2f}×); "
      f"5/7 poles: {d_p5:.2f}×/{d_p7:.2f}×; Fourier route: {np.round(d_four[1], 2).tolist()} ({d_four[0]:.2f}×). "
      f"After a 500 Hz low-pass — golden: {np.round(lp_gold[1], 2).tolist()} ({lp_gold[0]:.2f}×), "
      f"no denoise: {np.round(lp_none[1], 2).tolist()} ({lp_none[0]:.2f}×).  "
      f"With a σ=1 mm electrode source instead of σ=5 mm — Fourier route: {np.round(s1_four[1], 2).tolist()} "
      f"({s1_four[0]:.2f}×), golden in-band: {np.round(s1_gold[1], 2).tolist()} ({s1_gold[0]:.2f}×).  "
      f"Spatial engine with Butterworth φ-smoothing (the Fourier route's, ≈32 mm cutoff): "
      f"{np.round(d_butter[1], 2).tolist()} ({d_butter[0]:.2f}×)",
      "TWO findings. (i) A smooth 1.3–1.4× drift survives every preprocessing and a σ=1 mm source: the FEM "
      "cylinder decays ~30 % slower with depth than the analytical one over 7→20 mm — a volume-conductor "
      "difference still to be attributed (mesh/skin-layer resolution vs the analytical k-grid). (ii) The golden "
      "recipe's SFAP amplitude on FEM φ is erratic at the ±40 % level (not high-frequency: survives a 500 Hz "
      "low-pass; differs between σ=5 and σ=1 sources) because it integrates φ'' faithfully down to ~8 mm "
      "wavelengths where FEM φ carries mesh structure; the 3-monopole fit does not remove it (5/7 poles neither) "
      "and Butterworth φ-smoothing does — at the cost of the shallow-fibre detail",
      principle="a depth-dependent ratio means the two volume conductors (or the preprocessing) decay differently",
      known=True)

ang_r = []
for th in (10.0, 20.0, 30.0, 45.0):
    pa, _ = ana_phi(30.0, distfib=th); pf = fem_phi(30.0, th)
    ang_r.append(float(np.corrcoef(dc_free(pa)[core], dc_free(pf)[core])[0, 1]))
check(T, "A1.3 FEM φ(z) vs analytical for fibres off the electrode meridian (θ = 10–45°)",
      min(ang_r) >= 0.99, f"r = {np.round(ang_r, 4).tolist()}", "r ≥ 0.99",
      principle="lateral (circumferential) decay of the surface lead field")

# =========================================================================== A2
# --- A2.1 FEM-φ pipeline vs analytical-φ pipeline (isolates the volume conductor) ---
pipe_r, pipe_r_raw, jag = [], [], []
from emgforge.synthesis.metrics import jaggedness
for r in (33.0, 30.0, 25.0, 20.0):
    for posz, l1, l2 in ((0.0, 60, 60), (-20.0, 40, 80)):
        pa, _ = ana_phi(r); pf = fem_phi(r, 0.0)
        t_a, s_a = sfap(pa, DZ, l1, l2, posz); t_f, s_f = sfap(pf, DZ, l1, l2, posz)
        pipe_r.append(float(np.corrcoef(s_a, s_f)[0, 1]))
        t_n, s_n = sfap(pf, DZ, l1, l2, posz, golden_cfg(denoise="none"))
        pipe_r_raw.append(float(np.corrcoef(s_a, s_n)[0, 1])); jag.append((jaggedness(s_n), jaggedness(s_f)))
FIG["a2"] = (t_a, s_a, t_f, s_f, s_n)
four_r = []                                              # the sanity suite's route: both φ → Fourier
for r in (33.0, 25.0, 20.0):
    ta_, ma_ = fourier_physical(ana_phi(r)[0], DZ, 60, 60, 0.0); tf_, mf_ = fourier_physical(fem_phi(r, 0.0), DZ, 60, 60, 0.0)
    four_r.append(float(np.corrcoef(ma_, mf_)[0, 1]))
check(T, "A2.1 pipeline on FEM φ ≡ pipeline on analytical φ (4 depths × 2 NMJ offsets)",
      min(pipe_r) >= 0.95,
      f"golden spatial: r = {np.round(pipe_r, 3).tolist()} (depth 33,33,30,30,25,25,20,20 mm; no denoise: "
      f"{np.round(pipe_r_raw, 3).tolist()});  Fourier route (r=33/25/20): {np.round(four_r, 3).tolist()}",
      "r ≥ 0.95 with the golden recipe; deep fibres ≥ 0.99. The shallow-fibre residual is the FEM field "
      "(electrode model + mesh ripple through φ''), not the engine (A0.1)",
      principle="same engine on both φ, so this isolates the volume conductor's accuracy at the SFAP level",
      known=True)
check(T, "A2.2 monopole denoise removes FEM mesh ripple without changing the analytical answer",
      all(jf <= jn for jn, jf in jag) and min(pipe_r) >= min(pipe_r_raw) - 1e-6,
      f"SFAP jaggedness raw→denoised = {[f'{a:.3f}→{b:.3f}' for a, b in jag[:4]]} …",
      "denoised jaggedness ≤ raw; agreement not reduced", refs="GOLDEN_METHOD.md §2")

# --- A2.3 vs the Farina generator: timing of the end-of-fibre potential ---------
# EOF isolated by subtraction: SFAP(tendon at L) − SFAP(tendon at L+60): the two are identical
# until the wave front reaches the near tendon at L/v, so the onset of the difference IS the
# extinction instant — no peak-picking, no dependence on the IAP's internal shape.
def eof_onset(t, a, b, frac=0.03):
    d = np.abs(a - b); k = np.where(d > frac * d.max())[0]
    return float(t[k[0]]) if len(k) else np.nan
eofs = []
pf30 = fem_phi(30.0, 0.0)
for L_near in (40.0, 60.0):                             # NMJ at −20 → near tendon 40 mm away, or symmetric 60
    posz = -20.0 if L_near == 40.0 else 0.0
    l2 = 80.0 if L_near == 40.0 else 60.0
    t_F, s_F, _ = farina(zi=posz, L1=60.0, L2=60.0); t_F2, s_F2, _ = farina(zi=posz, L1=60.0, L2=120.0)
    t_s, s_s = sfap(pf30, DZ, L_near, l2, posz, golden_cfg(fiber_window="boxcar"))
    t_s2, s_s2 = sfap(pf30, DZ, L_near + 60.0, l2, posz, golden_cfg(fiber_window="boxcar"))
    eofs.append((L_near / V, eof_onset(t_F, s_F[0], s_F2[0]), eof_onset(t_s, s_s, s_s2)))
err_pipe = max(abs(e[2] - e[0]) for e in eofs); err_far = max(abs(e[1] - e[0]) for e in eofs)
check(T, "A2.3 end-of-fibre onset lands at L/v in the FEM pipeline (and where the Farina generator puts it)",
      err_pipe <= 0.5,
      "expected/Farina/pipeline (ms): " + ", ".join(f"{a:.1f}/{b:.2f}/{c:.2f}" for a, b, c in eofs)
      + f"  → pipeline |Δ| ≤ {err_pipe:.2f} ms; Farina onset leads by up to {err_far:.2f} ms "
        "(its IAP body sits ahead of the front that defines L/v — see A0.3)",
      "pipeline |Δ| ≤ 0.5 ms",
      principle="the non-propagating terminal potential starts when the wave front reaches the tendon",
      refs="Gootzen et al. 1991; Dimitrova & Dimitrov 2002; Rodriguez-Falces & Place 2018 (M-wave shoulder at L/CV)")

# --- A2.4 propagation / CV across a longitudinal array, both models --------------
t_F, s_F, zc = farina(zi=0.0, L1=100, L2=100, channels=9, dint=10.0)
cv_F, r2_F, _ = cv_from_array(s_F[5:], 10.0, FS)         # channels z = +10 … +40 (one direction)
pf = fem_phi(30.0, 0.0)
stack = np.array([sfap(shift_phi(pf, DZ, -zc_), DZ, 100, 100, 0.0)[1] for zc_ in zc[5:]])
cv_P, r2_P, _ = cv_from_array(stack, 10.0, FS)
FIG["a2cv"] = (t_F, s_F, stack, zc)
check(T, "A2.4 conduction velocity recovered from a 4-channel longitudinal array (both models)",
      abs(cv_F - V) / V <= 0.05 and abs(cv_P - V) / V <= 0.05 and min(r2_F, r2_P) > 0.99,
      f"Farina {cv_F:.2f} m/s (R²={r2_F:.3f}); FEM pipeline {cv_P:.2f} m/s (R²={r2_P:.3f}); set {V}",
      "within 5 % of the set CV, linear lag–distance R² > 0.99",
      refs="Farina & Merletti 2004 (CV estimation); Arendt-Nielsen & Zwarts 1989")

# --- A2.5 waveform agreement with the Farina generator (informational, see A0.3) --
wf_r = []
for posz, l1, l2 in geoms:
    t_F, s_F, _ = farina(zi=posz)
    t_s, s_s = sfap(fem_phi(30.0, 0.0), DZ, l1, l2, posz, golden_cfg(fiber_window="boxcar"))
    t_f, s_f = fourier_physical(fem_phi(30.0, 0.0), DZ, l1, l2, posz)
    wf_r.append((best_r(t_F, s_F[0], t_s, s_s)[0], best_r(t_F, s_F[0], t_f, s_f, 15)[0]))
FIG["a2wf"] = (t_F, s_F[0], t_s, s_s, t_f, s_f)
check(T, "A2.5 waveform |r| vs the Farina generator (sign-agnostic, ±6 ms lag)",
      min(abs(a) for a, _ in wf_r) >= 0.9,
      "spatial/Fourier: " + ", ".join(f"{a:+.2f}/{b:+.2f}" for a, b in wf_r) + " at NMJ 0/−20/−30 mm",
      "|r| ≥ 0.9 — currently limited by the IAP-orientation/sign conventions of the Fourier/Farina family (A0.3)",
      known=True)

# =========================================================================== A3
rot = [float(np.corrcoef(fem_phi(r, 0.0, key="phi_e_th30"), fem_phi(r, 30.0))[0, 1]) for r in (33.0, 25.0)]
rot_a = [fem_phi(r, 0.0, key="phi_e_th30")[W // 2] / fem_phi(r, 30.0)[W // 2] for r in (33.0, 25.0)]
check(T, "A3.1 rotational symmetry: electrode +30° ≡ fibre −30°",
      min(rot) > 0.999 and max(abs(a - 1) for a in rot_a) < 0.03,
      f"r = {np.round(rot, 5).tolist()}, peak ratio = {np.round(rot_a, 4).tolist()}",
      "r > 0.999, peak within 3 % (mesh-discretisation level)")
# compare on the fibre core (|z| ≤ 60): the +20 mm window would otherwise reach past the mesh end
tr = [float(np.corrcoef(fem_phi(r, 0.0, key="phi_e_z140")[core], fem_phi(r, 0.0, centre=ZE - 20.0)[core])[0, 1]) for r in (33.0, 25.0)]
tr_a = [fem_phi(r, 0.0, key="phi_e_z140")[W // 2] / fem_phi(r, 0.0, centre=ZE - 20.0)[W // 2] for r in (33.0, 25.0)]
check(T, "A3.2 axial translation: electrode +20 mm ≡ fibre window −20 mm (Neumann end effect)",
      min(tr) > 0.995 and max(abs(a - 1) for a in tr_a) < 0.03,
      f"r = {np.round(tr, 5).tolist()}, peak ratio = {np.round(tr_a, 4).tolist()} (electrode 100 mm from the mesh end)",
      "r > 0.995, peak within 3 % — the residual (≤0.5 %) is the finite-length (240 mm) mesh's end effect",
      refs="cf. tests/regression/NEUMANN_BUDGET.md (L240 vs L400: min MUAP r 0.988)")
a_at_b, b_at_a = cyl_fem.reciprocity_interior()
check(T, "A3.3 reciprocity between two interior points in the anisotropic muscle",
      abs(a_at_b / b_at_a - 1) < 0.05,
      f"φ_A(B) = {a_at_b:.4e}, φ_B(A) = {b_at_a:.4e}, ratio {a_at_b / b_at_a:.4f}",
      "ratio 1 ± 5 % (symmetric σ ⇒ Green's function symmetric)",
      principle="the whole lead-field method rests on reciprocity (Helmholtz)",
      refs="Plonsey 1963; Malmivuo & Plonsey 1995 ch. 11")

# =========================================================================== figure
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
ax = axes[0, 0]
for (t_r, s_r_, t_s, s_s), (posz, *_), c in zip(FIG["a0"], geoms, ("C0", "C1", "C2")):
    ax.plot(t_r, s_r_ / np.abs(s_r_).max(), color=c, lw=2.2, alpha=.5, label=f"first principles, NMJ {posz:+g}")
    ax.plot(t_s, s_s / np.abs(s_s).max(), color=c, lw=1, ls="--")
ax.set_xlim(-3, 30); ax.set_title("A0.1  spatial engine (dashed) on closest-form φ"); ax.legend(fontsize=7); ax.set_xlabel("ms")
ax = axes[0, 1]; t_r, s_r_, t_m, s_m, t_f, s_f = FIG["a0f"]
ax.plot(t_r, s_r_ / np.abs(s_r_).max(), "k", lw=2, label="first principles"); ax.plot(t_m, s_m / np.abs(s_m).max(), "0.6", lw=1.5, label="mirrored IAP")
ax.plot(t_f, s_f / np.abs(s_f).max(), "C3--", label="Fourier engine"); ax.set_xlim(-3, 30); ax.set_title("A0.3  Fourier engine vs first principles (NMJ −30)"); ax.legend(fontsize=7)
ax = axes[0, 2]
for r, a, f in FIG["a1"]:
    ax.plot(z, a / a.max(), lw=2, alpha=.5, label=f"analytical r={r:g}"); ax.plot(z, f / f.max(), "k--", lw=.8)
ax.set_xlim(-80, 80); ax.set_title("A1  φ(z): analytical (solid) vs FEM (dashed)"); ax.set_xlabel("z (mm)"); ax.legend(fontsize=7)
ax = axes[1, 0]; t_a, s_a, t_f2, s_f2, s_n = FIG["a2"]
ax.plot(t_a, s_a, "k", lw=2, label="analytical φ → golden"); ax.plot(t_f2, s_f2, "C0--", label="FEM φ → golden"); ax.plot(t_f2, s_n, "C1:", lw=.8, label="FEM φ, no denoise")
ax.set_xlim(-3, 30); ax.set_title("A2.1/2.2  pipeline: FEM vs analytical φ (r=20, NMJ −20)"); ax.legend(fontsize=7)
ax = axes[1, 1]; t_F, s_F, stack, zc = FIG["a2cv"]
for k in range(4):                                       # each channel peak-normalised
    ax.plot(t_F, s_F[5 + k] / np.abs(s_F[5 + k]).max() * 0.45 + k, "k", lw=1.2)
    ax.plot(t_F, stack[k] / np.abs(stack[k]).max() * 0.45 + k, "C0--", lw=1)
ax.set_xlim(-3, 30); ax.set_title(f"A2.4  array z=+10…+40: Farina (black) vs pipeline (dashed) → CV {cv_F:.2f}/{cv_P:.2f}"); ax.set_yticks([])
ax = axes[1, 2]; t_F, s_F0, t_s, s_s, t_f, s_f = FIG["a2wf"]
ax.plot(t_F, s_F0 / np.abs(s_F0).max(), "k", lw=2, label="Farina generator"); ax.plot(t_s, s_s / np.abs(s_s).max(), "C0--", label="FEM → spatial (boxcar)"); ax.plot(t_f, s_f / np.abs(s_f).max(), "C3:", label="FEM → Fourier")
ax.set_xlim(-3, 30); ax.set_title("A2.5  waveforms vs Farina (NMJ −30)"); ax.legend(fontsize=7)
fig.suptitle("Tier A — cylinder: analytical vs pipeline", fontsize=13); fig.tight_layout()
fig.savefig(OUT / "tier_a_cylinder.png", dpi=120); print("wrote", OUT / "tier_a_cylinder.png")
sys.exit(finish(T))
