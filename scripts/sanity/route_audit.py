"""Route audit — at every fork of the emgforge chain, a measured, justified verdict.

One command. Part A runs the cheap forks LIVE on the analytical cylinder (fibre 10 mm
below the skin, NMJ 20 mm proximal of the electrode, tendons 60 mm either side of the
electrode, v = 4 m/s, fs = 4096 Hz); part B reads the RECORDED evidence for the
expensive forks (the validation tiers under ``_results/validation`` and the paper's key
numbers on branch ``paper/arxiv``). Every fork gets: the options, the measured numbers
for each, the verdict (which to use, and when the alternative is legitimate) and a
PASS/FAIL of the *current default* against that verdict.

  1  synthesis engine          live      direct (spatial) vs Fourier vs closed-form oracle
  2  lead-field conditioning   live      none / butterworth / monopole(3, 5, 7) on FEM φ
  3  tendon window             live      boxcar / tukey / hann / one_sided
  4  sampling                  live      fs 2048 vs 4096, upsample 1/2/4, w 256 vs 512
  5  time base                 live      center_time False vs True
  6  amplitude constant        live      engine / oracle ratio at v = 2, 3, 4, 5 (the 1/v trap)
  7  volume conductor          recorded  analytical vs FEM; electrode source σ_s 5 vs 1 mm
  8  fibre bed                 recorded  straight vs harmonic single-NMJ (study_fibres)
  9  electrode grid            live+rec  legacy vertex-snapped cache vs regular ray-cast grid
  10 entry point               recorded  run_pipeline.py vs Simulator.from_mri vs f2_common
  11 activation defaults       recorded  onion skin, ISI CoV, force calibration, refractory floor

Run:   python scripts/sanity/route_audit.py                     (about one minute)
Out:   _results/sanity/ROUTE_AUDIT.md + ROUTE_AUDIT.json, and a copy of the markdown at
       docs/validation/ROUTE_AUDIT.md.  Exit status 1 if any default FAILs its verdict.

The recipe under test is ``emgforge.synthesis.production_config()`` — the single source
of truth for the direct line-source recipe (DIRECT_LINE_SOURCE.md §0). Every live variant
is ``dataclasses.replace`` of that object, so the audit always tests the shipped default.
"""
from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))          # harness, cyl_fem
OUT = ROOT / "_results/sanity"
OUT.mkdir(parents=True, exist_ok=True)
DOC_COPY = ROOT / "docs/validation/ROUTE_AUDIT.md"
T0 = time.time()

from harness import (V, FS, W, ana_phi, best_r, fourier_physical, mnf, p2p,   # noqa: E402
                     phi_inf, sfap_first_principles, shift_phi)
from emgforge.synthesis.engines.spatial import (                                # noqa: E402
    SpatialConfig, build_csd_matrix, compute_sfap_spatial)
from emgforge.synthesis.preprocessing import (                                  # noqa: E402
    create_fiber_windows, denoise_field_n, smooth_butterworth)
from emgforge.synthesis.metrics import jaggedness                               # noqa: E402
from emgforge.synthesis.fibres import FibreBed                                  # noqa: E402
from emgforge.synthesis.api import field_to_muap                                # noqa: E402

try:                                            # the single source of truth (other agent's work)
    from emgforge.synthesis import production_config
    PC_SRC = "emgforge.synthesis.production_config"
except ImportError:                             # pragma: no cover — before that lands
    from emgforge.mri.pipeline import production_config
    PC_SRC = "emgforge.mri.pipeline.production_config"

# --------------------------------------------------------------------------- the recipe
DEFAULT = production_config()                   # as shipped (MRI regime: fs 2048, v 4, w 256)
DOCUMENTED = dict(denoise="monopole", denoise_n_poles=3, csd_derivative=2, upsample_factor=2,
                  fiber_window="one_sided", tukey_alpha=0.25, edge_taper_left=5,
                  edge_taper_right=10, center_time=False, t_start_ms=-10.0, v=4.0,
                  fsamp=2048.0, w=256, polarity=1)                 # DIRECT_LINE_SOURCE.md §0


def recipe(**over) -> SpatialConfig:
    """The shipped recipe in the cylinder regime (fs 4096, dz = v/fs), plus a deliberate
    deviation ``**over`` for the fork under test."""
    return replace(DEFAULT, **{**dict(fsamp=FS, v=V, w=W), **over})


def sfap(phi, dz, len1, len2, posz, cfg=None):
    t, s, _ = compute_sfap_spatial(np.asarray(phi, float), float(dz), len1, len2, posz,
                                   cfg or recipe())
    return np.asarray(t, float), np.asarray(s, float)


# --------------------------------------------------------------------------- geometry
DZ = V * 1000.0 / FS                    # 0.977 mm
POSZ, L1, L2 = -20.0, 40.0, 80.0        # NMJ 20 mm proximal of the electrode; tendons at −60 / +60 mm
T_EOF = L1 / V                          # 10 ms: the near (proximal) tendon
DEPTH_R = 30.0                          # radial position 30 mm in a 40 mm cylinder → 10 mm below the skin
Z = (np.arange(W) - W // 2) * DZ
TT = np.arange(W) / FS * 1000.0 + DEFAULT.t_start_ms


# --------------------------------------------------------------------------- helpers
def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def eof_onset(t, a, b, frac=0.03):
    """Onset of |SFAP(tendon at L) − SFAP(tendon at L+60)| — the extinction instant."""
    d = np.abs(a - b)
    k = np.where(d > frac * d.max())[0]
    return float(t[k[0]]) if len(k) else float("nan")


def dc_free(p, n_edge=12):
    return p - np.mean(np.r_[p[:n_edge], p[-n_edge:]])


def dd(p, dz):
    return np.gradient(np.gradient(p, dz), dz)


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and (abs(x) < 1e-3 and x != 0 or abs(x) >= 1e5):
        return f"{x:.1e}"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def rng(x, nd=1):
    return f"{x[0]:.{nd}f}–{x[1]:.{nd}f}" if x else "n/a"


def opt(summary, verdict, **numbers):
    return dict(summary=summary, verdict=verdict, numbers=numbers)


FORKS: list[dict] = []


def fork(n, name, kind, options, verdict, default, thresholds=None, provenance=None, notes=None):
    rec = dict(fork=n, name=name, kind=kind, options=options, verdict=verdict, default=default,
               thresholds=thresholds or {}, provenance=provenance or [], notes=notes or [])
    FORKS.append(rec)
    tag = "PASS" if default["passed"] else "FAIL"
    print(f"[{tag}] fork {n:2d} · {name}  →  use: {verdict['use']}")
    for oname, o in options.items():
        print(f"        - {oname}: {o['summary']}\n            → {o['verdict']}")
    print(f"        default: {default['value']}" + (f"  ({default['note']})" if default.get("note") else ""))
    print()
    return rec


# --------------------------------------------------------------------------- recorded evidence
def validation(tier):
    p = ROOT / "_results/validation" / f"{tier}.json"
    return json.loads(p.read_text()) if p.exists() else []


def vrec(tier, prefix):
    return next((r for r in validation(tier) if r["name"].startswith(prefix)), None)


def paper_json(name):
    local = ROOT / "paper/figures" / f"{name}.json"
    if local.exists():
        return json.loads(local.read_text()), f"paper/figures/{name}.json (working tree)"
    out = subprocess.run(["git", "show", f"paper/arxiv:paper/figures/{name}.json"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode == 0:
        return json.loads(out.stdout), f"`git show paper/arxiv:paper/figures/{name}.json`"
    return None, f"{name}.json unavailable (no paper/figures on this branch, no paper/arxiv ref)"


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


print("=" * 78)
print("emgforge route audit — one verdict per fork, measured   (recipe:", PC_SRC + ")")
print("=" * 78, "\n")

# =========================================================================== recipe guard
mismatch = {k: (getattr(DEFAULT, k), v) for k, v in DOCUMENTED.items() if getattr(DEFAULT, k) != v}
bed1 = FibreBed.uniform(1, DZ, L1, L2, POSZ)
phi0 = phi_inf(Z)                                # closed-form φ, electrode 10 mm from the fibre line
res_none = field_to_muap(phi0, bed1, None)
engine_none = ("spatial (direct, physical time)" if res_none.time_convention == "physical"
               else "Fourier (window-centred)")
RECIPE_GUARD = dict(source=PC_SRC, documented=DOCUMENTED, shipped={k: getattr(DEFAULT, k) for k in DOCUMENTED},
                    mismatches=mismatch, field_to_muap_config_none=engine_none)

# =========================================================================== fork 1 — engine
t_o, s_o = sfap_first_principles(POSZ, L1, L2)                       # closed-form line-source oracle
cfg_eng = recipe(fiber_window="boxcar", denoise="none")               # isolates the integral
t_s, s_s = sfap(phi0, DZ, L1, L2, POSZ, cfg_eng)
r_s, lag_s, amp_s = best_r(t_o, s_o, t_s, s_s)
t_p, s_p = sfap(phi0, DZ, L1, L2, POSZ, recipe())                     # the full production recipe
r_p, lag_p, amp_p = best_r(t_o, s_o, t_p, s_p)
t_f, s_f = fourier_physical(phi0, DZ, L1, L2, POSZ)
r_f, lag_f, amp_f = best_r(t_o, s_o, t_f, s_f, max_lag_ms=15)
t_m, s_m = sfap_first_principles(POSZ, L1, L2, mirror=True)
r_fm = best_r(t_m, s_m, t_f, s_f, max_lag_ms=15)[0]
net = {}
for win in ("boxcar", "one_sided"):
    csd = build_csd_matrix(Z, TT, POSZ, L1, L2, DZ, recipe(fiber_window=win))
    net[win] = float(np.max(np.abs(csd.sum(1))) / np.max(np.abs(csd).sum(1)))
_, s_o2 = sfap_first_principles(POSZ, L1 + 60.0, L2)
_, s_s2 = sfap(phi0, DZ, L1 + 60.0, L2, POSZ, cfg_eng)
_, s_p2 = sfap(phi0, DZ, L1 + 60.0, L2, POSZ, recipe())
_, s_f2 = fourier_physical(phi0, DZ, L1 + 60.0, L2, POSZ)
eof = dict(oracle=eof_onset(t_o, s_o, s_o2), spatial_boxcar=eof_onset(t_s, s_s, s_s2),
           spatial_production=eof_onset(t_p, s_p, s_p2), fourier=eof_onset(t_f, s_f, s_f2))
TH1 = dict(r_min=0.999, lag_max_ms=0.1, amp_tol=0.02, monopole_free_max=1e-12, eof_tol_ms=0.5)
ok_spatial = (r_s >= TH1["r_min"] and abs(lag_s) <= TH1["lag_max_ms"] and abs(amp_s - 1) <= TH1["amp_tol"]
              and max(net.values()) < TH1["monopole_free_max"] and abs(eof["spatial_boxcar"] - T_EOF) <= TH1["eof_tol_ms"])
ok_fourier = r_f >= TH1["r_min"] and abs(lag_f) <= TH1["lag_max_ms"]
fork(1, "Synthesis engine", "live",
     {"direct (spatial line-source), boxcar / no denoise": opt(
         f"r = {r_s:.4f}, lag = {lag_s:+.2f} ms, amp ratio = {amp_s:.3f} vs the closed-form oracle; "
         f"monopole-free residual {net['boxcar']:.1e} (boxcar) / {net['one_sided']:.1e} (one_sided); "
         f"EOF onset {eof['spatial_boxcar']:.2f} ms (L/v = {T_EOF:.0f})",
         "USE — reproduces the integral" if ok_spatial else "unexpected: engine gate failed",
         r=r_s, lag_ms=lag_s, amp_ratio=amp_s, monopole_free=net, eof_onset_ms=eof["spatial_boxcar"]),
      "direct, full production recipe (monopole 3 + one_sided)": opt(
         f"r = {r_p:.4f}, lag = {lag_p:+.2f} ms, amp ratio = {amp_p:.3f} vs the sharp-tendon oracle "
         f"(the one-sided taper softens the tendon term — fork 3); EOF onset {eof['spatial_production']:.2f} ms",
         "USE (production) — same integral, tendon softened by design",
         r=r_p, lag_ms=lag_p, amp_ratio=amp_p, eof_onset_ms=eof["spatial_production"]),
      "Fourier (Farina 2001 radon/pare)": opt(
         f"signed r = {r_f:+.3f} at lag {lag_f:+.2f} ms, amp ratio = {amp_f:.3f}; vs a spatially mirrored IAP "
         f"r = {r_fm:+.3f}; EOF onset {eof['fourier']:.2f} ms; no CSD (monopole test n/a); window-centred time",
         "shape oracle only (r = 0.997 vs the Farina MATLAB reference); not for physical time or MRI",
         signed_r=r_f, lag_ms=lag_f, amp_ratio=amp_f, r_vs_mirrored_iap=r_fm, eof_onset_ms=eof["fourier"],
         gate_passed=ok_fourier)},
     dict(use="direct line-source (spatial) engine",
          when_alternative="Fourier only as the shape oracle against the Farina MATLAB reference set "
                           "(tests/synthesis/test_golden.py); its polarity is anti-phase and its time base "
                           "window-centred, so never for physical-time / MRI synthesis",
          rationale="the direct engine reproduces the closed-form line-source integral to r ≥ 0.999 with zero "
                    "lag, a monopole-free source and the EOF at L/v; the Fourier engine does not"),
     dict(value=f"production_config() → {type(DEFAULT).__name__}; field_to_muap(config=None) → {engine_none}",
          passed=isinstance(DEFAULT, SpatialConfig) and res_none.time_convention == "physical" and ok_spatial,
          note="both the recipe object and the no-config default must be the direct engine"),
     TH1, ["live: closed-form oracle harness.sfap_first_principles (Rosenfalck 1969 / Dimitrov & Dimitrova 1998)",
           "recorded: _results/validation/A.json A0.1–A0.3, A0.5; key_numbers_f1.json fig4_first_principles"],
     notes=["The production recipe's earlier EOF 'onset' is the one-sided taper starting 25 % of the 40 mm semi-fibre "
            "before the tendon (30 mm / v = 7.5 ms) — by design, see fork 3.",
            f"The Fourier amplitude ratio ({amp_f:.2f}) is low here because the closed-form φ does not decay at the FFT "
            "window edges — the reason analytical φ is excluded from the Fourier reference set (DIRECT_LINE_SOURCE.md §6); "
            "on decaying Gaussian φ the two engines agree to 0.84–1.01."])

# =========================================================================== fork 2 — conditioning
pa_raw, _ = ana_phi(DEPTH_R)
pa = dc_free(pa_raw)
phi_source = ""
FEM_LINE = None                                  # (line, z_abs, ze) of the cached FEM φ, for re-gridding
try:
    import cyl_fem
    if cyl_fem.CACHE.exists():
        cache = dict(np.load(cyl_fem.CACHE))
        i = int(np.argmin(np.abs(cache["radii"] - DEPTH_R))); j = int(np.argmin(np.abs(cache["thetas"] - 0.0)))
        FEM_LINE = (cache["phi"][i, j], cache["z_abs"], float(cache["ze"]))
        pf_raw = cyl_fem.window(*FEM_LINE, W, DZ)
        phi_source = (f"FEM cylinder lead field, {cyl_fem.CACHE.relative_to(ROOT)} (r = 30 mm, θ = 0, "
                      f"σ_s = {cyl_fem.SOURCE_SIGMA:g} mm electrode source, mesh cl 0.3)")
except Exception as e:                           # pragma: no cover
    print("  (cyl_fem cache unavailable:", e, ")")
if not phi_source:
    rng = np.random.default_rng(0)
    pf_raw = pa + pa.max() * (0.01 * np.sin(2 * np.pi * Z / 4.0) + 0.005 * rng.standard_normal(W))
    phi_source = "SYNTHETIC mesh-scale ripple on the analytical φ (FEM cache absent): 1 % sinusoid at 4 mm + 0.5 % noise"
pf = dc_free(pf_raw)
scale = float(pa.max() / pf.max())
pf = pf * scale                                  # peak-matched so amplitudes compare in shape terms
core = np.abs(Z) <= 60.0
dd_a = dd(pa, DZ)
r_phi = corr(pa[core], pf[core])
t_ref, s_ref = sfap(pa, DZ, L1, L2, POSZ, recipe(denoise="none"))      # clean analytical answer
mnf_ref, p2p_ref = mnf(s_ref, FS), p2p(s_ref)
from scipy.signal import butter, filtfilt                                   # noqa: E402
_bb, _ab = butter(4, 500.0 / (FS / 2))
_, s_raw = sfap(pf, DZ, L1, L2, POSZ, recipe(denoise="none"))
s_inband = filtfilt(_bb, _ab, s_raw)               # A1.2's in-band FEM reference: raw φ, 500 Hz low-pass
mnf_ib, p2p_ib = mnf(s_inband, FS), p2p(s_inband)
variants = {"none": ("none", None), "butterworth(0.03)": ("butterworth", None),
            "monopole(3)": ("monopole", 3), "monopole(5)": ("monopole", 5), "monopole(7)": ("monopole", 7)}
C2 = {}
for name, (den, npoles) in variants.items():
    cfg = recipe(denoise=den, **({"denoise_n_poles": npoles} if npoles else {}))
    if den == "monopole":
        pc = denoise_field_n(pf, DZ, n=npoles)
    elif den == "butterworth":
        pc = smooth_butterworth(pf, cfg.butterworth_cutoff, cfg.butterworth_order)
    else:
        pc = pf
    t1 = time.time()
    _, s_v = sfap(pf, DZ, L1, L2, POSZ, cfg)
    dt = time.time() - t1
    _, s_va = sfap(pa, DZ, L1, L2, POSZ, cfg)
    C2[name] = dict(jaggedness=jaggedness(s_v), r_phidd_vs_analytical=corr(dd(pc, DZ)[core], dd_a[core]),
                    mnf_retained=mnf(s_v, FS) / mnf_ib, p2p_retained=p2p(s_v) / p2p_ib,
                    r_sfap_vs_inband=corr(s_v, s_inband),
                    mnf_vs_analytical=mnf(s_v, FS) / mnf_ref, p2p_vs_analytical=p2p(s_v) / p2p_ref,
                    r_sfap_vs_analytical=corr(s_v, s_ref), r_on_analytical_phi=corr(s_va, s_ref),
                    p2p_on_analytical_phi=p2p(s_va) / p2p_ref, seconds=dt)
TH2 = dict(jaggedness_max=0.005, r_phidd_min=0.90, mnf_retained=[0.85, 1.15], p2p_retained=[0.8, 1.2],
           noop_r_min=0.99)
m3 = C2["monopole(3)"]
ok_m3 = (m3["jaggedness"] <= C2["none"]["jaggedness"] and m3["jaggedness"] <= TH2["jaggedness_max"]
         and m3["r_phidd_vs_analytical"] >= TH2["r_phidd_min"]
         and TH2["mnf_retained"][0] <= m3["mnf_retained"] <= TH2["mnf_retained"][1]
         and TH2["p2p_retained"][0] <= m3["p2p_retained"] <= TH2["p2p_retained"][1]
         and m3["r_on_analytical_phi"] >= TH2["noop_r_min"])
a12 = vrec("A", "A1.2")
options2 = {}
for name, c in C2.items():
    s = (f"SFAP jaggedness {c['jaggedness']:.4f}; r(φ'') vs the analytical φ'' {c['r_phidd_vs_analytical']:.3f}; "
         f"vs the in-band raw FEM SFAP: MNF ×{c['mnf_retained']:.2f}, p2p ×{c['p2p_retained']:.2f}, r {c['r_sfap_vs_inband']:.3f}; "
         f"vs the analytical SFAP: MNF ×{c['mnf_vs_analytical']:.2f}, p2p ×{c['p2p_vs_analytical']:.2f}, r {c['r_sfap_vs_analytical']:.3f}; "
         f"on the analytical φ itself: r {c['r_on_analytical_phi']:.4f}, p2p ×{c['p2p_on_analytical_phi']:.3f}; "
         f"{c['seconds'] * 1e3:.0f} ms")
    if name == "monopole(3)":
        v = "USE — removes the ripple, keeps φ'' and the spectrum" if ok_m3 else "unexpected: monopole(3) gate failed"
    elif name.startswith("butterworth"):
        v = ("legitimate ONLY for relative amplitude-vs-depth laws on FEM φ (A1.2: 1.22× spread vs 1.69× direct); "
             "it destroys the absolute amplitude (×%.2f in-band) and spectrum (MNF ×%.2f) and is not a no-op even "
             "on the clean analytical φ (r %.2f)" % (c["p2p_retained"], c["mnf_retained"], c["r_on_analytical_phi"]))
    elif name == "none":
        v = "diagnostic only (raw ripple reaches φ'')"
    else:
        v = "no gain over 3 poles (A1.2: 5/7 poles 1.49×/1.51× amplitude spread)"
    options2[name] = opt(s, v, **c)
fork(2, "Lead-field conditioning (φ denoise before the CSD)", "live",
     options2,
     dict(use="monopole(3) — production_config().denoise = 'monopole', denoise_n_poles = 3",
          when_alternative="Butterworth(0.03) when the question is an amplitude law across FEM depths "
                           "(it halves the spectral content, so never for shape/spectrum); 'none' only on clean "
                           "analytical φ, where monopole(3) is a near no-op anyway",
          rationale="the 3-monopole fit is the analytic form the field has: it removes the mesh ripple from φ'' "
                    "(the kernel the SFAP integrates) without rounding the peak. CAVEAT (recorded A1.2): on FEM φ "
                    "the SFAP amplitude stays erratic at ±40 % across depth — not high-frequency, and 5/7 poles "
                    "do not remove it; shape, timing and spectrum are faithful"),
     dict(value=f"denoise = {DEFAULT.denoise!r}, denoise_n_poles = {DEFAULT.denoise_n_poles}",
          passed=DEFAULT.denoise == "monopole" and DEFAULT.denoise_n_poles == 3 and ok_m3),
     TH2, [f"live φ: {phi_source}; analytical φ: harness.ana_phi(30) (Farina 2004, Ø10 mm disk electrode); "
           f"r(φ FEM, φ analytical) on |z| ≤ 60 mm = {r_phi:.4f}; FEM φ peak scaled ×{scale:.3g} to the analytical peak",
           "reference for 'retained': the raw-φ SFAP low-passed at 500 Hz (A1.2's in-band content). The FEM φ at this "
           "depth is narrower than the analytical one (FWHM 45 vs 53 mm, A1.1: σ_s = 5 mm blob vs Ø10 mm disk), so every "
           "variant's MNF/p2p vs the ANALYTICAL SFAP exceeds 1 — fork 7's volume-conductor difference, not the conditioning",
           "recorded: _results/validation/A.json A1.1b, A1.2 (amplitude ±40 %), A2.2; key_numbers_f1.json fig3_recipe_anatomy.phi_r30",
           ("A1.2 measured: " + a12["measured"]) if a12 else "A1.2 record not found"],
     notes=[("A1.2 recorded: " + a12["expect"]) if a12 else ""])

# =========================================================================== fork 3 — tendon window
wins = ("boxcar", "tukey", "hann", "one_sided")
n_pts = max(int((L1 + L2) / (DZ / DEFAULT.upsample_factor)), 2)
n_left = int(n_pts * L1 / (L1 + L2))
C3 = {}
for w_ in wins:
    t_w, s_w = sfap(phi0, DZ, L1, L2, POSZ, recipe(fiber_window=w_, denoise="none"))
    r_w, lag_w, amp_w = best_r(t_o, s_o, t_w, s_w)
    cfg = recipe(fiber_window=w_)
    ta, a = sfap(pa, DZ, L1, L2, POSZ, cfg)
    _, b = sfap(pa, DZ, L1 + 60.0, L2, POSZ, cfg)
    wl, wr = create_fiber_windows(n_pts, L1 / (L1 + L2), window_type=w_, tukey_alpha=DEFAULT.tukey_alpha)
    tg, g = sfap(pa, DZ, 60.0, 60.0, 0.0, cfg)                       # electrode over the NMJ
    mg = (tg >= -1.0) & (tg <= 3.0)
    C3[w_] = dict(r_vs_oracle=r_w, lag_ms=lag_w, amp_vs_oracle=amp_w,
                  eof_over_propagating=float(np.abs(a - b).max() / p2p(b)), eof_onset_ms=eof_onset(ta, a, b),
                  window_at_nmj=[float(wl[n_left - 1]), float(wr[n_left])],
                  generation_lobe_p2p=p2p(g[mg]), p2p=p2p(a))
gen0 = C3["boxcar"]["generation_lobe_p2p"]
for c in C3.values():
    c["generation_lobe_vs_boxcar"] = c["generation_lobe_p2p"] / gen0
TH3 = dict(one_sided_r_min=0.95, boxcar_r_min=0.999, eof_ratio_lit=[0.02, 0.2], junction_flat=1.0, gen_lobe_min=0.95)
os_ = C3["one_sided"]
ok_os = (os_["r_vs_oracle"] >= TH3["one_sided_r_min"] and min(os_["window_at_nmj"]) == 1.0
         and TH3["eof_ratio_lit"][0] <= os_["eof_over_propagating"] <= TH3["eof_ratio_lit"][1]
         and os_["generation_lobe_vs_boxcar"] >= TH3["gen_lobe_min"])
options3 = {}
for w_, c in C3.items():
    s = (f"r vs sharp-tendon oracle {c['r_vs_oracle']:.4f} (lag {c['lag_ms']:+.2f} ms, amp {c['amp_vs_oracle']:.3f}); "
         f"EOF/propagating {c['eof_over_propagating']:.3f} (lit. 0.02–0.2 at 7–20 mm), EOF onset {c['eof_onset_ms']:.2f} ms; "
         f"window at the NMJ {c['window_at_nmj'][0]:.2f}/{c['window_at_nmj'][1]:.2f}; generation lobe "
         f"{c['generation_lobe_vs_boxcar']:.3f}× boxcar")
    v = {"boxcar": "sharp-tendon oracle comparisons and 'see the EOF' diagnostics",
         "tukey": "ARTEFACT — notches the junction (window 0 at the NMJ); legacy default, do not use",
         "hann": "ARTEFACT — notches the junction; reference only",
         "one_sided": ("USE — flat through the NMJ, tendon softened" if ok_os else "unexpected: one_sided gate failed")}[w_]
    options3[w_] = opt(s, v, **c)
fork(3, "Tendon (fibre-end) window", "live", options3,
     dict(use="one_sided (α = 0.25) — production_config().fiber_window",
          when_alternative="boxcar when comparing against a sharp-tendon oracle (first principles, Farina) or "
                           "to isolate the end-of-fibre potential; never tukey/hann (junction notch)",
          rationale="a symmetric window notches the junction where the travelling wave is born (window = 0 at "
                    "the NMJ); one_sided keeps the generation lobe at boxcar strength, keeps the EOF/propagating "
                    "ratio inside the literature range and stays within r ≥ 0.95 of the sharp-tendon oracle"),
     dict(value=f"fiber_window = {DEFAULT.fiber_window!r}, tukey_alpha = {DEFAULT.tukey_alpha}",
          passed=DEFAULT.fiber_window == "one_sided" and ok_os),
     TH3, ["live: r on the closed-form φ vs harness.sfap_first_principles (boxcar oracle); EOF ratio on the "
           "analytical cylinder φ at 10 mm depth by tier B4's subtraction (near tendon moved +60 mm)",
           "recorded: _results/validation/B.json B4b (0.021–0.204 at 7–20 mm); key_numbers_f1.json fig3_recipe_anatomy.windows; "
           "DIRECT_LINE_SOURCE.md §3"],
     notes=["'EOF onset' for the tapered windows is where the two fibres first differ: one_sided's 8.55 ms is the start of "
            "its 10 mm tendon ramp (30 mm / v = 7.5 ms), tukey's and hann's ≈1 ms is the junction notch itself (its "
            "width changes with the semi-fibre length) — the notch is the artefact."])

# =========================================================================== fork 4 — sampling
pa4, dz4 = ana_phi(DEPTH_R, fs=4096.0)
pa2, dz2 = ana_phi(DEPTH_R, fs=2048.0)
t4, s4 = sfap(pa4, dz4, L1, L2, POSZ, recipe(fsamp=4096.0))
t2, s2 = sfap(pa2, dz2, L1, L2, POSZ, recipe(fsamp=2048.0))
m = t2 <= t4.max()
r_fs, a_fs = corr(np.interp(t2[m], t4, s4), s2[m]), p2p(s2) / p2p(s4)
t2b, s2b = sfap(pa4, dz4, L1, L2, POSZ, recipe(fsamp=2048.0))          # MRI regime: dz ≈ 1 mm, fs 2048
m = t2b <= t4.max()
r_fs_b, a_fs_b = corr(np.interp(t2b[m], t4, s4), s2b[m]), p2p(s2b) / p2p(s4)
from scipy.interpolate import CubicSpline                                   # noqa: E402


def regrid(phi, dz_from, dz_to):
    z_from = (np.arange(len(phi)) - len(phi) // 2) * dz_from
    z_to = (np.arange(W) - W // 2) * dz_to
    return CubicSpline(z_from, phi)(np.clip(z_to, z_from[0], z_from[-1]))


DZ_MRI = 1.02                                    # the FCU bed's arc dz (key_numbers_pipeline: 1.023 mm), ≠ v·dt = 1.95 mm
pa_mri = regrid(pa4, dz4, DZ_MRI)
pf_mri = cyl_fem.window(*FEM_LINE, W, DZ_MRI) if FEM_LINE else regrid(pf_raw, DZ, DZ_MRI)
UP = {}
for label, pa_, dz_, fs_, den in (
        ("analytical φ, dz 0.98 mm / fs 4096 (cylinder tiers; v·dt = dz)", pa4, dz4, 4096.0, "none"),
        ("analytical φ, dz 1.02 mm / fs 2048 (MRI regime; v·dt = 1.95 mm ≠ dz)", pa_mri, DZ_MRI, 2048.0, "none"),
        ("FEM φ, dz 1.02 mm / fs 2048 (MRI regime, monopole 3)", pf_mri, DZ_MRI, 2048.0, "monopole"),
        ("FEM φ RAW (no denoise), dz 1.02 mm / fs 2048", pf_mri, DZ_MRI, 2048.0, "none"),
        ("analytical φ, dz 1.95 mm / fs 2048 (v·dt = dz)", pa2, dz2, 2048.0, "none")):
    ref = sfap(pa_, dz_, L1, L2, POSZ, recipe(fsamp=fs_, upsample_factor=4, denoise=den))[1]
    UP[label] = {}
    for u in (1, 2, 4):
        t1 = time.time()
        s = sfap(pa_, dz_, L1, L2, POSZ, recipe(fsamp=fs_, upsample_factor=u, denoise=den))[1]
        UP[label][u] = dict(jaggedness=jaggedness(s), r_vs_x4=corr(s, ref), p2p_vs_x4=p2p(s) / p2p(ref),
                            seconds=time.time() - t1)
zig = {k: v[1]["jaggedness"] / max(v[2]["jaggedness"], 1e-12) for k, v in UP.items()}
t512, s512 = sfap(pa4, dz4, L1, L2, POSZ, recipe(w=512))
r_w, a_w = corr(s512[:W], s4), p2p(s512) / p2p(s4)
TH4 = dict(r_min=0.995, amp_tol=0.02, jaggedness_max_x2=0.01, r_x2_vs_x4_min=0.999)
PROD_REGIMES = [k for k in UP if "RAW" not in k]              # the raw-φ regime is a diagnostic, not a route
ok_up = all(UP[k][2]["jaggedness"] <= TH4["jaggedness_max_x2"] and UP[k][2]["r_vs_x4"] >= TH4["r_x2_vs_x4_min"] for k in PROD_REGIMES)
ok_fs = r_fs > TH4["r_min"] and abs(a_fs - 1) < TH4["amp_tol"] and r_fs_b > TH4["r_min"] and abs(a_fs_b - 1) < TH4["amp_tol"]
ok_w = r_w > 0.9999 and abs(a_w - 1) < 1e-6
options4 = {
    "fs 4096 (dz = v/fs = 0.98 mm) vs fs 2048": opt(
        f"fs 2048 with dz = v/fs = 1.95 mm: r = {r_fs:.5f}, p2p ratio {a_fs:.4f}; fs 2048 with dz = 0.98 mm "
        f"(the MRI regime, arc dz ≈ 1 mm): r = {r_fs_b:.5f}, p2p ratio {a_fs_b:.4f} — all vs fs 4096",
        "invariant: fs 4096 for the cylinder tiers, 2048 for MRI/HD-EMG (the recording rate)",
        r_fs2048_dz195=r_fs, p2p_fs2048_dz195=a_fs, r_fs2048_dz098=r_fs_b, p2p_fs2048_dz098=a_fs_b),
    "upsample 1 / 2 / 4": opt(
        "; ".join(f"{k}: jaggedness ×1 {v[1]['jaggedness']:.4f} → ×2 {v[2]['jaggedness']:.4f} → ×4 "
                  f"{v[4]['jaggedness']:.4f}, r(×2, ×4) {v[2]['r_vs_x4']:.5f}, p2p(×2)/p2p(×4) {v[2]['p2p_vs_x4']:.4f}, "
                  f"{v[1]['seconds'] * 1e3:.0f}/{v[2]['seconds'] * 1e3:.0f}/{v[4]['seconds'] * 1e3:.0f} ms" for k, v in UP.items()),
        ("KEEP 2 — converged (≡ ×4 to r ≥ 0.999) in every regime; jaggedness ×1/×2 = "
         + ", ".join(f"{z:.2f}" for z in zig.values()) + ": with the monopole-fitted φ, ×1 is already converged, so 2 is a "
         "cheap guard (+10–50 % per SFAP), not load-bearing; the recorded 0.08→0.01 came from the pre-monopole PM FEM route. "
         "On RAW FEM φ upsampling does not remove the ripple (it propagates it into φ''): the monopole fit does")
        if ok_up else "unexpected: ×2 not converged", zigzag_x1_over_x2=zig, **{k: v for k, v in UP.items()}),
    "w 256 vs 512": opt(f"first 256 samples identical: r = {r_w:.6f}, p2p ratio {a_w:.6f}",
                        "w only sets the window length; 256 (62.5 ms at 4096, 125 ms at 2048) holds the far EOF",
                        r=r_w, p2p_ratio=a_w)}
fork(4, "Sampling (fs, dz, upsample, w)", "live", options4,
     dict(use="fs 4096 / dz = v/fs on the cylinder tiers, fs 2048 / arc dz ≈ 1 mm on MRI, upsample_factor = 2, w = 256",
          when_alternative="upsample 4 only as a convergence reference; fs is the recording rate, not a physics knob",
          rationale="the SFAP is discretisation-converged: fs and w leave it unchanged to < 2 %, and ×2 ≡ ×4 to "
                    "r ≥ 0.999 in every regime. Measured here, ×1 is already converged once φ is monopole-fitted "
                    "(jaggedness ×1/×2 ≈ 1 even at dz ≠ v·dt), so upsample 2 is kept as a cheap guard for rougher "
                    "φ (the recorded 0.08→0.01 was on the pre-monopole PM FEM route); 4× buys nothing"),
     dict(value=f"upsample_factor = {DEFAULT.upsample_factor}, fsamp = {DEFAULT.fsamp:g} (MRI regime; harness uses {FS:g}), w = {DEFAULT.w}",
          passed=DEFAULT.upsample_factor == 2 and DEFAULT.w == 256 and ok_up and ok_fs and ok_w),
     TH4, ["live on the analytical cylinder φ at 10 mm depth", "recorded: _results/validation/A.json A0.4a; engines/spatial.py SpatialConfig.upsample_factor"])

# =========================================================================== fork 5 — time base
C5 = {}
for ct in (False, True):
    cfg = recipe(fiber_window="boxcar", center_time=ct)
    ta, a = sfap(pa, DZ, L1, L2, POSZ, cfg)
    _, b = sfap(pa, DZ, L1 + 60.0, L2, POSZ, cfg)
    lobes = {}
    for dzE in (20.0, 40.0):                                # electrode dzE mm from the NMJ (long fibre, EOF far)
        tl, sl = sfap(shift_phi(pa, DZ, dzE - 20.0), DZ, 100.0, 100.0, POSZ, cfg)
        lobes[dzE] = float(tl[np.argmin(sl)])
    slope_v = 20.0 / (lobes[40.0] - lobes[20.0])
    C5[ct] = dict(eof_onset_ms=eof_onset(ta, a, b), eof_expected_ms=T_EOF, lobe_t_neg_ms=lobes,
                  lobe_minus_dz_over_v_ms={k: v - k / V for k, v in lobes.items()}, cv_from_lobes=slope_v,
                  t0_ms=float(ta[0]), t_of_nmj_fire_ms=0.0)
TH5 = dict(eof_tol_ms=0.5, lobe_tol_ms=1.5, cv_tol=0.1)
ok5 = {ct: abs(c["eof_onset_ms"] - T_EOF) <= TH5["eof_tol_ms"]
       and all(abs(v) <= TH5["lobe_tol_ms"] for v in c["lobe_minus_dz_over_v_ms"].values())
       and abs(c["cv_from_lobes"] - V) / V <= TH5["cv_tol"] for ct, c in C5.items()}
options5 = {}
for ct, c in C5.items():
    s = (f"EOF onset {c['eof_onset_ms']:.2f} ms (L/v = {T_EOF:.0f}); propagating lobe at Δz = 20/40 mm: "
         f"{c['lobe_t_neg_ms'][20.0]:.2f}/{c['lobe_t_neg_ms'][40.0]:.2f} ms (Δz/v = 5/10; residual "
         f"{c['lobe_minus_dz_over_v_ms'][20.0]:+.2f}/{c['lobe_minus_dz_over_v_ms'][40.0]:+.2f} ms), CV from the lobes "
         f"{c['cv_from_lobes']:.2f} m/s; window starts at {c['t0_ms']:.2f} ms")
    options5[f"center_time = {ct}"] = opt(
        s, ("USE — physical time, t = 0 at the NMJ discharge" if not ct else
            "only to overlay a window-centred Fourier output; every timing is shifted by −w/(2·fs)"),
        gate_passed=ok5[ct], **c)
fork(5, "Time base", "live", options5,
     dict(use="physical time: center_time = False, t_start_ms = −10 (t = 0 is the NMJ discharge)",
          when_alternative="center_time = True only to overlay a Fourier (window-centred) trace",
          rationale="in physical time the EOF lands at L/v and the propagating lobe walks at Δz/v — the two "
                    "timings HD-EMG decomposition and CV estimation rely on; a centred axis breaks both"),
     dict(value=f"center_time = {DEFAULT.center_time}, t_start_ms = {DEFAULT.t_start_ms}",
          passed=(not DEFAULT.center_time) and DEFAULT.t_start_ms == -10.0 and ok5[False] and not ok5[True]),
     TH5, ["live on the analytical cylinder φ at 10 mm depth (boxcar so the EOF is sharp)",
           "recorded: _results/validation/A.json A0.4b, A2.3; key_numbers_f1.json fig5_physical_time"])

# =========================================================================== fork 6 — amplitude constant
AMP = {}
for v_ in (2.0, 3.0, 4.0, 5.0):
    t_r, s_r = sfap_first_principles(POSZ, L1, L2, v=v_)
    t_e, s_e = sfap(phi0, DZ, L1, L2, POSZ, recipe(fiber_window="boxcar", denoise="none", v=v_))
    AMP[v_] = best_r(t_r, s_r, t_e, s_e)[2]
ratios = np.array(list(AMP.values()))
spread = float(ratios.max() / ratios.min() - 1)
TH6 = dict(spread_max=0.02, mean_tol=0.05)
ok6 = spread <= TH6["spread_max"] and abs(ratios.mean() - 1) <= TH6["mean_tol"]
src_spatial = inspect.getsource(compute_sfap_spatial)
has_div_v = "/ cfg.v" in src_spatial.replace(" ", "").replace("/cfg.v", "/ cfg.v") and \
    any(line.strip().startswith("sfap =") and "/ cfg.v" in line for line in src_spatial.splitlines())
fork(6, "Amplitude constant (no 1/v prefactor)", "live",
     {"current constant: sfap = (CSD @ φ)·dz·polarity": opt(
         "engine/oracle amplitude ratio at v = 2/3/4/5 m/s: " + ", ".join(f"{r:.4f}" for r in ratios)
         + f" (mean {ratios.mean():.4f}, spread {spread * 100:.2f} %)",
         "USE — flat in v, equal to the closed-form constant" if ok6 else "unexpected: ratio not flat / not 1",
         ratio_by_v={str(k): v for k, v in AMP.items()}, mean=float(ratios.mean()), spread=spread),
      "the 1/v trap: sfap = (CSD @ φ)·dz·polarity / v (before 2026-09-15)": opt(
         "would give ratio/v = " + ", ".join(f"{r / v_:.3f}" for v_, r in AMP.items())
         + " — exactly 1/v (recorded 0.497 / 0.332 / 0.249 / 0.199), i.e. every MUAP 4× too small at v = 4",
         "NEVER — the IAP is defined in space, so the line-source integral is CV-independent; "
         "Nandedkar–Stålberg's amplitude ∝ 1/CV is for an IAP fixed in time and would have to come from "
         "stretching the IAP, not a prefactor",
         ratio_over_v={str(k): v / k for k, v in AMP.items()})},
     dict(use="the corrected constant (CV-independent integral)",
          when_alternative="none; if a CV-dependent amplitude is ever wanted it must come from stretching the spatial IAP",
          rationale="the engine/oracle ratio is flat at ≈0.994 across v = 2–5; the old /v made it exactly 1/v"),
     dict(value="compute_sfap_spatial has no '/ cfg.v' on the sfap line" if not has_div_v else "compute_sfap_spatial divides by v",
          passed=ok6 and not has_div_v),
     TH6, ["live vs harness.sfap_first_principles at four CVs", "recorded: A.json A0.2; key_numbers_f1.json fig4_first_principles.c_*"])

# =========================================================================== fork 7 — volume conductor (recorded)
f1, f1_src = paper_json("key_numbers_f1")
a11, a11b, a13 = vrec("A", "A1.1"), vrec("A", "A1.1b"), vrec("A", "A1.3")
b5, b12 = vrec("B", "B5"), vrec("B", "B12")
tr = get(f1, "fig2_cylinder_leadfield", "transverse_r30", default={}) or {}
prof = {k: np.array(tr.get(k, [])) for k in ("fem_peak_profile_sigma5_source", "fem_peak_profile_sigma1_source", "analytical_peak_profile_same_theta")}
dev = {}
if all(len(v) for v in prof.values()):
    dev = {"sigma5": float(np.abs(prof["fem_peak_profile_sigma5_source"] - prof["analytical_peak_profile_same_theta"]).max()),
           "sigma1": float(np.abs(prof["fem_peak_profile_sigma1_source"] - prof["analytical_peak_profile_same_theta"]).max())}
from emgforge.mri import pipeline as mri_pipeline                                   # noqa: E402
ss_default = inspect.signature(mri_pipeline.solve_grid_leadfields).parameters["source_sigma"].default
try:
    ss_cyl = cyl_fem.SOURCE_SIGMA
except Exception:                                # pragma: no cover
    ss_cyl = None
exp_ = get(f1, "fig9_validation_features", "depth_power_law", "exponent", default={}) or {}
fork(7, "Volume conductor and electrode source width", "recorded",
     {"analytical 4-layer cylinder (Farina 2004)": opt(
         "the reference for every law: depth power-law exponent n = %s (mono) / %s (SD), R² > 0.99 (B5); "
         "transverse FWHM at r = 30: %s mm; anisotropy elongation 1.48 (B12)" % (
             f(exp_.get("pipeline_mono")), f(exp_.get("pipeline_SD")), f(tr.get("fwhm_arc_ana_mm"), 1)),
         "USE for laws, thresholds and oracles (closed form, no mesh)",
         depth_exponent=exp_, fwhm_arc_ana_mm=tr.get("fwhm_arc_ana_mm")),
      "FEM cylinder, σ_s = 5 mm Gaussian electrode source": opt(
         (f"φ(z) vs analytical r = 0.995–1.000 at five depths, FWHM within 20 % (A1.1); φ'' r 0.92–0.96 at the "
          f"best-matched disk (A1.1b); FEM depth exponent {f(exp_.get('fem_mono'))} vs {f(exp_.get('pipeline_mono'))} "
          f"analytical → decays ~30 % slower over 7–20 mm (A1.2 (i)); transverse FWHM {f(tr.get('fwhm_arc_fem_mm'), 1)} mm "
          f"vs {f(tr.get('fwhm_arc_ana_mm'), 1)} analytical; max |profile − analytical| = {f(dev.get('sigma5'))}"),
         "USE for anatomy (MRI); keep σ_s = 5 mm ONLY for consistency with the released datasets (D2/D3, "
         "key_numbers_pipeline args.source_sigma = 5.0)",
         a11=a11["measured"] if a11 else None, a11b=a11b["measured"] if a11b else None,
         fwhm_arc_fem_mm=tr.get("fwhm_arc_fem_mm"), max_profile_dev=dev.get("sigma5"), profile=tr.get("fem_peak_profile_sigma5_source")),
      "FEM cylinder, σ_s = 1 mm source": opt(
         (f"transverse peak profile tracks the analytical one: max |profile − analytical| = {f(dev.get('sigma1'))} "
          f"(vs {f(dev.get('sigma5'))} at 5 mm); the 30 % slower depth decay SURVIVES σ_s = 1 mm (A1.2: Fourier-route "
          f"spread 1.41× vs 1.31×) — it is a volume-conductor difference, not the source"),
         "RECOMMENDED for new solves (the 5 mm blob reaches through 2 mm skin + 3 mm fat into the muscle)",
         max_profile_dev=dev.get("sigma1"), profile=tr.get("fem_peak_profile_sigma1_source"))},
     dict(use="analytical cylinder for laws; FEM/MRI for anatomy; σ_s = 1 mm for NEW solves, 5 mm only to stay "
              "consistent with the released datasets",
          when_alternative="σ_s = 5 mm when a result must be comparable to D1–D3 / key_numbers_pipeline",
          rationale="FEM reproduces the analytical φ to r ≥ 0.995 but with a 30 % slower depth decay that no "
                    "preprocessing or source width removes (open); the σ_s = 1 mm source removes the transverse "
                    "over-smoothing of the 5 mm blob (25 vs 40 mm FWHM)"),
     dict(value=f"solve_grid_leadfields(source_sigma = {ss_default}); cyl_fem.SOURCE_SIGMA = {ss_cyl}",
          passed=True,
          note="deliberate: 5 mm = released-dataset compatibility; pass source_sigma=1.0 for new solves"),
     {}, [f1_src, "_results/validation/A.json A1.1, A1.1b, A1.2, A1.3; B.json B5, B12",
          "key_numbers_f1.json fig2_cylinder_leadfield.transverse_r30 (σ_s 5 vs 1 mm profiles), fig9_validation_features.depth_power_law"],
     notes=[("A1.1: " + a11["measured"]) if a11 else "", ("A1.3: " + a13["measured"]) if a13 else "",
            ("B5: " + b5["measured"]) if b5 else "", ("B12: " + b12["measured"]) if b12 else ""])

# =========================================================================== fork 8 — fibre bed (recorded)
sf, sf_src = paper_json("key_numbers_study_fibres")
S = get(sf, "summary", default={}) or {}
hb = get(sf, "design", "harmonic_bed", default={}) or {}
fl = get(sf, "fibre_level", "bed", default={}) or {}
ident = get(sf, "check_straight_equals_released_tensor", default={}) or {}
f2, f2_src = paper_json("key_numbers_f2")
frac_reach = hb.get("frac_streamlines_reaching_iz", get(f2, "fig6_mri_fibres", "harmonic_nmj", "frac_streamlines_reaching_iz"))
n_trunc = get(f2, "fig6_mri_fibres", "harmonic_nmj", "n_not_reaching_iz")
fork(8, "Fibre bed (straight Poisson vs harmonic streamlines)", "recorded",
     {"straight Poisson bed (4 fibres/mm², 637 fibres, 204 mm)": opt(
         f"= the released tensor bit-for-bit (max |Δ| = {f(ident.get('max_abs_diff_V'))} V, identical = {ident.get('identical')}); "
         f"containment {f(get(fl, 'straight', 'containment'))}; CV {f(get(S, 'cv_m_per_s', 'straight', 'median'), 2)} m/s "
         f"(set 4.0); onset fraction {f(get(S, 'onset_fraction', 'straight', 'median'))}; EOF fraction "
         f"{f(get(S, 'eof_fraction', 'straight', 'median'))}; duration {f(get(S, 'duration_ms', 'straight', 'median'), 1)} ms",
         "USE (default) — reproducible, contained, the released numbers",
         identical=ident, containment=get(fl, "straight", "containment"), cv=get(S, "cv_m_per_s", "straight"),
         onset_fraction=get(S, "onset_fraction", "straight"), eof_fraction=get(S, "eof_fraction", "straight")),
      "harmonic single-NMJ bed, full-span units (18/24)": opt(
         f"p2p harmonic/straight median {f(get(S, 'p2p_ratio_harm_over_straight', 'full_span', 'median'))} "
         f"(q25–q75 {f(get(S, 'p2p_ratio_harm_over_straight', 'full_span', 'q25'))}–{f(get(S, 'p2p_ratio_harm_over_straight', 'full_span', 'q75'))}); "
         f"r(zero lag) median {f(get(S, 'corr_zero_lag', 'full_span', 'median'))} (min {f(get(S, 'corr_zero_lag', 'full_span', 'min'))}); "
         f"CV {f(get(S, 'cv_m_per_s', 'harmonic_full_span', 'median'), 2)} m/s (ratio {f(get(S, 'cv_m_per_s', 'harm_over_straight_full_span', 'median'))}); "
         f"containment {f(get(fl, 'harmonic', 'containment'))}; duration {f(get(S, 'duration_ms', 'harmonic', 'median'), 1)} ms",
         "legitimate for anatomy (curved, contained fibres) — same MUAPs within ~5 %",
         p2p_ratio=get(S, "p2p_ratio_harm_over_straight", "full_span"), r=get(S, "corr_zero_lag", "full_span"),
         cv=get(S, "cv_m_per_s", "harmonic_full_span"), containment=get(fl, "harmonic", "containment")),
      "harmonic bed, units with truncated streamlines (6/24)": opt(
         f"{f((1 - frac_reach) * 100 if frac_reach else None, 1)} % of streamlines ({n_trunc}) never reach the IZ plane → NMJ at the "
         f"fibre end: p2p ratio median {f(get(S, 'p2p_ratio_harm_over_straight', 'truncated', 'median'))} (max "
         f"{f(get(S, 'p2p_ratio_harm_over_straight', 'truncated', 'max'), 1)}), r median {f(get(S, 'corr_zero_lag', 'truncated', 'median'))}, "
         f"onset fraction {f(get(S, 'onset_fraction', 'harmonic_truncated', 'median'), 1)} (straight 0.18), CV unreadable "
         f"(0 of 6 with R² > 0.9)",
         "ARTEFACT — prune or re-innervate: the full-span variant restores r median "
         f"{f(get(S, 'corr_zero_lag', 'fullspan_variant_truncated_units', 'median'))}, onset {f(get(S, 'onset_fraction', 'harmonic_fullspan_variant', 'median'))}",
         frac_streamlines_reaching_iz=frac_reach, n_not_reaching_iz=n_trunc,
         p2p_ratio=get(S, "p2p_ratio_harm_over_straight", "truncated"), r=get(S, "corr_zero_lag", "truncated"),
         onset_fraction=get(S, "onset_fraction", "harmonic_truncated"), fullspan_variant=dict(
             r=get(S, "corr_zero_lag", "fullspan_variant_truncated_units"), onset=get(S, "onset_fraction", "harmonic_fullspan_variant"))),
      "series-fibered sampling": opt("no recorded numbers on this branch (C-04 fibre-length work)", "EXPERIMENTAL")},
     dict(use="straight Poisson bed (run_pipeline.py / mri.pipeline.poisson_bed)",
          when_alternative="harmonic streamlines for anatomy, with the ~16 % truncated streamlines pruned or re-innervated "
                           "(the study's full-span variant); series-fibering is experimental",
          rationale="on full-span units the two beds give the same MUAPs (p2p ratio 1.01, r 0.98, CV within 3 %); "
                    "the harmonic bed's only large differences come from streamlines that never reach the IZ"),
     dict(value="run_pipeline.py stage 4 = poisson_bed (straight, density 4/mm²)",
          passed=hasattr(mri_pipeline, "poisson_bed") and bool(ident.get("identical", False))),
     {}, [sf_src, f2_src + " fig6_mri_fibres.harmonic_nmj", "scripts/paper (study_fibres) on branch paper/arxiv"])

# =========================================================================== fork 9 — electrode grid (live + recorded)
leg_dir = ROOT / "_results/mu_pool/electrode_grid"
LEG = {}
try:
    g = np.load(leg_dir / "_phigrid_M5_dt15_z0.30-0.70_N637.npz", allow_pickle=True)
    e = np.asarray(g["elec_xyz"], float)
    flat = e.reshape(-1, 3)
    uniq = np.unique(np.round(flat, 3), axis=0)
    along = np.linalg.norm(np.diff(e, axis=0), axis=2); across = np.linalg.norm(np.diff(e, axis=1), axis=2)
    P, _ = paper_json("key_numbers_pipeline")
    cen = get(P, "default_n_mu_20", "stages", default=None)
    fcu_xy = None
    if cen:
        st = next((s for s in cen if s.get("key") == "bed"), None)
        fcu_xy = get(st or {}, "sizes", "centroid_xy")
    fcu_xy = np.array(fcu_xy) if fcu_xy else np.array([79.235, 99.493])
    LEG = dict(n_electrodes=int(flat.shape[0]), n_unique=int(uniq.shape[0]), n_duplicated=int(flat.shape[0] - uniq.shape[0]),
               ied_along_median=float(np.median(along)), ied_along_range=[float(along.min()), float(along.max())],
               ied_across_median=float(np.median(across)), ied_across_range=[float(across.min()), float(across.max())],
               grid_centre_xy=[float(flat[:, 0].mean()), float(flat[:, 1].mean())],
               centre_offset_from_fcu_centroid_mm=float(np.linalg.norm(flat[:, :2].mean(0) - fcu_xy)),
               fcu_centroid_xy=fcu_xy.tolist())
    pn = np.load(ROOT / "_results/mu_pool/spatial/mu_pool.npz", allow_pickle=True)
    LEG["single_channel_bank"] = dict(p2p_median_uV=float(np.median(pn["p2p"]) * 1e6), duration_median_ms=float(np.median(pn["duration_ms"])), n_mu=int(len(pn["p2p"])))
    gt = np.load(leg_dir / "muap_tensor_L8_M5.npz", allow_pickle=True)
    LEG["grid_tensor"] = dict(n_mu=int(gt["W"].shape[0]), p2p_max_over_grid_median_uV=float(np.median(np.ptp(gt["W"], axis=-1).max(1)) * 1e6))
except Exception as ex:                                  # pragma: no cover
    LEG = dict(error=str(ex))
f7 = get(f2, "fig7_mu_pool_muaps", default={}) or {}
gridf2 = get(f2, "fig6_mri_fibres", "grid", default={}) or {}
reg_centre = get(f7, "electrode_xyz")
REG = dict(ied_along_mm=gridf2.get("ied_along_mm"), ied_across_mm=gridf2.get("ied_across_mm"),
           ied_along_range=gridf2.get("ied_along_range"), ied_across_range=gridf2.get("ied_across_range"),
           p2p_median_uV=get(f7, "muap_p2p_uV", "median"), p2p_range_uV=[get(f7, "muap_p2p_uV", "min"), get(f7, "muap_p2p_uV", "max")],
           duration_median_ms=get(f7, "muap_duration_ms", "median"), centre_xyz=reg_centre,
           centre_offset_from_fcu_centroid_mm=(float(np.linalg.norm(np.array(reg_centre[:2]) - np.array(LEG.get("fcu_centroid_xy", [79.235, 99.493])))) if reg_centre else None))
from emgforge.simulator import Simulator                                            # noqa: E402
from_mri_src = inspect.getsource(Simulator.from_mri)
from_mri_prefers_pipeline = "pipeline_output.npz" in from_mri_src
from_mri_warns_on_legacy = ("electrode_grid" in from_mri_src) and ("warnings.warn" in from_mri_src)
# the legacy tensor may remain reachable, but only as a warned last resort behind the pipeline outputs
from_mri_legacy = not (from_mri_prefers_pipeline and from_mri_warns_on_legacy)
TH9 = dict(ied_tol_mm=1.0, duplicates_allowed=0)
ok_reg = (REG["ied_along_mm"] is not None and abs(REG["ied_along_mm"] - 10) <= TH9["ied_tol_mm"]
          and abs(REG["ied_across_mm"] - 10) <= TH9["ied_tol_mm"])
ok_leg = ("n_duplicated" in LEG and LEG["n_duplicated"] <= TH9["duplicates_allowed"]
          and abs(LEG["ied_along_median"] - 10) <= TH9["ied_tol_mm"] and abs(LEG["ied_across_median"] - 10) <= TH9["ied_tol_mm"])
fork(9, "Electrode grid", "live+recorded",
     {"legacy vertex-snapped grid (_results/mu_pool/electrode_grid, Simulator.from_mri)": opt(
         (f"{LEG.get('n_unique')} unique positions of {LEG.get('n_electrodes')} ({LEG.get('n_duplicated')} duplicated); "
          f"IED along {f(LEG.get('ied_along_median'), 1)} mm (range {rng(LEG.get('ied_along_range'))}), "
          f"across {f(LEG.get('ied_across_median'), 1)} mm (range {rng(LEG.get('ied_across_range'))}); "
          f"grid centre {f(LEG.get('centre_offset_from_fcu_centroid_mm'), 1)} mm from the FCU centroid (xy); "
          f"MUAPs: single-channel bank median {f(get(LEG, 'single_channel_bank', 'p2p_median_uV'), 2)} µV / "
          f"{f(get(LEG, 'single_channel_bank', 'duration_median_ms'), 1)} ms; grid tensor ({get(LEG, 'grid_tensor', 'n_mu')} MUs) "
          f"median {f(get(LEG, 'grid_tensor', 'p2p_max_over_grid_median_uV'), 2)} µV") if "error" not in LEG else f"cache unreadable: {LEG['error']}",
         "DEPRECATED — irregular spacing, duplicated electrodes, off the muscle; the source of the 0.6 µV / 33 ms scale",
         **LEG, gate_passed=ok_leg),
      "regular ray-cast grid (mri.pipeline.grid_electrodes / run_pipeline.py / f2_common)": opt(
         (f"5×5, IED along {f(REG['ied_along_mm'], 2)} mm (range {rng(REG.get('ied_along_range'), 2)}), "
          f"across {f(REG['ied_across_mm'], 2)} mm (range {rng(REG.get('ied_across_range'), 2)}); "
          f"centre electrode {f(REG['centre_offset_from_fcu_centroid_mm'], 1)} mm from the FCU centroid (xy, on the skin above it); "
          f"MUAP p2p median {f(REG['p2p_median_uV'], 1)} µV (range {f(REG['p2p_range_uV'][0], 2)}–{f(REG['p2p_range_uV'][1], 1)}), "
          f"duration median {f(REG['duration_median_ms'], 1)} ms; 100 MUs"),
         "USE" if ok_reg else "unexpected: IED off",
         **REG, gate_passed=ok_reg)},
     dict(use="the regular ray-cast grid built by emgforge.mri.pipeline.grid_electrodes (run_pipeline.py)",
          when_alternative="none — the legacy cache is kept only so old scripts still load; set EMGFORGE_MUAP_BANK to D2 for the chain checks",
          rationale="a grid snapped to mesh vertices had duplicated columns and irregular IED and sat off the muscle, "
                    "which is where the physiologically wrong 0.6 µV / 33 ms MUAPs came from; the regular grid gives "
                    "10.1 × 10.0 mm spacing and 24 µV / 18 ms MUAPs (S: 'MUAP physiological scale')"),
     dict(value=("run_pipeline.py → grid_electrodes (regular); Simulator.from_mri → legacy cache "
                 + ("(still the silent default of that constructor)" if from_mri_legacy else "(only as a warned fallback behind run_pipeline.py outputs)")),
          passed=ok_reg and not from_mri_legacy,
          note="Simulator.from_mri still loads the legacy vertex-snapped tensor silently — deprecate; use Simulator.from_pipeline" if from_mri_legacy else "Simulator.from_mri prefers the largest run_pipeline.py output; the legacy tensor loads only with a UserWarning"),
     TH9, ["live: _results/mu_pool/electrode_grid/_phigrid_M5_dt15_z0.30-0.70_N637.npz (elec_xyz), muap_tensor_L8_M5.npz, "
           "_results/mu_pool/spatial/mu_pool.npz; scripts/validation/muap_bank.py docstring",
           f2_src + " fig6_mri_fibres.grid, fig7_mu_pool_muaps", "src/emgforge/simulator.py Simulator.from_mri"])

# =========================================================================== fork 10 — entry point (recorded)
P, P_src = paper_json("key_numbers_pipeline")
def stage(sec, key):
    return next((s for s in get(P, sec, "stages", default=[]) or [] if s.get("key") == key), {})
ENT = {}
for sec in ("default_n_mu_20", "n_mu_100"):
    ENT[sec] = dict(total_wall_s=get(P, sec, "totals", "command_wall_s"), workers=get(P, sec, "args", "workers"),
                    n_mu=get(P, sec, "args", "n_mu"), mesh_s=stage(sec, "mesh").get("wall_s"),
                    leadfields_s=stage(sec, "leadfields").get("wall_s"), condition_s=stage(sec, "condition").get("wall_s"),
                    muaps_s=stage(sec, "muaps").get("wall_s"), activation_s=stage(sec, "activation").get("wall_s"),
                    verify=get(stage(sec, "muaps"), "notes", "recipe_check"),
                    cpu_s=(get(P, sec, "totals", "cpu_s_conditioning") or 0) + (get(P, sec, "totals", "cpu_s_muaps") or 0),
                    git_commit=get(P, sec, "environment", "git_commit"))
d23 = get(f2, "datasets_d2_d3", "runtimes", default={}) or {}
verify_ok = all(get(v, "verify", "identical") is True for v in ENT.values())
fork(10, "Entry point", "recorded",
     {"scripts/run_pipeline.py (cold) + Simulator.from_pipeline": opt(
         "; ".join(f"{v['n_mu']} MUs: {f(v['total_wall_s'], 0)} s wall with {v['workers']} workers (mesh {f(v['mesh_s'], 0)}, "
                   f"lead fields {f(v['leadfields_s'], 0)}, conditioning {f(v['condition_s'], 0)}, MUAPs {f(v['muaps_s'], 0)}, "
                   f"activation {f(v['activation_s'], 1)} s; {f(v['cpu_s'], 0)} CPU-s), verify identical = {get(v, 'verify', 'identical')} "
                   f"(max |Δ| {f(get(v, 'verify', 'max_abs_diff_V'))} V)" for v in ENT.values()),
         "USE — cold, reproducible, every stage timed and the factored conditioning verified bit-identical to the full recipe",
         **ENT),
      "paper f2_common tensor build (per-unit refits, 100 MUs)": opt(
         f"tensor wall {f(d23.get('tensor_wall_s'), 0)} s, {f(d23.get('tensor_cpu_s'), 0)} CPU-s (no conditioning factoring: each shared "
         f"fibre refitted per unit); same recipe, same grid → the released D2/D3 numbers",
         "legitimate for reproducing the paper figures only", **d23),
      "Simulator.from_mri (cached legacy tensor)": opt(
         f"instant load of _results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz — {get(LEG, 'grid_tensor', 'n_mu')} MUs on the "
         f"legacy vertex-snapped grid (fork 9)",
         "only for the released legacy tensor; deprecated for new work")},
     dict(use="python scripts/run_pipeline.py (cold, reproducible) → Simulator.from_pipeline(npz)",
          when_alternative="from_mri only to load the released legacy tensor; f2_common only to regenerate the paper figures",
          rationale="the runner's verify step shows the factored conditioning is bit-identical to the full recipe, "
                    "so the one-command cold run IS the recipe, with timings"),
     dict(value="scripts/run_pipeline.py present; Simulator.from_pipeline present; verify identical in both recorded runs",
          passed=(ROOT / "scripts/run_pipeline.py").exists() and hasattr(Simulator, "from_pipeline") and verify_ok),
     {}, [P_src + " default_n_mu_20 / n_mu_100 (stages, totals, muaps.notes.recipe_check)", f2_src + " datasets_d2_d3.runtimes"])

# =========================================================================== fork 11 — activation defaults (recorded)
c1, c2, c3, c6 = vrec("C", "C1"), vrec("C", "C2"), vrec("C", "C3"), vrec("C", "C6")
s_on, s_force = vrec("S", "Onion"), vrec("S", "EMG ↔ force")
fork(11, "Activation defaults (pool, rate coding, twitch/force)", "recorded",
     {"onion-skin rate coding (5–40 pps)": opt(
         (c1["measured"] if c1 else "n/a") + ("; sanity: " + s_on["measured"] if s_on else ""),
         "USE (passes)", passed=c1["passed"] if c1 else None),
      "ISI variability: Gaussian renewal, CoV 0.167": opt(
         c2["measured"] if c2 else "n/a",
         "CoV fine; OPEN ITEM — no refractory floor (1.58 % of ISIs < 20 ms, min 9.8 ms): clip the ISI or draw gamma/lognormal",
         passed=c2["passed"] if c2 else None),
      "recruitment thresholds / twitch range": opt(c3["measured"] if c3 else "n/a", "USE (passes)", passed=c3["passed"] if c3 else None),
      "force calibration (twitch pool)": opt(
         (s_force["measured"] if s_force else "n/a") + ("; C6: " + c6["measured"] if c6 else ""),
         "USE — ≈100 %MVC at full drive, monotone", passed=s_force["passed"] if s_force else None)},
     dict(use="the shipped activation defaults (MotoneuronPool / TwitchPool)",
          when_alternative="none needed; when sub-20 ms intervals matter, clip the ISI draw (open item C2)",
          rationale="onion skin, threshold skew, twitch range and MVC calibration all pass their literature checks; "
                    "the one failure is the missing refractory floor of the Gaussian renewal ISI"),
     dict(value="defaults unchanged", passed=all(r and r["passed"] for r in (c1, c3, s_on, s_force)),
          note="open item: C2 refractory floor (FAIL in C.json, not a known-flag)"),
     {}, ["_results/validation/C.json C1–C3, C6; S.json (simulator sanity)"])

# =========================================================================== write
runtime = time.time() - T0
n_fail = sum(1 for r in FORKS if not r["default"]["passed"])
rep = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), command="python scripts/sanity/route_audit.py",
           runtime_s=runtime, geometry=dict(depth_below_skin_mm=40.0 - DEPTH_R, oracle_rho_mm=10.0, posz_mm=POSZ,
                                            len1_mm=L1, len2_mm=L2, v=V, fs=FS, w=W, dz_mm=DZ),
           recipe_guard=RECIPE_GUARD, n_forks=len(FORKS), n_default_fail=n_fail, forks=FORKS)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


(OUT / "ROUTE_AUDIT.json").write_text(json.dumps(rep, indent=1, default=_json_default, ensure_ascii=False))

L = []
L.append("# Route audit — which pipeline to use, measured\n")
L.append(f"Generated {rep['generated']} by `{rep['command']}` in {runtime:.0f} s. "
         f"Live forks run on the analytical cylinder (fibre {40 - DEPTH_R:.0f} mm below the skin, NMJ {abs(POSZ):.0f} mm "
         f"proximal of the electrode, tendons at {POSZ - L1:+.0f}/{POSZ + L2:+.0f} mm from the electrode, v = {V:g} m/s, "
         f"fs = {FS:g} Hz, dz = {DZ:.3f} mm) and on the closed-form line-source oracle (electrode 10 mm from the fibre). "
         f"Recorded forks quote `_results/validation/*.json` and the paper key numbers (`git show paper/arxiv:paper/figures/…`). "
         f"The recipe under test is `{PC_SRC}()`.\n")
L.append("## Summary\n")
def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


L.append("| # | fork | use | current default | status |\n|---|---|---|---|---|")
for r in FORKS:
    L.append(f"| {r['fork']} | {cell(r['name'])} | {cell(r['verdict']['use'])} | {cell(r['default']['value'])} | "
             f"{'PASS' if r['default']['passed'] else 'FAIL'}{cell(' — ' + r['default']['note']) if r['default'].get('note') else ''} |")
L.append("")
L.append("## Recipe guard\n")
L.append(f"`{PC_SRC}()` vs the documented recipe (DIRECT_LINE_SOURCE.md §0): "
         + ("**no mismatch**" if not mismatch else "**MISMATCH** " + ", ".join(f"{k}: shipped {a!r} vs documented {b!r}" for k, (a, b) in mismatch.items()))
         + f". `field_to_muap(field, bed)` with no config → **{engine_none}**.\n")
for r in FORKS:
    L.append(f"## Fork {r['fork']} — {r['name']}  *({r['kind']})*\n")
    L.append(f"**Use:** {r['verdict']['use']}.  \n**Alternative is legitimate when:** {r['verdict']['when_alternative']}.  \n"
             f"**Why:** {r['verdict']['rationale']}.\n")
    L.append("| option | key numbers | verdict |\n|---|---|---|")
    for name, o in r["options"].items():
        L.append(f"| {cell(name)} | {cell(o['summary'])} | {cell(o['verdict'])} |")
    L.append("")
    L.append(f"**Current default:** {r['default']['value']} → **{'PASS' if r['default']['passed'] else 'FAIL'}**"
             + (f" ({r['default']['note']})" if r["default"].get("note") else "") + "  ")
    if r["thresholds"]:
        L.append("**Thresholds:** " + ", ".join(f"{k} = {v}" for k, v in r["thresholds"].items()) + "  ")
    L.append("**Evidence:** " + "; ".join(p for p in r["provenance"] if p) + "\n")
    for n in [n for n in r["notes"] if n]:
        L.append(f"> {n}\n")
md = "\n".join(L)
(OUT / "ROUTE_AUDIT.md").write_text(md)
DOC_COPY.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(OUT / "ROUTE_AUDIT.md", DOC_COPY)

print("\n" + "=" * 78)
print(f"  {len(FORKS) - n_fail}/{len(FORKS)} defaults match their verdict"
      + (f"  ·  {n_fail} FAIL" if n_fail else "") + f"  ·  {runtime:.0f} s")
print("=" * 78)
print("wrote", OUT / "ROUTE_AUDIT.md", "|", OUT / "ROUTE_AUDIT.json", "| copy →", DOC_COPY)
sys.exit(1 if n_fail else 0)
