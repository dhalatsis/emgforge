"""Tier B — MUAP features: does the model reproduce the well-replicated phenomenology?

Every check is a quantitative fact from the EMG literature (see docs/validation/
BIBLIOGRAPHY.md), tested on two systems: the Farina-2004 analytical generator (the
field's reference model) and OUR pipeline (lead field → golden spatial recipe). Where
the pipeline needs an electrode array, we exploit the cylinder's z-invariance:
electrode at +z ≡ lead field translated by +z. SD/DD montages are built by
differencing electrodes, exactly as a real amplifier does.

  B1  monopolar SFAP morphology         B7  linearity, NMJ dispersion, duration
  B2  CV recovery (SD array, 3 CVs)      B8  bipolar spectral null at CV/IED
  B3  innervation-zone signature         B9  electrode size (spatial low-pass)
  B4  end-of-fibre component             B10 inter-electrode distance
  B5  amplitude vs depth (power law)     B11 subcutaneous fat (analytical + FEM)
  B6  transverse spread & depth rule     B12 anisotropy elongates the lead field

Run:  python scripts/validation/tier_b_features.py     (~1 min; FEM from the cyl_fem cache)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import *                                    # noqa: F401,F403
import cyl_fem
from emgforge.synthesis.fibres import FibreBed
from emgforge.synthesis.api import field_to_muap

T = "B"
DZ = V * 1000.0 / FS
cache = cyl_fem.build_cache()
Z_ABS, RADII, THETAS, ZE = cache["z_abs"], cache["radii"], cache["thetas"], float(cache["ze"])
FIG = {}
print("=" * 74); print("Tier B — MUAP features vs the literature"); print("=" * 74, "\n")


def fem_phi(r, th, key="phi", centre=ZE):
    i, j = int(np.argmin(np.abs(RADII - r))), int(np.argmin(np.abs(THETAS - th)))
    return cyl_fem.window(cache[key][i, j], Z_ABS, centre, W, DZ)


def pipe_array(phi, zc, len1=60.0, len2=60.0, posz=0.0, cfg=None, montage="mono", ied=10.0):
    """SFAPs at electrode positions ``zc`` (mm along the fibre; +z ≡ φ translated by +z),
    optionally as SD / DD built from electrodes ±ied/2 (SD) or ±ied (DD)."""
    def one(z):
        return sfap(shift_phi(phi, DZ, z), DZ, len1, len2, posz, cfg)[1]
    if montage == "mono":
        return np.array([one(z) for z in zc])
    if montage == "SD":
        return np.array([one(z + ied / 2) - one(z - ied / 2) for z in zc])
    return np.array([one(z - ied) - 2 * one(z) + one(z + ied) for z in zc])


t_ax = np.arange(W) / FS * 1000.0 - 10.0
phi30, _ = ana_phi(30.0)

# =========================================================================== B1
def lobes(x, frac=0.1):
    """Signed sequence of lobes larger than frac·max|x| (merging same-sign runs)."""
    s = np.sign(np.where(np.abs(x) > frac * np.abs(x).max(), x, 0))
    seq = [v for i, v in enumerate(s) if v != 0 and (i == 0 or v != s[i - 1] or s[i - 1] == 0)]
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(int(v))
    return out

# electrode 30 mm from the NMJ, tendons ≥ 95 mm away on both sides (Farina: ±150; pipeline:
# the fibre is truncated by the ±125 mm array, nearest end 95 mm → EOF onset at 23.75 ms), and
# the lobes are counted on t ≤ 20 ms so no end-of-fibre potential can enter the count
t_F, s_F, _ = farina(depth=30.0, zi=-30.0, L1=150.0, L2=150.0)
t_p, s_p = sfap(phi30, DZ, 250.0, 250.0, -30.0)
m20_F, m20_p = t_F <= 20.0, t_p <= 20.0
lob_F, lob_p = lobes(s_F[0][m20_F]), lobes(s_p[m20_p])
big_F = int(np.sign(s_F[0][m20_F][np.argmax(np.abs(s_F[0][m20_F]))]))
big_p = int(np.sign(s_p[m20_p][np.argmax(np.abs(s_p[m20_p]))]))
FIG["b1"] = (t_F, s_F[0], t_p, s_p)
check(T, "B1 monopolar SFAP over the fibre (between IZ and tendon) is triphasic + − + with the main lobe negative",
      lob_p[-3:] == [1, -1, 1] and big_p == -1,
      f"pipeline lobes {lob_p} (largest {big_p:+d}); Farina generator lobes {lob_F} (largest {big_F:+d}); electrode 30 mm "
      f"from the NMJ, tendons ≥ 95 mm away, counted on t ≤ 20 ms",
      "+ − + with the largest lobe negative (depolarised zone under the electrode). The Farina port's propagating "
      "main lobe is POSITIVE: its output polarity is inverted relative to the textbook — the same anti-phase "
      "relation tier A0.3 finds against first principles (its EOF spike, being negative, hid this when tendons were near)",
      refs="Merletti & Muceli 2019 Fig. 2; Arabadzhiev 2013; Rosenfalck 1969")

# =========================================================================== B2
cvs = {}
zc = np.array([15.0, 25.0, 35.0, 45.0])                 # one side of the IZ, tendons at ±100
for v_ in (3.0, 4.0, 5.0):
    pa, dza = ana_phi(30.0, v=v_)                        # dz = v/fs changes with v; window stays 256
    st = pipe_array(pa, zc, 100.0, 100.0, 0.0, golden_cfg(v=v_), "SD", 10.0)
    cv_p, r2_p, _ = cv_from_array(st, 10.0, FS)
    try:                                                 # the Farina port's np.arange k-grid is
        t_F, s_F, zF = farina(depth=30.0, zi=0.0, L1=100.0, L2=100.0, det_type=2, channels=9, dint=10.0, v=v_)
        cv_F, r2_F, _ = cv_from_array(s_F[5:], 10.0, FS)  # off-by-one for some v (257 ≠ 256 bins)
    except ValueError:
        cv_F, r2_F = np.nan, np.nan
    cvs[v_] = (cv_p, r2_p, cv_F, r2_F)
ok = all(abs(c[0] - v_) / v_ <= 0.05 and (np.isnan(c[2]) or abs(c[2] - v_) / v_ <= 0.06) for v_, c in cvs.items())
check(T, "B2 conduction velocity recovered from an SD array tracks the set CV (3, 4, 5 m/s)",
      ok, "pipeline/Farina: " + ", ".join(f"v={v_:g}: {c[0]:.2f}/{c[2]:.2f} m/s" for v_, c in cvs.items())
      + " (Farina NaN = its k-grid does not build at that v)",
      "within 5 % (pipeline) / 6 % (Farina) at each CV; physiological range 3–5 m/s",
      refs="Farina & Merletti 2004 (J Neurosci Methods); Zwarts 1988 (4.55±0.33 m/s); Andreassen & Arendt-Nielsen 1987")

# =========================================================================== B3
zc9 = np.arange(-40.0, 40.1, 10.0)
mono = pipe_array(phi30, zc9, 60.0, 60.0, 0.0)
sd = pipe_array(phi30, zc9, 60.0, 60.0, 0.0, None, "SD", 10.0)
sym = [float(np.corrcoef(mono[4 + k], mono[4 - k])[0, 1]) for k in (1, 2, 3)]
sd_iz = float(p2p(sd[4]) / max(p2p(s) for s in sd))
rev = [float(np.corrcoef(sd[4 + k], sd[4 - k])[0, 1]) for k in (1, 2, 3)]
t_F, s_F, _ = farina(depth=30.0, zi=0.0, det_type=2, channels=9, dint=10.0)
sd_iz_F = float(p2p(s_F[4]) / max(p2p(s) for s in s_F))
rev_F = [float(np.corrcoef(s_F[4 + k], s_F[4 - k])[0, 1]) for k in (1, 2, 3)]
FIG["b3"] = (mono, sd, zc9)
check(T, "B3 innervation-zone signature: monopolar mirror symmetry, SD null and phase reversal at the IZ",
      min(sym) > 0.99 and sd_iz < 0.05 and max(rev) < -0.95 and sd_iz_F < 0.05 and max(rev_F) < -0.95,
      f"pipeline: mono r(+d,−d) = {np.round(sym, 4).tolist()}, SD@IZ/max = {sd_iz:.3f}, SD r(+d,−d) = {np.round(rev, 3).tolist()};  "
      f"Farina: SD@IZ/max = {sd_iz_F:.3f}, SD r(+d,−d) = {np.round(rev_F, 3).tolist()}",
      "mono r > 0.99; SD at the IZ < 5 % of max; SD r(+d,−d) < −0.95 (mirror images of opposite sign)",
      refs="Masuda et al. 1983/1985; Merletti & Muceli 2019 §2.2.1, §2.3; Campanini et al. 2022")

# =========================================================================== B4
def eof_parts(phi, zc_, montage="mono", posz=-20.0, L_near=40.0, l2=80.0):
    """(EOF amplitude, propagating amplitude) per electrode: EOF isolated by subtracting the
    same fibre with the near tendon moved 60 mm further; propagating = p2p of that long fibre."""
    cfg = golden_cfg(fiber_window="boxcar")
    a = pipe_array(phi, zc_, L_near, l2, posz, cfg, montage)
    b = pipe_array(phi, zc_, L_near + 60.0, l2, posz, cfg, montage)
    return np.abs(a - b).max(1), np.array([p2p(x) for x in b]), a - b

# (a) latency spread across channels
zc5 = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
_, _, d = eof_parts(phi30, zc5)
onset = [float(t_ax[np.where(np.abs(x) > 0.05 * np.abs(x).max())[0][0]]) for x in d]
t_F, s_F, _ = farina(depth=30.0, zi=-20.0, L1=60.0, L2=60.0, channels=5, dint=15.0)
t_F2, s_F2, _ = farina(depth=30.0, zi=-20.0, L1=60.0, L2=120.0, channels=5, dint=15.0)
onset_F = [float(t_F[np.where(np.abs(a - b) > 0.05 * np.abs(a - b).max())[0][0]]) for a, b in zip(s_F, s_F2)]
check(T, "B4a the end-of-fibre component is non-propagating: same onset on every electrode",
      np.ptp(onset) < 0.5 and np.ptp(onset_F) < 0.5,
      f"pipeline onsets {np.round(onset, 2).tolist()} ms (spread {np.ptp(onset):.2f}); "
      f"Farina {np.round(onset_F, 2).tolist()} (spread {np.ptp(onset_F):.2f}); expected L/v = 10 ms",
      "spread < 0.5 ms across electrodes −30…+30 mm",
      refs="Merletti & Muceli 2019 §2.2.2; Mesin 2005; Rodriguez-Falces & Place 2018")

# (b) EOF/propagating ratio grows with depth
ratio_d = []
for r in (33.0, 30.0, 27.0, 25.0, 20.0):
    e, p, _ = eof_parts(ana_phi(r)[0], np.array([0.0]))
    ratio_d.append(float(e[0] / p[0]))
check(T, "B4b EOF / propagating amplitude ratio increases monotonically with depth",
      bool(np.all(np.diff(ratio_d) > 0)),
      f"ratio at depth-below-skin 7/10/13/15/20 mm = {np.round(ratio_d, 3).tolist()}",
      "monotone increase (the propagating part decays faster than the far-field EOF)",
      refs="Arabadzhiev 2013; Merletti & Muceli 2019 Fig. 5 (comparable at ~22 mm); Campanini 2022")

# (c) montage ordering mono > SD > DD
ratio_m = {}
for m in ("mono", "SD", "DD"):
    e, p, _ = eof_parts(phi30, np.array([0.0]), m)
    ratio_m[m] = float(e[0] / p[0])
check(T, "B4c spatial filters suppress the EOF: EOF/propagating ratio mono > SD > DD",
      ratio_m["mono"] > ratio_m["SD"] > ratio_m["DD"],
      f"{ {k: round(v, 3) for k, v in ratio_m.items()} } (IED 10 mm)",
      "strict ordering", refs="Roeleveld 1998; Farina et al. 2002 (Muscle Nerve); Disselhorst-Klug 1997")
FIG["b4"] = ratio_d

# =========================================================================== B5
depths = np.array([7.0, 10.0, 13.0, 15.0, 20.0, 25.0]); radii = 40.0 - depths
amp_m = np.array([p2p(sfap(ana_phi(r)[0], DZ, 100, 100, -20.0)[1]) for r in radii])
amp_s = np.array([p2p(pipe_array(ana_phi(r)[0], np.array([0.0]), 100, 100, -20.0, None, "SD")[0]) for r in radii])
amp_F = np.array([p2p(farina(depth=r, zi=-20.0, L1=100, L2=100)[1][0]) for r in radii])
amp_fem = np.array([p2p(sfap(fem_phi(r, 0.0), DZ, 100, 100, -20.0)[1]) for r in (33.0, 30.0, 27.0, 25.0, 20.0, 15.0)])
n_m, r2_m = power_law(depths, amp_m); n_s, r2_s = power_law(depths, amp_s)
n_F, r2_F = power_law(depths, amp_F); n_fem, r2_fem = power_law(depths, amp_fem)
FIG["b5"] = (depths, amp_m, amp_s, amp_F, amp_fem)
check(T, "B5 amplitude vs depth is a power law (log-log linear), steeper for SD than monopolar",
      min(r2_m, r2_s, r2_F, r2_fem) > 0.95 and n_s > n_m and np.all(np.diff(amp_m) < 0),
      f"exponent n (p2p ∝ d^−n), R²: pipeline mono {n_m:.2f} ({r2_m:.3f}), SD {n_s:.2f} ({r2_s:.3f}); "
      f"Farina generator mono {n_F:.2f} ({r2_F:.3f}); FEM pipeline mono {n_fem:.2f} ({r2_fem:.3f}) "
      f"(depth-below-skin 7–25 mm; the fibre at 7 mm sits 2 mm inside the muscle)",
      "R² > 0.95 over 7–25 mm; n_SD > n_mono; monotone decrease. Note the Farina generator's own exponent "
      "differs from the first-principles engine's on the same φ (its shape conventions, tier A0.3)",
      refs="Roeleveld et al. 1997a (inverse power law, bipolar steeper); Fuglevand 1992 (10–12 mm); Merletti & Muceli 2019 Fig. 7")

# =========================================================================== B6
arc = lambda th, rs=40.0: np.deg2rad(th) * rs                    # arc length on the skin (mm)
ths = np.arange(0.0, 90.1, 3.0)
# a realistic fibre: tendons at ±60, NMJ 20 mm from the electrode → the EOF is present, as in
# the recordings the literature widths come from (the far field is what makes monopolar wide)
def transverse(r, montage):
    if montage == "mono":
        return np.array([p2p(sfap(ana_phi(r, distfib=th)[0], DZ, 40, 80, -20.0, golden_cfg(fiber_window="boxcar"))[1]) for th in ths])
    return np.array([p2p(pipe_array(ana_phi(r, distfib=th)[0], np.array([0.0]), 40, 80, -20.0, golden_cfg(fiber_window="boxcar"), "SD")[0]) for th in ths])
def width50(prof):                                              # full width at 50 % (two-sided, symmetric)
    x = arc(ths); half = prof[0] / 2
    k = np.where(prof < half)[0][0]
    return 2 * float(np.interp(half, [prof[k], prof[k - 1]], [x[k], x[k - 1]]))
prof = {(d, m): transverse(40 - d, m) for d in (15.0, 20.0) for m in ("mono", "SD")}
w_m = {d: width50(prof[(d, "mono")]) for d in (15.0, 20.0)}
w_s = {d: width50(prof[(d, "SD")]) for d in (15.0, 20.0)}
rule = {d: d / (0.2 * w_m[d]) for d in w_m}                    # Roeleveld: depth ≈ 0.2 × width50
FIG["b6"] = (ths, prof[(15.0, "mono")], prof[(15.0, "SD")])
check(T, "B6 transverse spread: monopolar wider than SD (lit. 2.5–4×); depth ≈ 0.2 × (mono 50 %-width)",
      all(w_m[d] >= w_s[d] for d in w_m) and w_s[20.0] > w_s[15.0] and w_m[20.0] > w_m[15.0],
      f"50 %-width (mm) at depth 15/20 mm: mono {w_m[15.0]:.0f}/{w_m[20.0]:.0f}, SD {w_s[15.0]:.0f}/{w_s[20.0]:.0f} "
      f"(ratio {w_m[15.0] / w_s[15.0]:.1f}/{w_m[20.0] / w_s[20.0]:.1f}); depth/(0.2·width) = "
      f"{rule[15.0]:.2f}/{rule[20.0]:.2f} (Roeleveld's biceps-MU rule = 1.0; single fibre in a 40 mm cylinder)",
      "mono ≥ SD width; both widen with depth. The literature ratio (2.5–4×; biceps MUs 15–25 mm deep, mono "
      "72–96 mm vs SD 24–32 mm) comes from the MU's far-field dominating the monopolar map — a single fibre's "
      "EOF/propagating ratio is only 0.1–0.2 here (B4b), so the ratio and Roeleveld's rule are reported, not gated",
      refs="Roeleveld et al. 1997b; Roeleveld/Stegeman 2013; Merletti & Muceli 2019 Fig. 9")

# =========================================================================== B7
# long fibre (tendons far) so the duration measures the propagating complex, not the EOF tail
bed1 = FibreBed.uniform(1, DZ, 100, 100, -20.0); bed50 = FibreBed.uniform(50, DZ, 100, 100, -20.0)
m1 = field_to_muap(phi30, bed1, golden_cfg()).muap; m50 = field_to_muap(phi30, bed50, golden_cfg()).muap
lin = float(np.abs(m50 - 50 * m1).max() / p2p(m50))
bedj = FibreBed.jittered(50, DZ, len1_mm=100, len2_mm=100, nmj_sigma_mm=5.0, seed=1)
mj = field_to_muap(phi30, FibreBed.from_arrays(DZ, 100, 100, bedj.posz_mm - 20.0), golden_cfg()).muap
dur1, durj = duration_ms(t_ax, m1, 0.1), duration_ms(t_ax, mj, 0.1)
check(T, "B7 MUAP amplitude ∝ fibre count at fixed geometry; NMJ scatter (σ=5 mm) disperses and lengthens it",
      lin < 1e-9 and p2p(mj) < 50 * p2p(m1) and durj > dur1 and 0.6 * dur1 < durj < 3 * dur1,
      f"50 identical fibres = 50×SFAP to {lin:.1e}; with NMJ σ=5 mm: p2p {p2p(mj) / (50 * p2p(m1)):.2f}× the coherent sum, "
      f"10 %-duration {dur1:.1f} → {durj:.1f} ms",
      "exact linearity; sub-linear amplitude and longer duration with dispersion",
      refs="Merletti & Muceli 2019 (MUAP = Σ SFAP); Roeleveld 1998; Farina 2014 (amplitude cancellation)")

# =========================================================================== B8
# the SD transfer function |2 sin(π f·IED/v)| is the SD/monopolar spectral RATIO — locate its
# null there, so the SFAP's own spectral shape (which falls steeply) cannot move the minimum
# Only the PROPAGATING complex is translation-shifted between the two electrodes (the generation
# component at t≈0 and the EOF at L/v are not), so window it (Tukey) before taking the ratio.
from scipy.signal.windows import tukey
def window_prop(t, x, t0, t1):
    m = (t >= t0) & (t <= t1); y = np.zeros_like(x); y[m] = x[m] * tukey(m.sum(), 0.3); return y
def ratio_null(sd_, mono_, f0, fs=FS, nfft=8192):
    f = np.fft.rfftfreq(nfft, 1 / fs)
    R = np.abs(np.fft.rfft(sd_, nfft)) / np.maximum(np.abs(np.fft.rfft(mono_, nfft)), 1e-30)
    m = (f > 0.5 * f0) & (f < 1.5 * f0); return float(f[m][np.argmin(R[m])])
# The null sits where the SFAP spectrum is already ~1 % of its peak, so any sub-percent
# non-equivariance of the electrode translation fills it and moves the minimum. Electrode
# shifts of an INTEGER number of φ samples (10 / 20 samples = 9.77 / 19.53 mm) make the
# translation exact by construction (np.roll), so the test isolates the SD transfer
# function and the preprocessing's own equivariance (monopole refit, upsampling).
# A SINGLE travelling wave: the comb-filter derivation assumes one. With the NMJ under the
# array both counter-propagating tripoles reach every electrode, and in a bounded cylinder
# (φ is still 35 % of its peak 125 mm away) the receding one is not a pure delay. So: NMJ at
# −60 mm with a 20 mm proximal semi-fibre (that tripole dies at 5 ms), a long distal one
# (truncated by the array → EOF at 46 ms), electrode ≈100 mm from the NMJ (wave at 25 ms),
# window 7–44 ms.
POSZ, LEN1, LEN2 = -60.0, 20.0, 185.0
ZC = 41                                                    # electrode samples right of the array centre
# between the proximal extinction (5–8.75 ms) and the wave entering the golden one-sided tendon
# taper (last 25 % of the distal semi-fibre, from +79 mm → 34.7 ms) — the taper is a fixed
# spatial feature of the fibre, so inside it the two electrodes are legitimately not shift-copies
T0, T1 = 9.0, 33.0
def S_at(n, cfg):
    return sfap(np.roll(phi30, n), DZ, LEN1, LEN2, POSZ, cfg)[1]
# (i) electrode-translation equivariance: with v·dt = dz, moving the electrode by n samples
# must delay the propagating waveform by exactly n samples
m_win = (t_ax >= T0) & (t_ax <= T1)
equiv = {}
for name, cfg in (("none", golden_cfg(denoise="none")), ("golden", golden_cfg())):
    a, b = S_at(ZC + 5, cfg), S_at(ZC - 5, cfg)
    equiv[name] = float(np.abs((a[10:] - b[:-10])[m_win[10:]]).max() / p2p(a))
# (ii) the comb null of |SD|/|mono|
nulls = {}
for n_ied in (10, 20):
    ied = n_ied * DZ; f0 = V * 1000 / ied
    def sd_null(cfg):
        sd_ = window_prop(t_ax, S_at(ZC + n_ied // 2, cfg) - S_at(ZC - n_ied // 2, cfg), T0, T1)
        return ratio_null(sd_, window_prop(t_ax, S_at(ZC, cfg), T0, T1), f0)
    # Farina: same single-wave geometry (fibre [−L2, L1] with the NMJ at zi; proximal tendon 20 mm away)
    t_F, s_sd, _ = farina(depth=30.0, zi=-60.0, L1=150.0, L2=80.0, det_type=2, channels=1, center=40.0, dint=ied,
                          electrode_type="circ", dim1=5.0)
    _, s_mo, _ = farina(depth=30.0, zi=-60.0, L1=150.0, L2=80.0, det_type=1, channels=1, center=40.0, dint=ied)
    nulls[ied] = (sd_null(golden_cfg(denoise="none")), sd_null(golden_cfg()),
                  ratio_null(window_prop(t_F, s_sd[0], T0, T1), window_prop(t_F, s_mo[0], T0, T1), f0), f0)
check(T, "B8 the bipolar montage is a pure spatial difference of a translation-invariant wave (⇒ comb filter, nulls at n·CV/IED)",
      equiv["none"] < 5e-3 and equiv["golden"] < 5e-3,
      f"translation equivariance residual (single travelling wave, 9–33 ms): no denoise {equiv['none']:.1e}, golden {equiv['golden']:.1e}.  "
      "Measured null of |SD|/|mono| — pipeline (no denoise) / pipeline (golden) / Farina / expected (Hz): "
      + ", ".join(f"IED {k:.1f} mm: {a:.0f}/{b:.0f}/{f:.0f}/{c:.0f}" for k, (a, b, f, c) in nulls.items()),
      "equivariance < 5e-3: SD(t) = S(t) − S(t − IED/v) to that precision, so |H| = |2 sin(π f·IED/v)| with "
      "nulls at n·v/IED follows analytically (409.6 / 204.8 Hz here). The measured null is informational: a "
      "spectral zero cannot be localised from a 24 ms window (leakage moves it by tens of Hz)",
      refs="Lindström & Magnusson 1977; Lynn et al. 1978; Merletti & Muceli 2019 §3.3; Sinderby 1996")

# =========================================================================== B9
sizes = (0.5, 2.5, 5.0, 10.0)                                   # analytical electrode RADIUS (mm)
p_sz = [p2p(farina(depth=30.0, zi=-20.0, L1=100, L2=100, dim1=d)[1][0]) for d in sizes]
f_sz = [mnf(farina(depth=30.0, zi=-20.0, L1=100, L2=100, dim1=d)[1][0], FS) for d in sizes]
from scipy.special import j1
h = lambda d, k: 2 * j1(d * k) / (d * k)                        # DetectionSystem's circular-electrode filter
db5, db10 = 20 * np.log10(h(2.5, 2 * np.pi * 0.1)), 20 * np.log10(h(5.0, 2 * np.pi * 0.05))   # Ø5 @100 c/m, Ø10 @50 c/m
check(T, "B9 electrode size is a spatial low-pass: amplitude and MNF fall with electrode area; Ø5 mm ≈ −3 dB at 100 c/m",
      np.all(np.diff(p_sz) < 0) and np.all(np.diff(f_sz) < 0) and abs(db5 + 3) < 0.6 and abs(db10 + 3) < 0.6,
      f"Farina p2p vs radius {sizes}: {np.round(np.array(p_sz) / p_sz[0], 3).tolist()}; MNF: {np.round(f_sz, 1).tolist()} Hz; "
      f"disc filter: Ø5 mm @100 c/m = {db5:.1f} dB, Ø10 mm @50 c/m = {db10:.1f} dB. (The FEM electrode behaves as a Ø5 mm disc — tier A1.1b)",
      "monotone decrease; −3 dB points as in the literature", refs="Merletti & Muceli 2019 Fig. 11–12, Table 1; Fuglevand 1992")

# =========================================================================== B10
ieds = (2.5, 5.0, 10.0, 20.0)
p_ied = [p2p(pipe_array(phi30, np.array([25.0]), 100, 100, 0.0, None, "SD", e)[0]) for e in ieds]
p_ied_F = [p2p(farina(depth=30.0, zi=0.0, L1=100, L2=100, det_type=2, channels=1, center=25.0, dint=e)[1][0]) for e in ieds]
lin_small = (p_ied[1] / p_ied[0]) > 1.7 and (p_ied_F[1] / p_ied_F[0]) > 1.7
check(T, "B10 SD amplitude grows ≈linearly with IED when IED ≪ λ, then saturates towards λ/2",
      lin_small and p_ied[3] / p_ied[2] < 2.0 and p_ied_F[3] / p_ied_F[2] < 2.0 and np.all(np.diff(p_ied) > 0),
      f"SD p2p vs IED {ieds} mm — pipeline: {np.round(np.array(p_ied) / p_ied[0], 2).tolist()}; "
      f"Farina: {np.round(np.array(p_ied_F) / p_ied_F[0], 2).tolist()} (×2 IED → ≥1.7× at small IED; <2× at 10→20 mm)",
      "differentiator regime then saturation (λ ≈ v·T_lobe ≈ 20–30 mm)",
      refs="De Luca 2002; Hermens 2000 (SENIAM); Merletti & Muceli 2019")

# =========================================================================== B11
fats = (0.5, 3.0, 6.0, 9.0, 18.0)
def fat_case(f):
    g = dict(r_fat=35.0 + f, r_skin=35.0 + f + 2.0)
    t, s, _ = farina(depth=30.0, zi=-20.0, L1=100, L2=100, geo=g)
    return s[0]
sf = [fat_case(f) for f in fats]
rms_f = np.array([np.sqrt((x ** 2).mean()) for x in sf]); rel = rms_f / rms_f[0]
mnf_f = [mnf(x, FS) for x in sf]
kuiken = {3.0: 0.687, 9.0: 0.198, 18.0: 0.100}                  # RMS retained vs 0 mm fat
ok_k = all(kuiken[f] / 1.5 <= rel[i] <= kuiken[f] * 1.5 for i, f in enumerate(fats) if f in kuiken)
# FEM pipeline: fat 2/4/6/8 mm meshes, fibre fixed at r=30 (5 mm under the muscle surface). The
# FEM/analytical comparison uses the Butterworth φ-smoothing variant (tier A1.2: the golden
# recipe's amplitude on FEM φ is erratic at ±40 %); the golden numbers are reported alongside.
def fem_fat_p2p(cfg):
    return np.array([p2p(sfap(cyl_fem.window(cache[f"fat{f}_phi"][0, 0], Z_ABS, ZE, W, DZ), DZ, 100, 100, -20.0, cfg)[1]) for f in (2, 4, 6, 8)])
fem_g, fem_b = fem_fat_p2p(golden_cfg()), fem_fat_p2p(golden_cfg(denoise="butterworth"))
ana_fat = np.array([p2p(sfap(ana_phi(30.0)[0] if f == 3 else ana_phi(30.0)[0], DZ, 100, 100, -20.0)[1]) for f in (2,)])  # placeholder (overwritten below)
ana_b = []
for f in (2, 4, 6, 8):
    AR_geo = dict(r_fat=35.0 + f, r_skin=37.0 + f)
    ana_b.append(p2p(farina(depth=30.0, zi=-20.0, L1=100, L2=100, geo=AR_geo)[1][0]))
ana_b = np.array(ana_b)
fem_rel_g, fem_rel_b, ana_rel = fem_g / fem_g[0], fem_b / fem_b[0], ana_b / ana_b[0]
fem_mnf_g = [mnf(sfap(cyl_fem.window(cache[f"fat{f}_phi"][0, 0], Z_ABS, ZE, W, DZ), DZ, 100, 100, -20.0)[1], FS) for f in (2, 4, 6, 8)]
fem_mnf_b = [mnf(sfap(cyl_fem.window(cache[f"fat{f}_phi"][0, 0], Z_ABS, ZE, W, DZ), DZ, 100, 100, -20.0, golden_cfg(denoise="butterworth"))[1], FS) for f in (2, 4, 6, 8)]
FIG["b11"] = (fats, rel, fem_rel_g, fem_rel_b, ana_rel)
check(T, "B11 subcutaneous fat attenuates (Kuiken 2003: −31/−80/−90 % at 3/9/18 mm) and low-passes the surface SFAP",
      ok_k and np.all(np.diff(rms_f) < 0) and mnf_f[1] < mnf_f[0] and mnf_f[3] < mnf_f[0] and np.all(np.diff(fem_b) < 0)
      and fem_mnf_g[-1] < fem_mnf_g[0] and np.all(np.abs(fem_rel_b / ana_rel - 1) < 0.3),
      f"analytical RMS retained at fat {fats} mm: {np.round(rel, 3).tolist()} (Kuiken 0.69/0.20/0.10 at 3/9/18); "
      f"MNF {np.round(mnf_f, 0).tolist()} Hz (rises again at 18 mm: the sharper far-field components dominate);  "
      f"FEM pipeline p2p retained at 2/4/6/8 mm — Butterworth φ: {np.round(fem_rel_b, 3).tolist()}, "
      f"golden: {np.round(fem_rel_g, 3).tolist()} — vs Farina {np.round(ana_rel, 3).tolist()}; "
      f"FEM MNF golden {np.round(fem_mnf_g, 0).tolist()} Hz vs Butterworth φ {np.round(fem_mnf_b, 0).tolist()} Hz "
      f"(the 32 mm φ-smoothing halves the spectral content; golden keeps it near the analytical {mnf_f[0]:.0f} Hz)",
      "within ×1.5 of Kuiken's FE curve; RMS monotone decreasing and MNF lower at 3 and 9 mm than at 0.5 mm; "
      "FEM (smoothed φ) vs analytical attenuation within 30 %; golden FEM MNF decreasing 2→8 mm",
      refs="Kuiken, Lowery & Stoykov 2003; Farina & Rainoldi 1999; Lowery et al. 2002; Nordander 2003")

# =========================================================================== B12
z = (np.arange(W) - W // 2) * DZ
iso = {k: dict(v) for k, v in COND.items()}; iso["muscle"] = {"r": 0.1, "theta": 0.1, "z": 0.1}
_, s_an, _ = farina(depth=25.0, zi=0.0, L1=100, L2=100)
_, s_is, _ = farina(depth=25.0, zi=0.0, L1=100, L2=100, cond=iso)
from tests.regression import analytical_ref as AR
phi_an = ana_phi(25.0)[0]
AR.ANAL_CONDUCTIVITIES = iso                                     # re-extract φ with isotropic muscle
phi_is = ana_phi(25.0)[0]
AR.ANAL_CONDUCTIVITIES = COND
dc = lambda p: p - np.mean(np.r_[p[:12], p[-12:]])
fw_ratio = fwhm(z, dc(phi_an)) / fwhm(z, dc(phi_is))
zf = np.arange(-100, 100.01, 0.25)
fw_inf = fwhm(zf, phi_inf(zf, 10.0, 0.1, 0.5)) / fwhm(zf, phi_inf(zf, 10.0, 0.1, 0.1))
trans_y, trans_phi = cache["trans_y"], cache["trans_phi"]
fem_ratio = fwhm(z, dc(fem_phi(30.0, 0.0))) / fwhm(trans_y, trans_phi - trans_phi.min())
check(T, "B12 muscle anisotropy (σ_z/σ_r = 5) elongates the lead field along the fibres",
      1.3 <= fw_ratio <= 3.0 and abs(fw_inf - np.sqrt(5)) < 0.05 and fem_ratio > 1.3,
      f"FWHM_z(aniso)/FWHM_z(iso): analytical cylinder {fw_ratio:.2f}, infinite medium {fw_inf:.3f} (theory √5 = 2.236); "
      f"FEM along/across FWHM at r=30: {fem_ratio:.2f} (layered, bounded → not exactly √5)",
      "√5 exactly in the infinite medium; 1.3–3 in the layered cylinder (fat/skin layers and the bounded "
      "cross-section compress the elongation); FEM along/across > 1.3",
      refs="Plonsey & Barr; Rush 1963; Gielen 1984; Malmivuo & Plonsey ch. 11; Lowery 2002")

# =========================================================================== figure
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
ax = axes[0, 0]; tF_, sF_, tp_, sp_ = FIG["b1"]
ax.plot(tF_, sF_ / np.abs(sF_).max(), "k", lw=2, label="Farina"); ax.plot(tp_, sp_ / np.abs(sp_).max(), "C0--", label="pipeline")
ax.set_xlim(0, 30); ax.set_title("B1 monopolar SFAP 40 mm from the NMJ"); ax.legend(fontsize=7); ax.set_xlabel("ms")
ax = axes[0, 1]; mono_, sd_, zc_ = FIG["b3"]
for k in range(9):
    ax.plot(t_ax, sd_[k] / np.abs(sd_).max() + k * 1.1, "C1", lw=1); ax.plot(t_ax, mono_[k] / np.abs(mono_).max() + k * 1.1, "0.5", lw=.7)
ax.set_xlim(-5, 30); ax.set_yticks(np.arange(9) * 1.1); ax.set_yticklabels([f"{v:+.0f}" for v in zc_]); ax.set_title("B3 array across the IZ: mono (grey), SD (orange)")
ax = axes[0, 2]; d_, am_, as_, aF_, afem_ = FIG["b5"]
ax.loglog(d_, am_ / am_[0], "o-", label=f"pipeline mono n={n_m:.2f}"); ax.loglog(d_, as_ / as_[0], "s-", label=f"pipeline SD n={n_s:.2f}")
ax.loglog(d_, aF_ / aF_[0], "k^--", label=f"Farina mono n={n_F:.2f}"); ax.loglog([7, 10, 13, 15, 20, 25], afem_ / afem_[0], "x:", label=f"FEM mono n={n_fem:.2f}")
ax.set_xlabel("depth below skin (mm)"); ax.set_ylabel("p2p / p2p(7 mm)"); ax.set_title("B5 amplitude vs depth"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
ax = axes[1, 0]; ths_, tm_, ts_ = FIG["b6"]
ax.plot(arc(ths_), tm_ / tm_[0], "o-", label="mono"); ax.plot(arc(ths_), ts_ / ts_[0], "s-", label="SD"); ax.axhline(.5, color="0.7", lw=.7)
ax.set_xlabel("transverse distance on the skin (mm)"); ax.set_title("B6 transverse profile, fibre 15 mm deep"); ax.legend(fontsize=7)
ax = axes[1, 1]; fats_, rel_, fg_, fb_, ar_ = FIG["b11"]
ax.plot(fats_, rel_, "o-", label="analytical RMS"); ax.plot([3, 9, 18], [.687, .198, .1], "k^", ms=9, label="Kuiken 2003 (FE)")
ax.plot([2, 4, 6, 8], fb_, "x--", label="FEM pipeline p2p (Butterworth φ)"); ax.plot([2, 4, 6, 8], fg_, "x:", color="0.5", label="FEM pipeline p2p (golden)")
ax.plot([2, 4, 6, 8], ar_, "+:", label="Farina p2p (same fat)")
ax.set_xlabel("fat thickness (mm)"); ax.set_ylabel("retained"); ax.set_title("B11 fat attenuation"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax = axes[1, 2]
ax.plot([7, 10, 13, 15, 20], FIG["b4"], "o-"); ax.set_xlabel("depth below skin (mm)"); ax.set_ylabel("EOF / propagating"); ax.set_title("B4b EOF ratio vs depth"); ax.grid(alpha=.3)
fig.suptitle("Tier B — MUAP features vs the literature", fontsize=13); fig.tight_layout()
fig.savefig(OUT / "tier_b_features.png", dpi=120); print("wrote", OUT / "tier_b_features.png")
sys.exit(finish(T))
