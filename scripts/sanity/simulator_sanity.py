"""Sanity checks — the emergent physics/physiology of the full simulator chain.

These are not code-correctness checks (that is the pytest suite). They ask a different
question: *does the model behave like a muscle?* Each check measures a property of the
end-to-end chain (anatomy → FEM → pool → MUAP → activation → EMG/force) and holds it
against a physiological expectation:

  1. orderly recruitment (Henneman size principle)
  2. onion-skin rate coding
  3. interference-EMG amplitude grows with contraction
  4. EMG ↔ force: monotone, MVC-calibrated
  5. conduction velocity from HD-EMG propagation   (the whole chain, one number)
  6. spatial selectivity of a MUAP on the array
  7. non-stationarity over a movement
  8. MUAP physiological scale                       (honest C-04 fibre-geometry flag)

Run: python scripts/sanity/simulator_sanity.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emgforge.activation import MotoneuronPool, TwitchPool, drive, compound_emg
from emgforge.activation.dynamic import (
    angle_track, drive_from_angle, amp_from_angle, warp_from_angle, dynamic_compound_emg,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/sanity/simulator"; OUT.mkdir(parents=True, exist_ok=True)
FS = 2048
RESULTS = []          # (name, passed, known)
STORE = {}            # values kept for the summary figure


RECORDS = []          # same record format as scripts/validation/harness.py → tier S rows


def check(name, passed, measured, expect, note="", known=False):
    RESULTS.append((name, bool(passed), known))
    RECORDS.append(dict(tier="S", name=name, passed=bool(passed), known=known, measured=str(measured),
                        expect=str(expect) + (f" ({note})" if note else ""), principle="", refs=""))
    tag = "KNOWN" if (known and not passed) else ("PASS" if passed else "FAIL")
    mark = {"PASS": "✓", "FAIL": "✗", "KNOWN": "⚠"}[tag]
    print(f"  [{mark} {tag:5s}] {name}")
    print(f"            measured : {measured}")
    print(f"            expect   : {expect}" + (f"   ({note})" if note else ""))
    print()


# ----------------------------------------------------------------------------- load
import sys as _sys; _sys.path.insert(0, str(ROOT / "scripts/validation"))
from muap_bank import load_bank                              # EMGFORGE_MUAP_BANK selects the bank
B = load_bank(ROOT)
muaps1 = B.muaps                                             # (100, 256) @ 2048 Hz, volts
p2p, dur = B.p2p, B.duration_ms
Wg, M = B.W, B.M                                             # (n_mu, M*M, 256)
ied_z = B.ied_z                                              # mm, along fibre
pool = MotoneuronPool(n_mu=len(muaps1), fs=FS)
print(f"MUAP bank: {B.name}")

print("=" * 74)
print("emgforge — simulator sanity checks   (physics/physiology of the full chain)")
print("=" * 74, "\n")


# ---------------------------------------------------------- 1) orderly recruitment
levels = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
nact = np.array([pool.n_active(np.array([L])) for L in levels])
ordered = bool(np.all(np.diff(pool.rte) >= 0))              # small units recruited first
mono = bool(np.all(np.diff(nact) >= 0))
STORE["recruit"] = (levels, nact)
check("Orderly recruitment (Henneman size principle)",
      mono and ordered,
      f"n_active @ drive {levels.tolist()} = {nact.tolist()}; thresholds sorted={ordered}",
      "n_active non-decreasing in drive; smallest units recruited first")


# ---------------------------------------------------------- 2) onion-skin rate coding
rate = pool.firing_rate(np.array([0.7]))[:, 0]
act = np.where(rate > 0)[0]
onion = rate[act[0]] > rate[act[-1]]
check("Onion-skin rate coding",
      onion,
      f"@drive 0.7: first-recruited {rate[act[0]]:.1f} Hz vs last {rate[act[-1]]:.1f} Hz "
      f"({len(act)} active)",
      "earlier (smaller) units discharge faster than later (larger) ones")


# ------------------------------------------------ 3) EMG amplitude grows with contraction
levs = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
rms = []
for L in levs:
    E = drive.trapezoid(L, 0.3, 1.2, 0.3, fs=FS, lead_s=0.15)
    sp = pool.spike_trains(E, seed=0)
    emg = compound_emg(sp, muaps1, n_samples=len(E))
    lo, hi = int(0.35 * len(emg)), int(0.65 * len(emg))     # plateau only
    rms.append(float(np.sqrt((emg[lo:hi] ** 2).mean())))
rms = np.array(rms)
STORE["emg_rms"] = (levs, rms)
check("Interference-EMG amplitude ↑ with contraction",
      bool(np.all(np.diff(rms) > 0)),
      f"plateau RMS(µV) @ {levs} = {np.round(rms * 1e6, 3).tolist()}",
      "monotone increase with drive")


# ---------------------------------------------------- 4) EMG ↔ force monotone + MVC
tw = TwitchPool(pool, fs=FS)
forces = []
for L in levs + [1.0]:
    E = drive.trapezoid(L, 0.5, 1.5, 0.5, fs=FS, lead_s=0.2)
    sp = pool.spike_trains(E, seed=0)
    f = tw.force(sp, n_samples=len(E))
    forces.append(float(np.percentile(f, 95)))              # plateau force (fraction of MVC)
forces = np.array(forces)
STORE["force"] = (levs + [1.0], forces)
mono_f = bool(np.all(np.diff(forces) > 0))
mvc_ok = 0.8 <= forces[-1] <= 1.2
check("EMG ↔ force: monotone and MVC-calibrated",
      mono_f and mvc_ok,
      f"plateau force(%MVC) @ {levs + [1.0]} = {np.round(forces * 100, 1).tolist()}",
      "monotone in drive; ≈100 %MVC at full drive")


# ------------------------------------------------- 5) conduction velocity from propagation
def _sub_lag(a, b):
    """Cross-correlation lag of b vs a, refined to sub-sample by parabolic interpolation."""
    cc = np.correlate(b, a, "full")
    k = int(np.argmax(cc))
    delta = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return (k + delta) - (len(a) - 1)


def column_cv(mu):
    """Best-column CV (m/s), fit quality R², for one MU on the grid."""
    grid = Wg[mu].reshape(M, M, -1)                         # (row=along fibre, col, t)
    best = None
    for j in range(M):
        col = grid[:, j]                                    # (M, t) rows along the fibre
        if np.ptp(col) == 0:
            continue
        lags = np.array([0.0] + [_sub_lag(col[0], col[i]) for i in range(1, M)])
        A = np.vstack([np.arange(M), np.ones(M)]).T
        slope, _ = np.linalg.lstsq(A, lags, rcond=None)[0]
        fit = A @ np.linalg.lstsq(A, lags, rcond=None)[0]
        ss = 1 - np.sum((lags - fit) ** 2) / max(np.sum((lags - lags.mean()) ** 2), 1e-9)
        if abs(slope) < 1e-6:
            continue
        dt_ms = 1000.0 / FS
        cv = ied_z / (abs(slope) * dt_ms)                  # mm / ms = m/s
        energy = float(np.sqrt((col ** 2).mean()))
        cand = (ss * energy, cv, ss)
        if best is None or cand[0] > best[0]:
            best = cand
    return best                                            # (score, cv, r2)

cv_list = [(mu, *c) for mu in range(Wg.shape[0])
           if (c := column_cv(mu)) is not None and c[2] > 0.9]   # (mu, score, cv, r2)
cvs = np.array([c[2] for c in cv_list])
sz = B.sizes[[c[0] for c in cv_list]]
cv_med = float(np.median(cvs))
corr_sz = float(np.corrcoef(sz, cvs)[0, 1])                 # ~0 → single model CV, no diameter scaling
mu_rep = max(cv_list, key=lambda c: c[1])[0]               # strongest signal, for the figure/selectivity
STORE["cv"] = (mu_rep, cv_med, cvs)
check("Conduction velocity from HD-EMG propagation",
      3.0 <= cv_med <= 6.0,
      f"pool median CV {cv_med:.2f} m/s (range {cvs.min():.1f}–{cvs.max():.1f} over {len(cvs)} MUs, "
      f"IED {ied_z:.1f} mm); corr(size, CV) = {corr_sz:+.2f}",
      "physiological muscle-fibre CV, mean ≈4 m/s — the model uses one fibre CV (≈uniform across the pool)")


# ------------------------------------------------------ 6) spatial selectivity of a MUAP
grid_rms = np.sqrt((Wg[mu_rep].reshape(M, M, -1) ** 2).mean(-1))
sel = float(grid_rms.max() / grid_rms.mean())
STORE["sel"] = (mu_rep, grid_rms)
check("Spatial selectivity of a MUAP on the array",
      sel > 1.5,
      f"peak-channel RMS / grid-mean RMS = {sel:.2f} (MU {mu_rep})",
      "a MUAP is localised on the array, not a uniform far field (ratio > 1.5)")


# ------------------------------------------------------- 7) non-stationarity over movement
DUR = 6.0
angle = angle_track(DUR, fs=FS, cycles=2)
Em = drive.add_common_drive(drive_from_angle(angle), sigma=0.012, fs=FS, seed=1)
amp = amp_from_angle(angle, gain=0.8); warp = warp_from_angle(angle, gain=0.22)
spm = pool.spike_trains(Em, seed=0)
dyn = dynamic_compound_emg(spm, muaps1, amp, warp, n_samples=len(Em))
flex = angle > 0.75; ext = angle < 0.25
rms_flex = float(np.sqrt((dyn[flex] ** 2).mean()))
rms_ext = float(np.sqrt((dyn[ext] ** 2).mean()))
ratio = rms_flex / max(rms_ext, 1e-12)
STORE["dyn"] = (np.arange(len(dyn)) / FS, dyn, angle)
check("Non-stationarity over a movement",
      ratio > 1.15,
      f"RMS flexed / RMS extended = {ratio:.2f}  ({rms_flex*1e6:.3f} vs {rms_ext*1e6:.3f} µV)",
      "EMG amplitude is modulated by joint angle, not just by the drive")


# ------------------------------------------------------- 8) MUAP physiological scale (C-04)
valid = p2p > 0
med_p2p = float(np.median(p2p[valid]) * 1e6)               # µV
med_dur = float(np.median(dur))                            # ms
# surface MUAP: tens to hundreds of µV (largest superficial units 1–2 mV); main complex
# 5–20 ms (Merletti & Muceli 2019 ≈15 ms for a 60 mm fibre at 4 m/s; Farina 2014 ~20 ms)
scale_ok = (20 <= med_p2p <= 800) and (5 <= med_dur <= 20)
STORE["scale"] = (p2p[valid] * 1e6, med_dur)
check("MUAP physiological scale",
      scale_ok,
      f"median p2p {med_p2p:.2f} µV, median duration {med_dur:.1f} ms",
      "single-MU surface MUAP ≈ 20–800 µV, 5–20 ms (Merletti & Muceli 2019; Farina 2014)",
      note="the 0.6 µV / 33 ms of earlier runs was an electrode-placement artefact plus the 1/v constant",
      known=True)


# ------------------------------------------------------------------------- summary
n_real = sum(1 for _, p, k in RESULTS if not k)
n_pass = sum(1 for _, p, k in RESULTS if p and not k)
n_known = sum(1 for _, p, k in RESULTS if k and not p)
print("=" * 74)
print(f"  {n_pass}/{n_real} physics checks passed"
      + (f"  ·  {n_known} known limitation flagged (C-04 fibre geometry)" if n_known else ""))
print("=" * 74)


# --------------------------------------------------------------------- summary figure
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])                              # recruitment
lv, na = STORE["recruit"]; ax.plot(lv, na, "o-", color="tab:blue")
ax.set_title("1 · recruitment"); ax.set_xlabel("drive"); ax.set_ylabel("# active MUs"); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[0, 1])                              # EMG RMS + force vs drive
lv, rr = STORE["emg_rms"]; fl_, ff = STORE["force"]
ax.plot(lv, rr * 1e6, "s-", color="tab:green", label="EMG RMS (µV)")
axb = ax.twinx(); axb.plot(fl_, ff * 100, "^-", color="tab:red", label="force (%MVC)")
ax.set_title("3–4 · EMG & force ↑ drive"); ax.set_xlabel("drive")
ax.set_ylabel("EMG RMS (µV)", color="tab:green"); axb.set_ylabel("force (%MVC)", color="tab:red")
ax.grid(alpha=.3)

ax = fig.add_subplot(gs[0, 2])                              # CV waterfall
mu_rep, cv_med, cvs = STORE["cv"]; grid = Wg[mu_rep].reshape(M, M, -1)
je = int(np.argmax(np.sqrt((grid ** 2).mean(-1)).max(0)))
tms = np.arange(grid.shape[-1]) / FS * 1000
for i in range(M):
    ax.plot(tms, grid[i, je] * 1e6 + i * grid.max() * 1e6 * 1.2, color=plt.cm.viridis(i / M))
ax.set_title(f"5 · propagation → CV {cv_med:.1f} m/s"); ax.set_xlabel("ms"); ax.set_ylabel("row")
ax.set_yticks([]); ax.set_xlim(0, min(60, tms[-1]))

ax = fig.add_subplot(gs[1, 0])                              # spatial selectivity
mu_s, grms = STORE["sel"]
im = ax.imshow(grms * 1e6, cmap="magma"); ax.set_title(f"6 · selectivity (MU {mu_s})")
ax.set_xlabel("across arm"); ax.set_ylabel("along arm"); fig.colorbar(im, ax=ax, shrink=.8, label="RMS µV")

ax = fig.add_subplot(gs[1, 1])                              # dynamic non-stationarity
td, dd, ang = STORE["dyn"]
ax.plot(td, dd * 1e6, lw=.3, color="0.6")
axb = ax.twinx(); axb.plot(td, ang, color="tab:purple", lw=1.6)
ax.set_title("7 · non-stationary EMG"); ax.set_xlabel("s"); ax.set_ylabel("µV")
axb.set_ylabel("joint angle", color="tab:purple")

ax = fig.add_subplot(gs[1, 2])                              # MUAP scale vs physiology
vals, medd = STORE["scale"]
ax.hist(vals, bins=30, color="tab:gray"); ax.axvspan(20, 800, color="tab:green", alpha=.15)
ax.axvline(np.median(vals), color="tab:red", lw=1.5)
ax.set_title("8 · MUAP p2p vs physiology"); ax.set_xlabel("p2p (µV)"); ax.set_ylabel("# MUs")
ax.text(np.median(vals) + 15, ax.get_ylim()[1] * 0.6, f"median\n{np.median(vals):.1f} µV",
        color="tab:red", fontsize=8, va="center")
ax.text(.97, .9, "physiological\nband (shaded)\n→ C-04 gap", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="tab:green")

fig.suptitle(f"emgforge sanity — {n_pass}/{n_real} physics checks pass"
             + (f", {n_known} known C-04 flag" if n_known else ""), fontsize=13)
fig.savefig(OUT / "simulator_sanity.png", dpi=130, bbox_inches="tight")
print("\nwrote", OUT / "simulator_sanity.png")
import json
_vout = ROOT / "_results/validation"; _vout.mkdir(parents=True, exist_ok=True)
(_vout / "S.json").write_text(json.dumps(RECORDS, indent=1))     # rows for the validation report

import sys
sys.exit(0 if n_pass == n_real else 1)              # known-limitation flags don't fail the run
