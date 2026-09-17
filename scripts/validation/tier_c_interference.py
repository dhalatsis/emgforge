"""Tier C — interference EMG and the motor-unit pool against the literature.

Complements scripts/sanity/simulator_sanity.py (recruitment order, onion skin, EMG↑drive,
EMG↔force monotone/MVC, HD-EMG conduction velocity, selectivity, non-stationarity,
MUAP scale) with the *statistical* signatures the literature gives numbers for:

  C1  discharge rates 5–40 pps, onion-skin ordering       C5  amplitude cancellation ≈30 % → ≈60 %
  C2  ISI variability CoV 0.1–0.3, refractory floor        C6  RMS–force: RMS(50 %)/RMS(100 %) ∈ [0.3, 0.6]
  C3  recruitment thresholds right-skewed; twitch ≈100×    C7  spectral content (MNF/MDF) — C-04 flagged
  C4  amplitude PDF: kurtosis > 3 at low force → ≈3 high   C8  MU size ↔ MUAP amplitude (Henneman, recording side)

Run:  python scripts/validation/tier_c_interference.py   (~30 s; uses the cached MU pool)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import *                                    # noqa: F401,F403
from scipy.stats import kurtosis, skew, spearmanr
from emgforge.activation import MotoneuronPool, TwitchPool, drive, compound_emg
from emgforge.activation.pool import ISI_CV

T = "C"
FS_C = 2048.0
from muap_bank import load_bank                          # EMGFORGE_MUAP_BANK selects the bank
B = load_bank(ROOT)
muaps = B.muaps                                          # (100, 256) @ 2048 Hz, monopolar, volts
Wg, sizes = B.W, B.sizes                                 # (n_mu, 25, 256), fibres per MU
pool = MotoneuronPool(n_mu=len(muaps), fs=FS_C)
print(f"MUAP bank: {B.name}\n")
print("=" * 74); print("Tier C — interference EMG & motor-unit pool vs the literature"); print("=" * 74, "\n")


def plateau(level, hold=2.0, seed=0):
    E = drive.trapezoid(level, 0.3, hold, 0.3, fs=FS_C, lead_s=0.15)
    sp = pool.spike_trains(E, seed=seed)
    lo, hi = int((0.45 + 0.3) * FS_C), int((0.45 + 0.3 + hold - 0.3) * FS_C)   # steady part
    return E, sp, slice(lo, hi)

# =========================================================================== C1
E, sp, sl = plateau(0.5)
dur = (sl.stop - sl.start) / FS_C
rates = np.array([np.sum((s >= sl.start) & (s < sl.stop)) / dur for s in sp])
act = rates > 0
rho_onion = spearmanr(pool.rte[act], rates[act]).correlation
E1, sp1, sl1 = plateau(1.0)
rates1 = np.array([np.sum((s >= sl1.start) & (s < sl1.stop)) / ((sl1.stop - sl1.start) / FS_C) for s in sp1])
check(T, "C1 discharge rates lie in 5–40 pps and follow the onion skin (earlier-recruited fire faster)",
      pool.min_fr.min() >= 5 and pool.peak_fr.max() <= 40 and rates[act].min() >= 5
      and rates1.max() <= 1.1 * pool.peak_fr.max() and rho_onion < -0.9,
      f"model: min rate {pool.min_fr.min():.0f}, peak {pool.peak_fr.max():.0f} pps; measured @drive 0.5: {act.sum()} active, "
      f"{rates[act].min():.1f}–{rates[act].max():.1f} pps, Spearman(threshold, rate) = {rho_onion:+.2f}; "
      f"@drive 1.0 max {rates1.max():.1f} pps (renewal-process estimate runs ~5 % above the nominal peak)",
      "model rates within 5–40 pps; measured within 10 % of nominal; ρ(threshold, rate) < −0.9",
      refs="De Luca & Hostage 2010 (onion skin); Merletti & Muceli 2019; Farina 2014")

# =========================================================================== C2
isi = np.concatenate([np.diff(s[(s >= sl.start) & (s < sl.stop)]) for s in sp if np.sum((s >= sl.start) & (s < sl.stop)) > 3]) / FS_C
cov = np.array([np.std(d) / np.mean(d) for s in sp if len(d := np.diff(s[(s >= sl.start) & (s < sl.stop)]) / FS_C) > 3])
check(T, "C2 inter-spike-interval variability: CoV 0.1–0.3, no ISI below the ~20 ms refractory floor",
      0.1 <= np.median(cov) <= 0.3 and np.mean(isi < 0.020) < 0.01,
      f"median per-MU ISI CoV {np.median(cov):.2f} (model ISI_CV = {ISI_CV:.3f}); ISIs < 20 ms: {100 * np.mean(isi < 0.02):.2f} %; min ISI {1000 * isi.min():.1f} ms",
      "CoV ∈ [0.1, 0.3]; < 1 % of ISIs under 20 ms. The Gaussian renewal model has no refractory floor — "
      "at 40 pps (ISI 25 ms, σ 4 ms) sub-20 ms intervals are common; clip the ISI or use a gamma/lognormal draw",
      refs="Clamann 1969; Nordstrom 1992; Dideriksen 2012 (20 % used in models)")

# =========================================================================== C3
tw = TwitchPool(pool, fs=FS_C)
check(T, "C3 recruitment thresholds are right-skewed (many low, few high); twitch forces span ≈100×",
      skew(pool.rte) > 0.5 and 50 <= tw.P.max() / tw.P.min() <= 200,
      f"threshold skewness {skew(pool.rte):+.2f} (median/max = {np.median(pool.rte) / pool.rte.max():.2f}); "
      f"twitch P_max/P_min = {tw.P.max() / tw.P.min():.0f}",
      "skewness > 0.5; 50–200× (Fuglevand RP = 100; FDI 130)",
      refs="Fuglevand, Winter & Patla 1993; Petersen & Rostalski 2019")

# =========================================================================== C4
def emg_at(level, seed=0):
    E, sp, sl = plateau(level, seed=seed)
    return compound_emg(sp, muaps, n_samples=len(E))[sl]
kurt = {L: float(kurtosis(emg_at(L), fisher=False)) for L in (0.05, 0.1, 0.5, 0.9)}
arv_rms = {L: float(np.mean(np.abs(x := emg_at(L))) / np.sqrt(np.mean(x ** 2))) for L in (0.3, 0.7)}
check(T, "C4 amplitude PDF: super-Gaussian at low force, → Gaussian (kurtosis 3–3.5) at high; ARV/RMS ∈ [0.71, 0.80]",
      kurt[0.05] > kurt[0.9] and 2.8 <= kurt[0.9] <= 3.6 and all(0.71 <= v <= 0.80 for v in arv_rms.values()),
      f"kurtosis @drive 0.05/0.1/0.5/0.9 = {[round(v, 2) for v in kurt.values()]}; ARV/RMS @0.3/0.7 = "
      f"{[round(v, 3) for v in arv_rms.values()]} (Laplacian 0.707, Gaussian 0.798)",
      "kurtosis decreasing with force to 2.8–3.6; ARV/RMS between Laplacian and Gaussian. NB: with the C-04 "
      "MUAPs (33 ms long) the low-force EMG is denser than real, so the low-force excess kurtosis is understated",
      refs="Clancy & Hogan 1999; Nazarpour 2013; Merletti & Muceli 2019 (filling factor 0.5–0.63)")

# =========================================================================== C5
def cancellation(level):
    E, sp, sl = plateau(level)
    total = compound_emg(sp, muaps, n_samples=len(E))[sl]
    rect_sum = sum(np.mean(np.abs(compound_emg([s], muaps[i:i + 1], n_samples=len(E))[sl])) for i, s in enumerate(sp) if len(s))
    return 1.0 - np.mean(np.abs(total)) / rect_sum
canc = {L: float(cancellation(L)) for L in (0.2, 0.5, 1.0)}
check(T, "C5 amplitude cancellation grows with excitation (≈30 % at 20 % → ≈60 % at maximum)",
      canc[0.2] < canc[1.0] and 0.15 <= canc[0.2] <= 0.6 and 0.4 <= canc[1.0] <= 0.85,
      f"1 − mean|Σ trains| / Σ mean|train_i| @drive 0.2/0.5/1.0 = {[round(100 * v) for v in canc.values()]} %",
      "increasing; 15–60 % at 0.2, 40–85 % at 1.0 (Keenan 2005: 33 % → 62–65 %). Cancellation is set by MUAP "
      "duration × active-MU count: with the C-04 33 ms MUAPs it saturates already at low drive",
      refs="Keenan, Farina, Maluf, Merletti & Enoka 2005; Farina, Merletti & Enoka 2014", known=True)

# =========================================================================== C6
levels = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
rms, force = [], []
for L in levels:
    E, sp, sl = plateau(L, hold=1.5)
    rms.append(float(np.sqrt(np.mean(compound_emg(sp, muaps, n_samples=len(E))[sl] ** 2))))
    force.append(float(np.percentile(tw.force(sp, n_samples=len(E))[sl], 50)))
rms, force = np.array(rms), np.array(force)
r50 = float(np.interp(0.5 * force[-1], force, rms) / rms[-1])
check(T, "C6 EMG–force relation lies between linear and quadratic: RMS at 50 % force = 0.3–0.6 of RMS at 100 %",
      0.3 <= r50 <= 0.6 and np.all(np.diff(force) > 0),
      f"RMS(50 % MVC)/RMS(100 % MVC) = {r50:.2f}; force plateau %MVC = {np.round(100 * force).astype(int).tolist()} at drive {levels}",
      "0.3 (quadratic, biceps/triceps) – 0.6 (linear, FDI/soleus); force monotone in drive. A ratio above 0.6 "
      "(EMG concave in force) follows from the saturated cancellation in C5 — the same C-04 signature",
      refs="Lawrence & De Luca 1983; Woods & Bigland-Ritchie 1983; Fuglevand 1993; Keenan & Valero-Cuevas 2007",
      known=True)

# =========================================================================== C7
def mdf(x, fs):
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    c = np.cumsum(X); return float(f[np.searchsorted(c, c[-1] / 2)])
x5 = emg_at(0.5)
spec = dict(mnf=mnf(x5, FS_C), mdf=mdf(x5, FS_C))
check(T, "C7 interference-EMG spectrum: MNF/MDF at moderate force in 70–130 Hz",
      70 <= spec["mdf"] <= 130,
      f"@drive 0.5 (monopolar): MNF {spec['mnf']:.0f} Hz, MDF {spec['mdf']:.0f} Hz",
      "MDF 70–130 Hz (bipolar norms: biceps 90±18, TA 116±20). Expected to fail while the C-04 fibre geometry "
      "yields 33 ms MUAPs — the spectrum is bounded by the MUAP duration",
      refs="Lindström & Magnusson 1977; initial-MDF norms (J Clin Neurophysiol 1998); De Luca 2002",
      known=True)

# =========================================================================== C8
p2p_grid = np.array([np.ptp(Wg[k].reshape(-1)) for k in range(Wg.shape[0])])
rho_sz = spearmanr(sizes, p2p_grid).correlation
check(T, "C8 larger motor units produce larger MUAPs (size ↔ amplitude on the recording side)",
      rho_sz > 0.5,
      f"Spearman(fibres per MU, grid-max p2p) = {rho_sz:+.2f} over {len(sizes)} MUs (sizes {sizes.min()}–{sizes.max()} fibres)",
      "ρ > 0.5 (amplitude ∝ fibre count at fixed depth; depth scatter lowers ρ)",
      refs="Roeleveld 1998; Merletti & Muceli 2019 §2.4; Del Vecchio 2017 (size ↔ MUAP)")

# =========================================================================== figure
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
ax = axes[0]; ax.scatter(pool.rte[act], rates[act], s=10); ax.set_xlabel("recruitment threshold"); ax.set_ylabel("rate @0.5 (pps)"); ax.set_title(f"C1 onion skin ρ={rho_onion:+.2f}")
ax = axes[1]
for L in (0.05, 0.9):
    x = emg_at(L); ax.hist(x / x.std(), bins=80, density=True, histtype="step", label=f"drive {L:g}, k={kurt[L]:.2f}")
xx = np.linspace(-4, 4, 200); ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="Gaussian")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1); ax.legend(fontsize=7); ax.set_title("C4 amplitude PDF")
ax = axes[2]; ax.plot(100 * force, rms / rms[-1], "o-"); ax.plot([0, 100], [0, 1], "k:", lw=.8, label="linear"); ax.plot(np.linspace(0, 100), (np.linspace(0, 1)) ** 2, "k--", lw=.8, label="quadratic")
ax.set_xlabel("force (%MVC)"); ax.set_ylabel("RMS / RMS(MVC)"); ax.set_title(f"C6 EMG–force, RMS(50%)={r50:.2f}"); ax.legend(fontsize=7)
ax = axes[3]; ax.plot(list(canc.keys()), [100 * v for v in canc.values()], "o-"); ax.plot([0.2, 0.8], [33, 65], "k^", label="Keenan 2005"); ax.set_xlabel("drive"); ax.set_ylabel("cancellation (%)"); ax.set_title("C5 amplitude cancellation"); ax.legend(fontsize=7)
fig.suptitle("Tier C — interference EMG & pool vs the literature", fontsize=12); fig.tight_layout()
fig.savefig(OUT / "tier_c_interference.png", dpi=120); print("wrote", OUT / "tier_c_interference.png")
sys.exit(finish(T))
