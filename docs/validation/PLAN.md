# Model validation plan

How we know the simulator is right — a tiered set of sanity checks, each holding one
measurable property of the chain against a literature principle with a numeric pass
criterion. Code: `scripts/validation/`. Report with live numbers:
`_results/validation/VALIDATION_REPORT.md` (regenerate with `python scripts/validation/run_all.py`).
Literature and the numbers behind every criterion: `docs/validation/BIBLIOGRAPHY.md`.

```
tier A  cylinder must reproduce     first principles → analytical (Farina 2004) → FEM → golden pipeline
tier B  MUAP features               the replicated phenomenology, on the analytical oracle AND on our pipeline
tier C  interference EMG & pool     the statistical signatures of the whole chain
tier S  chain-level sanity          scripts/sanity/simulator_sanity.py (recruitment, EMG–force, CV, …)
```

What is deliberately *not* here: code-correctness gates (`tests/`, 273 green) and the
byte-exact regression set — those pin behaviour; this plan asks whether the behaviour is
physically right.

---

## How to run

```bash
P=/home/dc23/miniconda3/envs/fenicsx-env/bin/python
$P scripts/validation/run_all.py            # everything, ~4 min (+2 min first time: FEM cache)
$P scripts/validation/tier_a_cylinder.py    # one tier at a time
```

The FEM lead fields come from one cache (`scripts/validation/cyl_fem.py` →
`_results/validation/fem_cache/`): a single reciprocal solve on the analytical-radii
cylinder (10/35/38/40 mm) sampled along full-length fibre lines at six depths and ten
angles, plus symmetry partners (electrode rotated 30°, shifted 20 mm), interior-point
reciprocity, a σ=1 mm source, and the four fat-thickness meshes. The cylinder is
z-invariant, so *an electrode at +z ≡ the lead field translated by +z*; that is how
every array, SD and DD montage in tiers A/B is built without extra solves.

Every check calls `harness.check(tier, name, passed, measured, expect, principle, refs,
known=…)`. `known=True` marks a documented limitation: it is reported, keeps its
criterion (so it flips to PASS when fixed), and does not fail the tier.

---

## Tier A — the cylinder must reproduce

| id | check | criterion | principle / ref |
|---|---|---|---|
| A0.1 | spatial engine ≡ first-principles line-source integral (closed-form φ in an infinite anisotropic medium, dual form `∫ Vm·win·φ'' dz` on a 0.02 mm grid, no numerical derivative of the tendon step) | r ≥ 0.999, \|lag\| ≤ 0.1 ms, three NMJ offsets | Rosenfalck 1969; Dimitrov & Dimitrova 1998 |
| A0.2 | the engine's amplitude constant | ratio 1 ± 5 %, independent of v | core-conductor line source |
| A0.3 | Fourier engine vs first principles (+ mirrored-IAP diagnostic) | signed r ≥ 0.99, no lag | Farina & Merletti 2001; GOLDEN_METHOD §6/§8 |
| A0.4a | sampling invariance (fsamp, dz) | r > 0.995, amplitude within 2 % | discretisation convergence |
| A0.4b | electrode translation Δz delays the propagating lobe by Δz/v | within one sample | z-invariance |
| A0.4c | superposition | exact | linearity |
| A0.4d | CV scaling: duration ∝ 1/v, MNF ∝ v | within 8 % | Lindström & Magnusson 1977 |
| A0.5 | the CSD source is monopole-free, ∫ i(z,t) dz = 0 ∀t | < 1e-6 | Petersen 2016; Merletti 1999 |
| A1.1 | FEM φ(z) vs analytical φ(z), five depths | r ≥ 0.99, FWHM within 20 % | Farina 2004; Maksymenko 2023 |
| A1.1b | … and φ''(z), the kernel the SFAP integrates, vs analytical electrode radius | r ≥ 0.98 at the best-matched radius | dual form |
| A1.2 | FEM/analytical SFAP amplitude ratio constant across depth | ≤ 15 % spread (Fourier route; smoothed spatial ≤ 35 %) | same depth law |
| A1.3 | FEM φ vs analytical for fibres 10–45° off the meridian | r ≥ 0.99 | lateral decay |
| A2.1 | golden pipeline on FEM φ ≡ on analytical φ | r ≥ 0.95 (deep ≥ 0.99) | isolates the volume conductor |
| A2.2 | monopole denoise removes ripple without changing the analytical answer | jaggedness ↓, agreement not reduced | GOLDEN_METHOD §2 |
| A2.3 | end-of-fibre onset at L/v (EOF isolated by subtracting a longer fibre) | pipeline within 0.5 ms; Farina reported | Gootzen 1991; Rodriguez-Falces & Place 2018 |
| A2.4 | CV from a longitudinal array, both models | within 5 %, R² > 0.99 | Farina & Merletti 2004 |
| A2.5 | waveform \|r\| vs the Farina generator (sign-agnostic, ±6 ms) | ≥ 0.9 | — |
| A3.1 | rotational symmetry (electrode +30° ≡ fibre −30°) | r > 0.999, peak within 3 % | cylinder symmetry |
| A3.2 | axial translation (electrode +20 mm ≡ window −20 mm) | r > 0.995, peak within 3 % | Neumann end effect |
| A3.3 | reciprocity between two interior points | ratio 1 ± 5 % | Helmholtz; Malmivuo & Plonsey ch. 11 |

## Tier B — MUAP features vs the literature

| id | check | criterion | ref |
|---|---|---|---|
| B1 | monopolar SFAP between IZ and tendon: + − +, main lobe negative (both models) | lobe sequence | Merletti & Muceli 2019 Fig. 2 |
| B2 | CV from an SD array tracks the set CV at 3/4/5 m/s | within 5 % | Farina & Merletti 2004; Zwarts 1988 |
| B3 | IZ signature: monopolar mirror symmetry; SD null (< 5 %) and phase reversal on the IZ | as stated | Masuda 1983/85 |
| B4a | EOF is non-propagating: same onset on all electrodes | spread < 0.5 ms | Mesin 2005 |
| B4b | EOF/propagating ratio increases with depth | monotone | Dimitrova & Dimitrov 2013 |
| B4c | mono > SD > DD for the EOF ratio | strict | Roeleveld 1998; Farina 2002 |
| B5 | amplitude vs depth is a power law, SD steeper than mono (pipeline, Farina, FEM) | R² > 0.95 | Roeleveld 1997a; Fuglevand 1992 |
| B6 | transverse spread: mono ≥ SD, widths grow with depth; Roeleveld's depth ≈ 0.2 × 50 %-width reported | as stated | Roeleveld 1997b/2013 |
| B7 | MUAP ∝ fibre count; NMJ scatter disperses and lengthens | exact / qualitative | Merletti & Muceli 2019 |
| B8 | bipolar spectral null at f = CV/IED (on the SD/mono ratio, propagating complex windowed) | within 6 % | Lindström & Magnusson 1977; Lynn 1978 |
| B9 | electrode size is a spatial low-pass; Ø5 mm ≈ −3 dB at 100 c/m | monotone; −3 dB points | Merletti & Muceli 2019 Table 1 |
| B10 | SD amplitude ∝ IED when IED ≪ λ, saturating near λ/2 | ×2 IED → ≥ 1.7× small; < 2× at 10→20 mm | De Luca 2002 |
| B11 | fat attenuates (Kuiken −31/−80/−90 % at 3/9/18 mm) and low-passes; FEM vs analytical attenuation | ×1.5 of Kuiken; within 30 % | Kuiken 2003; Lowery 2002 |
| B12 | anisotropy elongates the lead field along the fibres | √5 in the infinite medium; 1.3–3 layered | Rush 1963; Plonsey & Barr |

## Tier C — interference EMG and the pool

| id | check | criterion | ref |
|---|---|---|---|
| C1 | discharge rates 5–40 pps, onion skin | ρ(threshold, rate) < −0.9 | De Luca & Hostage 2010 |
| C2 | ISI CoV 0.1–0.3, refractory floor | < 1 % ISIs under 20 ms | Dideriksen 2012 |
| C3 | thresholds right-skewed; twitch range ≈ 100× | skew > 0.5; 50–200× | Fuglevand 1993 |
| C4 | amplitude PDF: super-Gaussian at low force → Gaussian; ARV/RMS 0.71–0.80 | kurtosis 2.8–3.6 at high force | Clancy & Hogan 1999 |
| C5 | amplitude cancellation ≈ 30 % → ≈ 60 % | increasing, in band | Keenan 2005 |
| C6 | EMG–force between linear and quadratic | RMS(50 %)/RMS(100 %) ∈ [0.3, 0.6] | Lawrence & De Luca 1983 |
| C7 | MDF at moderate force 70–130 Hz | in band | Stulen & De Luca 1981 |
| C8 | larger MUs → larger MUAPs | Spearman > 0.5 | Roeleveld 1998 |

Tier S (existing, `scripts/sanity/simulator_sanity.py`): orderly recruitment, onion skin,
EMG ↑ drive, EMG–force monotone and MVC-calibrated, HD-EMG conduction velocity, spatial
selectivity, non-stationarity, MUAP physiological scale (C-04 flag).

---

## Status and findings (2026-09-10)

Tier A 14/15 (4 known; the fail is A1.1b), tier B 14/14, tier C 4/5 (3 known; the fail
is C2), tier S 7/7 (1 known). The numbers are in the report; the findings that matter:

**1. The spatial engine is the physics.** Against a closed-form line-source oracle it
scores r = 1.0000 with zero lag in every geometry (A0.1), its source is monopole-free to
1e-16 (A0.5), it is sampling-, translation- and superposition-exact (A0.4), and its
end-of-fibre onset lands at L/v within 0.4 ms (A2.3). The golden recipe's one-sided
tendon taper costs r 0.998 vs the sharp-tendon oracle — a deliberate physics choice.

**2. The Fourier/Farina family disagrees with first principles.** The Fourier engine
scores r = −0.64 … −0.79 with 1–2.5 ms lags against the same oracle (A0.3), the Farina
generator's EOF onset leads L/v by 2.5 ms (A2.3, B4a) and its amplitude-vs-depth exponent
differs from the first-principles one on its own φ (B5). This resolves the three
"pinned disagreements" in `GOLDEN_METHOD.md` §6 in favour of the spatial engine and
makes the Fourier engine's IAP orientation/sign conventions (`signal_generator.py`:
`V2 = −flip(…)`, the output `flip`) the thing to audit. The Fourier "r = 0.997 vs MATLAB"
oracle has no artifact in the repo. Its `np.arange` k-grid also fails to build at some CVs
(257 ≠ 256 bins at v = 3 or 5 m/s).

**3. Amplitude constant: `sfap = (CSD @ φ)·dz/v` divides by v once too often.** CSD is
already ∂²Vm/∂z², so for the engine's spatially-fixed IAP the potential is CV-independent;
the measured ratio to first principles is exactly 1/v (0.497 at v = 2, 0.249 at v = 4;
A0.2). A physiological CV dependence (Nandedkar & Stålberg 1983: amplitude ∝ 1/CV) should
come from stretching the IAP in space with v, not from a prefactor. Relevant to the open
amplitude audit and to the "MUAPs 40× too small" C-04 flag (a factor 3–4 of it).

**4. The golden recipe's SFAP *amplitude* on FEM φ is not yet trustworthy (±40 %),
its shape and spectrum are.** On FEM fields the spatial engine integrates φ'' faithfully
down to ~8 mm wavelengths, where FEM φ carries mesh structure; the 3-monopole fit does not
remove it (5/7 poles neither), and the result is an erratic FEM/analytical amplitude ratio
across depth (1.7× spread, in-band, different for σ = 5 and σ = 1 mm sources; A1.2), and
r ≈ 0.90–0.97 vs the analytical-φ pipeline for shallow fibres (A2.1; deep fibres 0.998).
Butterworth φ-smoothing (the Fourier route) makes the amplitude smooth but halves the
spectral content (FEM SFAP MNF 46 Hz vs 112 Hz golden, 91 Hz analytical; B11). The
monopole fit is translation-equivariant to 2e-3 (B8), fine for montages. A cleaner FEM φ (finer
mesh near the fibre, or a smoothing-free φ'' estimate such as a local polynomial fit)
is the fix; until then, FEM amplitude laws should be read through the smoothed route.

**5. The FEM cylinder decays ~30 % slower with depth than the analytical one over
7→20 mm** (A1.2, smooth 1.3–1.4× drift that survives every preprocessing and a σ = 1 mm
source), and at mid-depth (r = 27 mm) its φ'' matches the analytical one only to r = 0.93
whatever electrode size is assumed (A1.1b — the one honest tier-A fail). Not the
buried-source hypothesis; still to be attributed (mesh/skin-layer resolution vs the
analytical k-grid truncation at high spatial frequencies). The FEM electrode blob behaves as
a Ø5 mm disc (A1.1b at r = 33, B9).

**6. Phenomenology holds where the engine can be tested.** CV recovery within 5 % at
3/4/5 m/s; IZ symmetry, SD null (4 %) and phase reversal; EOF non-propagating (0.00 ms
spread), growing with depth, mono > SD > DD; power-law depth decay with SD steeper
(n 3.5 vs 2.9); fat curve within ×1.5 of Kuiken; electrode-size and IED laws; anisotropy
elongation √5 exactly in the infinite medium.

**7. Pool-level, two genuine model gaps.** The Gaussian renewal ISI has no refractory
floor (1.6 % of ISIs < 20 ms, min 9.8 ms; C2) — clip or use a gamma/lognormal draw. And
everything that depends on MUAP duration inherits C-04: cancellation saturates at 68 %
already at 20 % drive (C5), the EMG–force curve is slightly concave (C6), MDF is 61 Hz
(C7). These are gated with their real criteria and flagged, so they turn green with the
fibre-geometry fix.

---

## Next steps, in order

1. **Fix the 1/v amplitude constant** in `engines/spatial.py` (and re-pin the 21
   regression cases and the golden amplitude gate); decide whether CV should stretch the
   IAP.
2. **Audit the Fourier engine's conventions** against A0.3 (IAP orientation, sign, the
   window-centre shift); either fix it to match first principles or retire it as an oracle.
3. **FEM φ fidelity for the spatial recipe**: try a finer mesh around the fibre band and a
   smoothing-free φ'' (local quadratic fit over ~5 mm), re-run A1.2/A2.1/B8.
4. **Attribute the 30 % depth-law difference** between the FEM and analytical cylinders
   (mesh refinement study; analytical k-grid w = 512).
5. **Refractory floor** in `activation/pool.py`.
6. **Experimental confrontation** (BIBLIOGRAPHY Part III #12): spike-triggered surface
   MUAPs from the WR HD-sEMG units vs the MRI pipeline — nRMSE ≤ 20–30 %, xcorr ≥ 0.9
   (Lowery 2004; Botelho 2019); this is what the harmonic-fibre work feeds.
7. Promote tiers A–C to CI as slow tests once the FEM cache is a committed fixture
   (`_refactor/TODO.md` S8).
