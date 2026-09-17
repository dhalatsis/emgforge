# Which pipeline to use — the decision record

Read this first. At every fork in the chain (segmentation → mesh → lead field → SFAP →
MUAP → grid → activation) there is one route to use, a number that justifies it, and a
statement of when the alternative is legitimate. Every number below is measured, not
asserted; the command that re-measures the cheap forks live and re-reads the recorded
evidence for the expensive ones is

```
python scripts/sanity/route_audit.py        # ~5 s; writes _results/sanity/ROUTE_AUDIT.{md,json}
                                            # and copies the markdown to docs/validation/ROUTE_AUDIT.md
```

It exits non-zero when a shipped default disagrees with its verdict. The full tables with
every number are in [`validation/ROUTE_AUDIT.md`](validation/ROUTE_AUDIT.md); the
step-by-step justification of the synthesis recipe is
[`src/emgforge/synthesis/DIRECT_LINE_SOURCE.md`](../src/emgforge/synthesis/DIRECT_LINE_SOURCE.md).

**The short answer.** Run `python scripts/run_pipeline.py` and load its output with
`Simulator.from_pipeline(npz)`. The synthesis it performs is
`emgforge.synthesis.production_config()` — **the recipe** — fed to `field_to_muap`, which
is also what `field_to_muap(field, bed)` does with no config at all. Do not build a
`SpatialConfig` by hand; derive variants from `production_config()` with
`dataclasses.replace` so a deviation is a deliberate act.

## The defaults the code guards

`production_config()` is the single source of truth. The audit's *recipe guard* checks it
field by field against the documented recipe (no mismatch on 2026-09-17), the 21-case
regression (`tests/synthesis/test_muap_reference.py`) pins its output byte for byte, and
`run_pipeline.py`'s verify step shows the factored lead-field conditioning is bit-identical
to the unfactored recipe (`max |Δ| = 0.0 V` in both recorded runs).

| knob | default | where it lives | guarded by |
|---|---|---|---|
| synthesis engine | direct line-source (`SpatialConfig`) | `production_config()`; `field_to_muap(config=None)` | fork 1: r = 0.9999, lag −0.05 ms vs the closed-form oracle |
| φ conditioning | `denoise="monopole"`, `denoise_n_poles=3` | `production_config()`; `mri.pipeline.condition_leadfields(n_poles=3)` | fork 2 |
| current source | `csd_derivative=2` (triphasic i_m ∝ ∂²Vm/∂z²) | `production_config()` | A0.1, A0.5 (monopole-free 7.7e-17) |
| φ upsampling | `upsample_factor=2` (cubic) | `production_config()` | fork 4 |
| tendon window | `fiber_window="one_sided"`, `tukey_alpha=0.25` | `production_config()` | fork 3 |
| φ endpoint taper | `edge_taper_left=5`, `edge_taper_right=10` | `production_config()` | Gibbs guard; outside the fibre extent |
| time base | `center_time=False`, `t_start_ms=-10` | `production_config()` | fork 5 |
| amplitude constant | `sfap = (CSD @ φ)·dz·polarity`, **no 1/v** | `engines/spatial.py::compute_sfap_spatial` | fork 6: ratio 0.997 flat over v = 2–5 |
| conduction velocity | `v=4.0` m/s (PM regime 3.3) | `production_config(v=…)` | B2: CV recovered within 5 % at 3/4/5 |
| sampling | `fsamp=2048`, `w=256` on MRI; the cylinder harness uses 4096 | `production_config(fs=…)`; `scripts/validation/harness.py` | fork 4: invariant to < 1 % |
| polarity | `+1` | `production_config()` | B1: main lobe negative, + − + |
| electrode source width | `source_sigma=5.0` mm | `mri.pipeline.solve_grid_leadfields` | fork 7 (**use 1 mm for new solves**) |
| fibre bed | straight Poisson, 4 fibres/mm², 200 points/fibre | `mri.pipeline.poisson_bed` | fork 8 |
| innervation zone | `iz_frac=0.305 ± 0.02` (FCU) | `mri.pipeline.IZ_FRAC_FCU`, `IZ_JITTER` | study_fibres |
| electrode grid | 5×5, 10 mm IED, ray-cast onto the skin over the muscle | `mri.pipeline.grid_electrodes` | fork 9: 10.15 × 9.96 mm measured |
| activation | `MotoneuronPool` (5–40 pps, ISI CoV 0.167), `TwitchPool` | `emgforge.activation` | fork 11 |

## The forks, one verdict each

The live forks below run on the analytical Farina-2004 cylinder (fibre 10 mm below the
skin, NMJ 20 mm proximal of the electrode, tendons 60 mm either side of the electrode,
v = 4 m/s, fs = 4096 Hz) and on a closed-form line-source oracle (electrode 10 mm from an
infinite fibre in an anisotropic medium). Recorded forks quote
`_results/validation/{A,B,C,S}.json` and the paper key numbers
(`git show paper/arxiv:paper/figures/key_numbers_*.json`).

### 1. Synthesis engine — use the direct line-source (spatial) engine

Against the closed-form oracle the direct engine scores r = 0.9999 at −0.05 ms lag with an
amplitude ratio of 0.998, its source is monopole-free to 7.7e-17, and its end-of-fibre
potential starts at 10.26 ms for L/v = 10 ms. The Fourier engine on the same φ scores a
*signed* r of −0.68 at +2.55 ms lag (it matches a spatially mirrored IAP equally badly),
i.e. anti-phase and late. Its only legitimate use is as the shape oracle against the Farina
MATLAB reference set (r = 0.997, `tests/synthesis/test_golden.py`); never for physical time
or MRI. `production_config()` is a `SpatialConfig` and `field_to_muap(config=None)` picks
it — **PASS**.

### 2. Lead-field conditioning — use the 3-monopole fit

On the FEM cylinder φ at the same depth (mesh ripple included), measured against the
in-band content of the raw-φ SFAP (500 Hz low-pass, the reference tier A1.2 uses):

| variant | SFAP jaggedness | r(φ'') vs analytical φ'' | MNF retained | p2p retained | no-op on clean φ (r) |
|---|---|---|---|---|---|
| none | 0.0114 | 0.73 | ×1.22 | ×1.16 | 1.000 |
| butterworth(0.03) | 0.0009 | 0.93 | **×0.46** | **×0.23** | **0.77** |
| **monopole(3)** | 0.0022 | 0.91 | ×0.99 | ×0.98 | 0.9999 |
| monopole(5) / (7) | 0.0022 | 0.92 / 0.91 | ×0.99 / ×1.00 | ×0.98 / ×0.99 | 0.9999 / 1.000 |

The 3-pole fit is the analytic form the field actually has: it takes the ripple out of φ''
(0.73 → 0.91 against the analytical φ'') while keeping the spectrum and amplitude of the
in-band signal, and it is a no-op on a clean analytical φ. Five or seven poles buy nothing
at 5× the cost (35 → 190 ms per fit). Butterworth is legitimate **only** for *relative*
amplitude-vs-depth laws on FEM φ (A1.2: 1.22× spread across depth vs 1.69× for the direct
recipe); it destroys the absolute amplitude and halves the spectrum, and is not a no-op even
on the clean φ. Caveat that the fit does not fix (recorded A1.2 (ii)): on FEM φ the SFAP
*amplitude* is erratic at ±40 % across depth — mesh structure at ~8 mm wavelengths reaching
φ''; shape, timing and spectrum are faithful. **PASS**.

### 3. Tendon window — use `one_sided` (boxcar for sharp-tendon oracles)

| window | r vs sharp-tendon oracle | window at the NMJ | generation lobe vs boxcar | EOF / propagating (lit. 0.02–0.2) |
|---|---|---|---|---|
| boxcar | 0.9999 | 1.00 | 1.000 | 0.053 |
| tukey (α 0.25) | 0.9953 | **0.00** | **0.632** | 0.098 |
| hann | 0.9634 | **0.00** | **0.093** | 0.149 |
| **one_sided** (α 0.25) | 0.9964 | 1.00 | 1.000 | 0.061 |

A symmetric window is zero at the junction where the travelling wave is born, so tukey and
hann cut the generation lobe to 0.63× and 0.09× — an artefact, whatever their r. The
one-sided ramp tapers only the tendon end (its extinction starts 7.5 ms in, where the last
25 % of the 40 mm semi-fibre begins, versus 10.26 ms for the hard cut) and keeps the
EOF-to-propagating ratio inside the literature range. Use `boxcar` when comparing with a
sharp-tendon oracle (first principles, Farina) or to isolate the end-of-fibre potential.
**PASS**.

### 4. Sampling — fs 4096 on the cylinder, 2048 on MRI; `upsample_factor = 2`; w = 256

fs 2048 vs 4096 leaves the SFAP unchanged (r = 0.99935, amplitude 0.991 with dz = v/fs;
r = 1.00000 with the MRI's dz ≈ 1 mm), and w only sets the window length (first 256 samples
identical). The upsampling verdict is more modest than the docstring's "0.08 → 0.01": in
every production regime measured — analytical φ at dz 0.98/1.02/1.95 mm and the
monopole-fitted FEM φ at the MRI's dz 1.02 mm, fs 2048 (where dz ≠ v·dt) — ×1 is already
converged (jaggedness ×1/×2 = 1.00, ×2 ≡ ×4 to r ≥ 0.9994). On *raw* FEM φ upsampling
even raises the jaggedness (0.0098 → 0.0136) because the spline propagates the ripple into
φ''; the monopole fit is what removes it (→ 0.0043). So `upsample_factor = 2` is kept as
a cheap guard (+10–50 % per SFAP) for rougher φ, not as a load-bearing step; 4 buys nothing.
**PASS**.

### 5. Time base — physical time (`center_time = False`, `t_start_ms = −10`)

With t = 0 at the NMJ discharge the EOF onset lands at 10.26 ms (L/v = 10) and the
propagating lobe at 6.11 / 11.00 ms for electrodes 20 / 40 mm from the NMJ (Δz/v = 5 / 10
plus the lobe's own ≈1 ms delay; CV from the two lobes 4.10 m/s). `center_time = True`
shifts every timing by −w/(2 fs): EOF at −21 ms, lobes at −25 / −20 ms — the two timings
HD-EMG decomposition and CV estimation rely on, both broken. Use it only to overlay a
window-centred Fourier trace. **PASS**.

### 6. Amplitude constant — no 1/v

The engine/oracle amplitude ratio is 0.998 / 0.996 / 0.998 / 0.998 at v = 2 / 3 / 4 / 5 m/s
(spread 0.13 %). The pre-2026-09-15 `/ cfg.v` would give 0.499 / 0.332 / 0.249 / 0.200 —
exactly 1/v, every MUAP 4× too small at v = 4. The IAP is defined in space, so the
line-source integral is CV-independent; Nandedkar–Stålberg's amplitude ∝ 1/CV is for an IAP
fixed in time and would have to come from stretching the spatial IAP, never a prefactor.
The audit reads `compute_sfap_spatial`'s source to confirm the division is gone. **PASS**.

### 7. Volume conductor — analytical for laws, FEM/MRI for anatomy; σ_s = 1 mm for new solves

FEM reproduces the analytical cylinder φ(z) to r = 0.995–1.000 at five depths and
r = 0.997–1.000 at 10–45° off the electrode meridian (A1.1, A1.3), but it decays about 30 %
more slowly with depth (power-law exponent 3.17 vs 2.94; A1.2 (i)) — a volume-conductor
difference that survives every preprocessing and a σ_s = 1 mm source, still to be
attributed. The transverse footprint tells the source-width story: with the default
σ_s = 5 mm Gaussian electrode source the FEM FWHM across the fibre is 25.4 mm against
39.7 mm analytical (the blob reaches through 2 mm skin + 3 mm fat into the muscle;
max deviation of the peak profile 0.17), with σ_s = 1 mm the profile tracks the analytical
one (max deviation 0.06). **Use σ_s = 1 mm for new solves** (`solve_grid_leadfields(...,
source_sigma=1.0)`); the shipped default stays 5 mm **only** so that new numbers remain
comparable with the released datasets D1–D3 (`key_numbers_pipeline args.source_sigma = 5.0`).
Use the analytical cylinder for every law, threshold and oracle. **PASS (deliberate)**.

### 8. Fibre bed — straight Poisson bed; harmonic only with truncated streamlines fixed

The straight bed reproduces the released tensor bit for bit. On the 18 sampled units whose
harmonic streamlines all reach the innervation plane the two beds give the same MUAPs:
p2p ratio median 1.006 (q25–q75 0.96–1.09), r 0.985 (min 0.88), CV 4.28 vs 4.30 m/s, and
the harmonic bed is better contained (0.9995 vs 0.966). The harmonic bed's large
differences all come from the 16.3 % of streamlines (135 of 827) that never reach the plane
and get their NMJ at the fibre end: on those 6 units the p2p ratio is 2.6 (up to 18.9),
r 0.16, onset fraction 5.8 versus 0.18, CV unreadable. Pruning or re-innervating them (the
study's full-span variant) restores r 0.984 and onset 0.156. So: straight by default;
harmonic for anatomy **with** the truncated streamlines pruned or re-innervated;
series-fibered sampling remains experimental (no recorded numbers). **PASS**.

### 9. Electrode grid — the regular ray-cast grid; the legacy cache is deprecated

Measured live on the caches: the legacy vertex-snapped grid
(`_results/mu_pool/electrode_grid`, what `Simulator.from_mri` loads) has 23 unique
positions out of 25 (two duplicated electrodes), IED 25.4 mm along (20.7–30.9) and 5.2 mm
across (0.0–20.4), and its MUAP bank is 0.57 µV / 33.2 ms — the physiologically wrong
scale the sanity suite used to flag. The regular grid built by
`mri.pipeline.grid_electrodes` measures 10.15 × 9.96 mm (ranges 10.03–10.57 / 9.67–10.00),
its centre electrode sits on the skin 6.5 mm (xy) from the FCU centroid, and its MUAPs are
23.6 µV / 18.1 ms median over 100 units (S: "MUAP physiological scale", pass). `Simulator.from_mri`
now resolves, in order, an explicit `path=`, the largest (then newest) `run_pipeline.py`
output under `_results/pipeline/`, and only then the legacy tensor — with a `UserWarning`
that names the artefact and points to `scripts/run_pipeline.py`; `run_pipeline.py` itself
is correct. For the chain-level checks set `EMGFORGE_MUAP_BANK` to the released D2 file.
**PASS** (legacy path deprecated, reachable only with a warning).

### 10. Entry point — `run_pipeline.py` (cold) + `Simulator.from_pipeline`

Recorded cold runs with 2 workers: 20 units in 415 s (mesh 39, lead fields 12,
conditioning 221, MUAPs 128 s; 673 CPU-s) and 100 units in 993 s (42 / 18 / 253 / 659 s;
1796 CPU-s), and in both the runner's verify step finds the factored conditioning
bit-identical to the full recipe (`max |Δ| = 0.0 V`). The paper's `f2_common` route builds
the same 100-unit tensor in 1254 s wall / 12 530 CPU-s because it refits every shared fibre
per unit — legitimate only to regenerate the paper figures. `Simulator.from_mri` with no
arguments now loads the largest `run_pipeline.py` output and falls back to the legacy
42-unit tensor only with a warning (fork 9).
**PASS**.

### 11. Activation defaults — keep them; one open item

Onion-skin rate coding holds (Spearman(threshold, rate) = −0.98, 7.6–36.5 pps at half
drive), thresholds are right-skewed (+1.25) with a 95× twitch range, force is
MVC-calibrated (102.5 %MVC at full drive, monotone; RMS at 50 % MVC = 0.56 of RMS at
100 %). The one failing check is C2: the Gaussian renewal ISI has no refractory floor
(1.58 % of intervals under 20 ms, minimum 9.8 ms) — clip the ISI draw or use a
gamma/lognormal when sub-20 ms intervals matter. **PASS (open item C2)**.

## Open items the audit keeps visible

- FEM SFAP amplitude erratic at ±40 % across depth (A1.2 (ii)) and the 30 % slower FEM
  depth decay (A1.2 (i)) — volume-conductor findings, not the recipe's.
- The direct engine's end-of-fibre potential is φ(tendon)-weighted and under-produced
  against the Fourier `pare` EOF (`_results/sanity/README.md` [OPEN]).
- The legacy vertex-snapped tensor still exists under `_results/mu_pool/electrode_grid`;
  `Simulator.from_mri` reaches it only as a warned last resort (fork 9).
- No refractory floor in the ISI model (C2).
- Harmonic-bed truncated streamlines (16 %) need pruning or re-innervation before the
  harmonic bed can be a default; series-fibering is experimental.

## Provenance

Live numbers: `scripts/sanity/route_audit.py` on the analytical cylinder
(`tests/regression/analytical_ref.py`), the closed-form oracle and the FEM cylinder cache
(`_results/validation/fem_cache/cyl_lines.npz`, σ_s = 5 mm), reusing
`scripts/validation/harness.py`. Recorded numbers: `_results/validation/{A,B,C,S}.json`
(tiers A/B/C and the simulator sanity suite) and `paper/figures/key_numbers_{f1,f2,
study_fibres,study_crosstalk,pipeline}.json` read from branch `paper/arxiv` (they are not on
this branch). The generated report with every number is `docs/validation/ROUTE_AUDIT.md`
(a copy of `_results/sanity/ROUTE_AUDIT.md`; the JSON next to it has the raw values).
