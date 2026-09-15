# The Golden MUAP Synthesis Method — Justification

A reference for *why* the production ("golden") single-fibre / MUAP synthesis recipe is
what it is, and the experiments and regression gates that back each choice. Distilled from
the code, tests, and the sanity/investigation suites. File paths are relative to the repo root.

---

## 0. The recipe, in one place

Production = the **spatial line-source engine** with this config
(`engines/spatial.py` `SpatialConfig`; the `spat_prod_*` cases in `tests/synthesis/muap_cases.py`):

```python
SpatialConfig(
    denoise="monopole", denoise_n_poles=3,   # denoise the FEM lead field before the CSD
    csd_derivative=2,                          # physical current-source density (triphasic)
    upsample_factor=2,                         # keep the 2nd-derivative integral off Nyquist
    fiber_window="one_sided",                  # taper the tendon end, stay flat at the NMJ
    edge_taper_left=5, edge_taper_right=10,    # Gibbs guard on the phi endpoints
    center_time=False, t_start_ms=-10.0,       # physical time: t=0 is the NMJ fire
    v=4.0, fsamp=2048.0, w=256, polarity=+1,   # (PM/MRI regime: v=3.3, fsamp=2048)
)
```

The whole engine is one identity — no FFT, no `radon`, no `np.flip` — so timing and the
fibre-end terminations are explicit in the code (`engines/spatial.py` header docstring):

```
SFAP(t) = INTEGRAL  phi(z) * i_m(z,t) dz,   i_m = sigma_in * pi * a^2 * d2Vm/dz2   ~=  (CSD @ phi) * dz * polarity
```

There is no `1/v` prefactor: the IAP is defined in space, so the line-source integral is
CV-independent (the former `/v` was removed 2026-09-15 — see §8 and validation check A0.2).

---

## 1. The physics — SFAP = ∫ CSD·φ

- **Rosenfalck IAP** `Vm(z) = A z^3 e^{-z}`, `A = 96e-3 V` (`iap.py:22,25-31`). The mV→V
  scaling (audit 2026-06-10) fixed a ~10^3× MUAP-amplitude mismatch vs the Neurodec ground
  truth. Onset is smooth (V, V′, V″ → 0), so there is no spurious wavefront source. One
  amplitude is shared by both engines on purpose.
- **CSD = ∂²Vm/∂z² (`csd_derivative=2`).** The source that drives the SFAP is the
  transmembrane current `i_m ∝ ∂²Vm/∂z²` — the physical current-source density, triphasic
  (+/−/+). It is taken numerically over the **full bidirectional Vm field**
  `Vm(v·t − |z − posz|)`, which is what captures both the NMJ-junction source (the `|·|`
  cusp) and the two tendon-end sources (the window edges) (`engines/spatial.py:104-108,
  194-202`). `csd_derivative=1` is the old biphasic `dVm/dz` behaviour, which **kills the
  trailing end-of-fibre lobe** — kept only as a regression contrast (`spat_csd1_biphasic`).
- Evidence: with the correct numerical 2nd-derivative construction the spatial engine scores
  **r ≈ 0.99–1.00** against the analytical cylinder; the earlier per-half analytic
  construction scored **r ≈ 0.37** (`scripts/synthesis/README.md:31-42`; `engines/README.md:88-93`).

---

## 2. Monopole denoising — the load-bearing FEM step

- **What it fits** (`preprocessing.py:_n_monopoles`): a small number (`n=3`) of free
  analytic point sources plus an offset,
  `phi(z) = Σ_i A_i / sqrt(d_i^2 + (z − z_i)^2) + c`, greedily fit pole-by-pole with
  `scipy.optimize.curve_fit` (`preprocessing.py:217-264`). It **replaces** the sampled φ
  with the smooth analytic fit.
- **Why** (`engines/spatial.py:80-85`, `preprocessing.py:224-235`): a FEM-sampled φ(z)
  carries mesh-scale ripple, and the CSD's 2nd derivative amplifies it into noise.
  Low-passing would also round the physical peak; fitting *the analytic form the field
  actually has* (a sum of monopoles) removes the ripple without that cost.
- **Where in the pipeline** (`engines/spatial.py:130-147`): monopole runs on the **raw φ,
  before** the endpoint edge-taper (a taper would corrupt the tails it fits against);
  Butterworth (the alternative / Fourier default) runs after.
- **Near-noop on clean φ, essential on FEM φ:** pinned by
  `tests/synthesis/test_field.py:166-186` (r>0.99 vs `denoise="none"` on a clean analytic
  field; a real change on a noisy field). Cases `spat_prod_clean` ("denoise near no-op") vs
  `spat_prod_noisy` ("the real use case", FEM-like ripple).

---

## 3. The rest of the preprocessing

- **`upsample_factor=2`** (`engines/spatial.py:73-77`): the 2nd-derivative CSD needs a finer
  grid than the raw φ or the SFAP develops a Nyquist zigzag. 2× cubic-spline upsampling
  drops jaggedness 0.08→0.01 and lifts r vs Neurodec 0.52→0.78. 2 is the sweet spot.
- **`fiber_window="one_sided"`** — tapers only the **tendon** end of each semi-fibre and
  stays **flat through the NMJ**. A symmetric window (tukey/hann) notches the junction
  "where the travelling wave is born" — an artifact. Dropping the one-sided window costs
  r 0.79–0.92 and 0.31× amplitude against the settled pipeline, so it is load-bearing, not
  cosmetic (`preprocessing.py:121-164`; `tests/synthesis/test_windows.py`).

  | window | tendon edge | EOF lobe | NMJ junction | use |
  |---|---|---|---|---|
  | `boxcar` | hard cut | full-strength + ringing | flat | diagnostic / "see the EOF" |
  | `tukey` (α=0.25) | symmetric taper | softened | **notches junction (artifact)** | legacy default |
  | `hann` | full Hann per half | strong taper | notches junction | reference only |
  | `one_sided` (α=0.25) | tendon only | softened | **flat** | **production** |

- **`edge_taper_left=5 / right=10`** (`engines/spatial.py:70-71,137-141`): a cosine ramp on
  the φ *endpoints* — a Gibbs/truncation guard, separate from the fibre-end window,
  asymmetric because φ truncation differs at the two ends.

---

## 4. Physical time and the travelling lobe

- **t=0 is the NMJ fire** (`center_time=False`). The proximal end-of-fibre lands at `len1/v`,
  the distal at `len2/v`; the wave leaves `posz` at `posz ± v·t` — timing is in the kernel,
  not post-hoc (`engines/spatial.py:110-123,196,236,241`). `t_start_ms=-10` only prepends
  baseline (the signal is genuinely 0 for t<0), so it never changes the waveform.
- **Verified:** `_results/sanity/physical_time/verify_physical_time.py` (Fourier P1/P2 peak
  at the window centre ~31 ms vs the physical-time engine with the NMJ at 0 and the EOF at
  `L/v`). The waterfall sweeps (`_results/sanity/spatial_vs_fourier/build_p3_boxcar_waterfalls.py`,
  `build_sfap_waterfalls.py`) show the detector lobe walking later with electrode distance
  (`|Δz|/v`) while the EOF stays pinned at `L/v` — the classic propagation/EOF check.
  Reproduced across three geometry tiers: cylinder (`cylinder_baseline/`, r≈0.99–1.00 vs the
  analytical model), ellipse (`ellipse_baseline/`), real MRI forearm
  (`mri/fem_baseline/build_mri_longitudinal_waterfall.py`, EOF at `L/v=15 ms`).
- **Conduction velocity** is the first-class parameter `v` (mm/ms ≡ m/s); cylinder tier uses
  4.0, PM/MRI 3.3 (`spat_pm_velocity` pins that regime). Because the lobe timing is `Δz/v`
  and reproduces the analytical model at r≈0.99–1.00, a physiological CV emerges by
  construction rather than being imposed downstream.

---

## 5. End-of-fibre (EOF)

- The EOF terminal potential is produced by `csd_derivative=2` (`=1` kills it) acting on a
  windowed field: a hard/`boxcar` tendon edge is a step, so `∂²/∂z²` gives a sharp tendon
  source. `one_sided` softens it while preserving the propagating physics. Both are run from
  the same solves in the MRI waterfall so the softening is directly visible.
- **Open finding:** the spatial EOF is `φ(tendon)`-weighted (local) and under-produced vs the
  Fourier `pare` EOF (global). `_results/sanity/spatial_vs_fourier/_diag_eof.py` shows the
  spatial EOF appears only where φ(tendon) is non-trivial (L=20/40/60 sweep). Tracked
  `[OPEN]` in `_results/sanity/README.md`. In coherent multi-fibre sums (strong MU) the EOF
  reappears; weak/deep MUs and tendon scatter suppress it (`mri/neurodec_baseline/build_pm_muap_strong_weak.py`).

---

## 6. Validation — the golden gate

- **Oracle:** the **Fourier engine**, scored **r ≈ 0.997 against a Farina (2004) MATLAB
  reference** (`test_golden.py:2-6`; `scripts/synthesis/build_golden.py`). The golden set is
  12 clean, decaying cylinder φ (Gaussian and difference-of-Gaussian) plus the Fourier SFAP
  each produces. `analytical_phi_along_fibre` is deliberately excluded (its extracted φ is
  inconsistent with the pipeline and does not decay at the FFT window edges).
- **Operator-consistency gate** (`test_golden.py`): the spatial engine on the same 12 φ,
  under a *matched* config, agrees on **shape at |r| ≥ 0.95** (mean 0.997). It asserts the
  two engines compute the same integral, not that either is correct against physics.
- **Two disagreements are deliberately pinned** (so a change is a deliberate act, not
  silent drift): anti-phase **polarity** (signed r ≤ −0.95) and a stable **~−2 ms lag**
  (physical- vs window-centred time). The **amplitude ratio** is pinned at **0.80–1.05**
  (measured 0.84–1.01, mean 0.95, CV-independent: the same σ = 8 mm Gaussian field at
  v = 3/4/5 m/s gives 0.951/0.962/0.963). Before the 2026-09-15 `1/v` fix it was 0.15–0.35 and *looked*
  non-constant (1.65× spread) — that spread was the 1/v factor across the set's v = 3/3.3/4/5.
- **21 self-regression cases** (`tests/synthesis/muap_cases.py`), pure NumPy, exercising every
  knob a refactor could disturb: the four windows, `denoise` none vs monopole,
  `csd_derivative` 1 vs 2, cylinder vs PM regime, asymmetric semi-lengths, NMJ offset,
  polarity flip, multi-fibre NMJ scatter, and the full production recipe on clean vs
  FEM-like-noisy φ. Two-tier gate (`test_muap_reference.py`): byte-identical
  (`np.array_equal`, the strict canary) + signed correlation ≥ 1−1e-9 / amplitude within
  1e-9 (the semantic gate), plus guards that the cases stay distinct and non-trivial.

---

## 7. Why the spatial engine carries production (not Fourier)

The Fourier engine is the *shape* oracle, but the spatial engine is **physical-time native**:
`posz` is a true time-of-flight, so per-fibre NMJ scatter produces real temporal dispersion
across a fibre bed (`spat_multi_nmj`, ±12 mm), whereas the Fourier output is window-centred
and `posz` "barely shifts a fibre in time" (`four_multi_posz`). MRI/HD-EMG needs that real
dispersion, and the monopole-denoise + one-sided-taper recipe is what makes the physical-time
engine robust on FEM-sampled φ. That is the whole reason the recipe exists.

---

## 8. Open items (kept honest)

> **2026-09-10 validation suite** (`docs/validation/PLAN.md`, findings 1–5) settles several of
> these: the spatial engine reproduces a closed-form line-source oracle at r = 1.0000 with zero
> lag and a monopole-free source, so the pinned polarity/lag disagreements below are the
> **Fourier engine's** (r = −0.64…−0.79 vs first principles; the Farina port's propagating main
> lobe is positive, i.e. inverted vs the textbook). Two new items on the spatial side: the
> amplitude constant divided by `v` once too often (`compute_sfap_spatial`, ratio to first
> principles = 1/v exactly — **fixed 2026-09-15**, see below), and on FEM φ the SFAP
> *amplitude* is erratic at ±40 % across depth (mesh structure reaching φ''; the 3-monopole
> fit does not remove it) while shape, timing and spectrum are faithful.

- **Amplitude constant — FIXED 2026-09-15.** `compute_sfap_spatial` computed
  `(CSD @ φ)·dz·polarity / v`, but the CSD is already the physical
  `σ_in·π·a²·∂²Vm/∂z²` of a *spatially*-defined IAP, so the integral is CV-independent
  and the `/v` was spurious. Validation check A0.2 (engine vs the closed-form line-source
  oracle, `scripts/validation/harness.py::sfap_first_principles`) measured an amplitude
  ratio of exactly 1/v before the fix (0.497 at v = 2, 0.249 at v = 4) and 1.00 at both
  velocities after it. Every spatial MUAP is now larger by exactly v (×4.0 cylinder regime,
  ×3.3 PM regime) with a bit-identical peak-normalised waveform; the 21-case reference was
  regenerated and the golden amplitude gate re-pinned (0.15–0.35 → 0.80–1.05). The
  Nandedkar & Stålberg "amplitude ∝ 1/CV" law holds for an IAP fixed in *time*; whether CV
  should stretch this engine's spatial IAP is a separate, still-open design decision — no
  CV-dependent stretching was added.
- **Amplitude, spatial vs Fourier:** with the constant fixed the two engines agree to
  0.84–1.01 (mean 0.95) on the golden set, CV-independent. The residual ≤ 16 % is the
  window/φ-handling difference between the methods (`test_golden.py`;
  `tests/regression/SHAPE_VS_AMPLITUDE.md`).
- **Sign:** which engine carries the correct polarity is an open physics question, pinned not
  resolved.
- **Spatial EOF** under-produced vs Fourier (φ(tendon)-weighted + numerically smeared) — open.
- **~8–9 ms constant offset** vs the Neurodec ground truth, attributed to a timing-reference
  convention difference on their side (offset is independent of Δz), not a physics error here
  (`_investigations/over_nmj_muap_discrepancy/SUMMARY.md`) — open.
- **Amplitude gap (C-04):** with the current straight, full-length fibre geometry the
  synthesised fields are still broader/weaker than a real motor unit; this is the fibre-length
  problem the experimental series-fibered sampling is meant to close (see the harmonic-fibre
  work).
- Housekeeping: the committed golden `.npz` still embeds the pre-rename module name
  `muap_generator.fourier`; regenerating the set refreshes it (r=0.997 unaffected).

---

## Evidence map

| Claim | Where it is shown |
|---|---|
| CSD 2nd-deriv correct (r≈0.99–1.00) | `scripts/synthesis/README.md`; `engines/README.md` |
| Monopole denoise near-noop clean / real on FEM | `tests/synthesis/test_field.py:166-186`; cases `spat_prod_clean/noisy` |
| One-sided window load-bearing | `tests/synthesis/test_windows.py`; `preprocessing.py:121-164` |
| Physical-time propagation, EOF at L/v | `_results/sanity/physical_time/`, `.../spatial_vs_fourier/` waterfalls |
| Cross-geometry accuracy | `_results/sanity/{cylinder,ellipse,mri}_baseline/` |
| Spatial EOF φ-weighted (open) | `_results/sanity/spatial_vs_fourier/_diag_eof.py` |
| Fourier oracle r=0.997 vs MATLAB | `test_golden.py`; `scripts/synthesis/build_golden.py` |
| Engine agreement (shape) + pinned disagreements | `tests/synthesis/test_golden.py` |
| 21-case byte-identical regression | `tests/synthesis/muap_cases.py`; `test_muap_reference.py` |
| Physical-time > Fourier for NMJ dispersion | `spat_multi_nmj` vs `four_multi_posz` |
