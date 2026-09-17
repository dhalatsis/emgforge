# `spatial_refactor` — time-domain MUAP engine

A clean, FFT-free **spatial / time-domain** SFAP→MUAP engine — an alternative to
the production **Fourier** pipeline (`muap_generator/fourier.py`, Farina 2004 with
`pare`/`radon_section`). Where the Fourier engine works in the `(kz, kt)` frequency
domain, this one integrates the line source directly in physical time, so the
travelling-wave sign and the fibre-end termination are *visible* in the code.

## The method in one line

```
SFAP(t) = ∫ φ(z) · i_m(z, t) dz,  i_m = σ_in · π · a² · ∂²Vm/∂z²   ≈   (CSD @ φ) · dz · polarity
```

No `1/v` prefactor: the IAP is defined in space (`Vm(v·t − |z − z₀|)`), so the
line-source integral is CV-independent (validation check A0.2 — the former `/v`
made the amplitude exactly 1/v of the closed-form oracle; fixed 2026-09-15).

- `φ(z)` — the lead field sampled along the fibre (reciprocity: potential from a
  source at the **electrode**). Comes from the analytical cylinder or an FEM solve
  along a real fibre path. The engine doesn't care which.
- `CSD(z, t)` — current-source density of the travelling intracellular AP: the
  **numerical `∂²/∂z²`** of the full bidirectional Rosenfalck Vm field
  `Vm_iap(v·t − |z − z₀|)`, windowed at the tendons (captures the NMJ-junction and
  tendon-end sources).

No FFT, no `pare`, no `radon`, no `np.flip`.

## The CSD construction

The current-source density that drives the SFAP is `i_m ∝ ∂²Vm/∂z²`. The engine
builds the **full bidirectional Vm field** `Vm_iap(v·t − |z − z₀|)`, windows it to
the fibre extent, and takes its **numerical** `∂²/∂z²` — capturing the NMJ-junction
source (the `|·|` cusp) and the tendon-end sources (the window edges).
`SpatialConfig.csd_derivative` (default 2) sets the derivative order;
`upsample_factor` (default 2) keeps the numerical derivative stable.

The engine is anchored to a cylindrical reference set (12 cases, mean
**r = 0.997** vs the validated Fourier pipeline). Regenerate it with
`scripts/synthesis/build_golden.py`, then re-check the engine with
`pytest tests/synthesis` after any change. The production recipe built on this
engine — direct line-source synthesis — is documented in
[`../DIRECT_LINE_SOURCE.md`](../DIRECT_LINE_SOURCE.md).

## Spatial vs Fourier — which to use

On clean analytical/cylinder φ the two engines agree at **r ≈ 0.997** (12/12 reference
cases): they compute the same line-source integral in different domains. On the
array-wide MRI target the **Fourier engine is the more robust method** (median raw
Pearson r ≈ 0.55 vs ≈ 0.08 for spatial; aligned r > 0.5 on 98/105 vs 75/105
electrodes). Rule of thumb: use **Fourier for array-wide** synthesis; reach for the
spatial engine as an independent cross-check, or when you want explicit
physical-time propagation (`center_time=False`: t = 0 is the NMJ fire, proximal EOF
at `len1/v`).

## Files

| Path | What |
|---|---|
| `spatial.py` | The engine: `build_csd_matrix`, `compute_sfap_spatial` (single fibre), `SpatialConfig`. The Rosenfalck IAP is in `emgforge.synthesis.iap`; the multi-fibre sum in `emgforge.synthesis.field_to_muap`. |
| `fourier.py` | `build_pare` → E1 → radon (Farina 2004), window-centred. |
| `__init__.py` | Public exports |
| `scripts/synthesis/` | Reference-set builder (`build_golden.py`), the MUAP-reference builder (`build_muap_reference.py`), a plotting companion (`plot_golden.py`). The operator-consistency gate vs Fourier is `tests/synthesis/test_golden.py` |

Waveform-comparison metrics (`align_score`, `nrmse_aligned`, `lobe_metrics`,
`jaggedness`) live in `emgforge.synthesis.metrics`.

The `.npz` fixtures (the cylindrical reference set and per-tier datasets) are **not shipped** — they
are regenerable from the builder scripts above.

## Usage

```python
from emgforge.synthesis import production_config
from emgforge.synthesis.engines.spatial import compute_sfap_spatial

# single fibre — the engine's boundary (φ passed in, no summation). config=None is
# production_config(); pass production_config(v=3.26) etc. to set the regime.
t, sfap, dbg = compute_sfap_spatial(phi_z, dz_mm, len1_mm=64, len2_mm=144,
                                    posz_mm=0.0, config=production_config(v=3.26))

# a motor unit — sum over a FibreBed via the unified entry (engine chosen by
# config type; None = the production recipe). `field` is φ(z) per fibre; `bed` is
# the geometry + conduction.
from emgforge.synthesis import field_to_muap, FibreBed
bed = FibreBed.from_arrays(dz_mm, len1_mm=Lprox, len2_mm=Ldist, posz_mm=0.0, v=vs)
res = field_to_muap(field, bed)                        # res.muap, res.time_convention="physical"
```

`production_config()` is the single definition of the validated recipe (monopole
denoise, one-sided tendon window, `csd_derivative=2`, 2× upsampling, physical time
from −10 ms). A bare `SpatialConfig()` (`fiber_window='tukey'`, Butterworth
denoise, `t_start_ms=0`) is **not** the recipe — its defaults are frozen only
because the reference sets were recorded with them.

## Provenance

The spatial line-source method originated in late-2024 exploratory work, matured
into a `MotorUnit` class in early 2025, and was recovered and packaged here as a
clean engine wired into `muap_generator`. The key fix in this revival: the CSD is
built from the **numerical 2nd derivative of the full bidirectional Vm field**
(capturing the junction and tendon sources), not a per-half analytic 2nd derivative
— the latter misses those sources and scored only r ≈ 0.37 against the reference set.
