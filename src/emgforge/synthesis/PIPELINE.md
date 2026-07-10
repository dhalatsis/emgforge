# MUAP Generation Pipeline

Fourier-domain SFAP/MUAP generation from reciprocal-field data, based on
Farina & Merletti (2004).

## Method Overview

The pipeline converts a spatial **reciprocal lead field** `phi(z)` (from FEM
or analytical models) into a Motor Unit Action Potential (MUAP) waveform in the
time domain. The key idea is that the reciprocal field encodes the electrode's
sensitivity pattern along the fiber axis, and multiplying by the intracellular
action potential (IAP) in the frequency domain gives the detected signal.

### Pipeline Steps

```
phi(z)  ──[smooth]──[upsample]──[resample]──[window]──[FFT]──> C(kz)
                                                                  │
C(kz) × pare(ka,kb) × exp(j·kz·posz) ─────────────────────────> E(kz,kt)
                                                                  │
E × spe2(kz) × (j·kz) × (1/v) ───────────────────────────────> E1(kz,kt)
                                                                  │
Radon section at z_det ────────────────────────────────────────> SFAP(t)
```

For a MUAP, repeat for each fiber and sum `E` before applying IAP and Radon.

### Step-by-step

1. **Preprocessing** (optional, in `api.py`):
   - **Smoothing**: Butterworth lowpass filter (cutoff=0.03, order=2) removes
     high-frequency noise from FEM-sampled lead fields.
   - **Edge taper** (`edge_taper`): Cosine-taper the first/last N samples of
     phi(z) to zero before resampling.  **Critical for MRI/FEM data** where
     the lead field is truncated (non-zero at measurement boundaries).  The
     zero-padding during resampling (e.g. 211→256 pts) creates step
     discontinuities that the `j·kz` spectral multiplication amplifies into
     persistent oscillations.  `edge_taper=10` eliminates this artifact.

2. **Resample** to the coupled grid spacing `dz = v * 1000 / fsamp` (mm).
   The Fourier grids couple time and space frequencies via conduction velocity,
   so the spatial step must match.

3. **Windowing** (optional): Hann window suppresses spectral leakage from
   truncated phi(z). For FEM fields that naturally decay to zero at boundaries,
   windowing is not needed (`apply_z_window=False`).  For truncated fields,
   use `edge_taper` instead — it only affects the edges, preserving the main
   signal (unlike Hann which tapers the entire signal and loses ~70% PTP).

4. **FFT**: Centered FFT gives `C(kz) = dz * FFTc{phi(z)}`, the spatial
   transfer function relating fiber activity to electrode potential.

5. **Fiber-end function** `pare(ka, kb)`: Models finite fiber length.
   Both L1 and L2 are **positive** semi-fiber lengths (MATLAB convention):

   ```
   pare = exp(-j*L1/2*ka) * L1 * sinc(L1/2*ka/pi)
        - exp(+j*L2/2*kb) * L2 * sinc(L2/2*kb/pi)
   ```

   where `ka = kz + kt/v` (propagating wave) and `kb = kz - kt/v` (counter-
   propagating wave). `L1 = len1_mm` (lower semi-length, positive),
   `L2 = len2_mm` (upper semi-length, positive).

   This matches the original Farina MATLAB code and the analytical
   `signal_generator.py` when called with positive L1 values.

6. **IAP spectrum** `spe2(kz)`: Rosenfalck intracellular action potential
   `V(z) = 96 * exp(-z) * (3z^2 - z^3)` for `z >= 0`, zero-padded and FFT'd.

7. **Assembly**: `E1 = (1/v) * spe2 * E * (j*kz)` where `(j*kz)` is the
   spatial derivative (converts IAP to current source density).

8. **Radon section**: Collapse 2D spectrum at detector position `z_det` to
   get the time-domain waveform.

## Parameters

### SFAPParams (fourier.py — low-level)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 4.0 | m/s | Conduction velocity |
| `fsamp` | 4096.0 | Hz | Sampling frequency |
| `w` | 256 | samples | Output length (time and frequency grid size) |
| `len1_mm` | 60.0 | mm | Semi-fiber length to lower end (positive; internally negated) |
| `len2_mm` | 60.0 | mm | Semi-fiber length to upper end (positive) |
| `posz_mm` | 0.0 | mm | Axial NMJ offset from lead-field center |
| `z_det_mm` | 0.0 | mm | Detector axial position for Radon section |
| `apply_z_window` | True | — | Apply Hann window before FFT |
| `enforce_delta_s_match` | True | — | Resample phi(z) to match coupled grid |
| `delta_s_tol` | 1e-6 | mm | Tolerance for spacing mismatch check |
| `edge_taper` | 0 | samples | Cosine-taper N samples at each edge to zero |

**Validated settings** for FEM lead fields (cylindrical, naturally zero at edges):
`apply_z_window=False`, `enforce_delta_s_match=True`.

**Validated settings** for MRI lead fields (truncated, non-zero at edges):
`apply_z_window=False`, `enforce_delta_s_match=True`, `edge_taper=10`.

### MUAPConfig (api.py — high-level)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `smoothing_method` | `'butterworth'` | `'butterworth'`, `'savgol'`, `'gaussian'`, `'none'` |
| `butterworth_cutoff` | 0.03 | Normalized cutoff frequency (0-1) |
| `butterworth_order` | 2 | Filter order |
| `savgol_window` | 21 | Savitzky-Golay window length (odd) |
| `savgol_polyorder` | 3 | Polynomial order |
| `gaussian_sigma` | 3.0 | Gaussian kernel std dev |
| `upsample_factor` | 1 | Cubic spline upsampling factor (1=off) |
| `edge_taper` | 0 | Cosine taper N edge samples to zero (10=MRI) |
| `v` | 4.0 | Conduction velocity (m/s) |
| `fsamp` | 4096.0 | Sampling frequency (Hz) |
| `w` | 256 | Output samples |
| `len1_mm` | 60.0 | Lower semi-fiber length (mm) |
| `len2_mm` | 60.0 | Upper semi-fiber length (mm) |
| `n_fibers` | 50 | Fibers to sample (NPZ workflow) |
| `r_min` | 10 | Min radius for fiber sampling (grid cells) |
| `r_max` | 35 | Max radius for fiber sampling (grid cells) |
| `seed` | 42 | Random seed |

### Preset Configurations

| Config | Smoothing | Window `w` | Edge taper | Use case |
|--------|-----------|------------|------------|----------|
| `get_optimal_config()` | BW c=0.03 o=2 | 256 (fixed) | 0 | Legacy default (back-compat) |
| `get_fast_config()` | Savgol w=21 | 256 (fixed) | 0 | Quick evaluation |
| `get_high_quality_config()` | BW c=0.03 o=2 | 256 (fixed) | 0 | Publication figures |
| `get_adaptive_config()` | BW c=0.03 o=2 | **adaptive [256, 1024]** | 0 | New code with FEM/analytical inputs |
| `get_truncated_input_config()` | BW c=0.03 o=2 | 256 (fixed) | 15 | When input φ is known-truncated |

#### When to use which

- **`get_optimal_config()`** — keep the legacy 256-wide window. Best when
  the caller has exactly 256 samples of φ at the matched `dz` and that's
  what they want to keep. Backwards-compatible with all existing callers.

- **`get_adaptive_config()`** — opt into the Workstream A heuristic.
  The pipeline picks `w` per input from `max(6·λ_decay, L_fibre + 30 mm)`,
  fitted from the input φ tail. The chosen `w` is capped at the input's
  own length when φ is NOT decayed at its boundary (going wider would
  zero-pad over a real-valued tail). Best when the caller supplies more
  than 256 samples of φ — e.g. a 400 mm FEM mesh, an analytical extract
  at w=1024, or an MRI volume.

- **`get_truncated_input_config()`** — last-resort crutch for inputs that
  ARE truncated mid-decay. The cosine edge_taper forces φ→0 over the
  outer 15 samples, suppressing Gibbs ringing. **Only use when the input
  is genuinely truncated** — applying this to clean analytical φ ruins
  long-fibre operator-consistency (regression bench A_longfib drops from
  r=0.99 → 0.12). See `tests/regression/WORKSTREAM_A_RESULTS.md`.

For MRI data, prefer `get_adaptive_config()` over `edge_taper=15` as the
first move — the adaptive window often makes tapering unnecessary by
choosing a `w` where φ has actually decayed. Fall back to
`get_truncated_input_config()` only when adaptive can't help.

## Derived Quantities

| Quantity | Formula | Default value |
|----------|---------|---------------|
| Spatial step | `dz = v * 1000 / fsamp` | 0.977 mm |
| Max spatial freq | `fm = fsamp / (2 * v * 1000)` | 0.512 /mm |
| Time window | `T = w / fsamp` | 62.5 ms |
| Spatial window | `Z = w * dz` | 250 mm |

## Usage Examples

### From voxel grid NPZ (neural network output)

```python
from muap_generator import generate_muap_from_npz, get_optimal_config

# Target (ground truth) MUAP
result = generate_muap_from_npz("sample.npz", n_fibers=50)

# Predicted MUAP
result_pred = generate_muap_from_npz("sample.npz", n_fibers=50, use_prediction=True)

# Compare
print(f"PtP target: {result.metrics['peak_to_peak']:.2e}")
print(f"PtP pred:   {result_pred.metrics['peak_to_peak']:.2e}")
```

### From phi(z) matrix

```python
from muap_generator import generate_muap_from_phi, MUAPConfig

config = MUAPConfig(
    smoothing_method='butterworth',
    butterworth_cutoff=0.1,
    upsample_factor=2,
    len1_mm=60.0,
    len2_mm=60.0,
)

# phi_mat: (Nfib, Nz) array of lead field lines
result = generate_muap_from_phi(phi_mat, dz_mm=0.977, config=config)
result.plot()
```

### Low-level single-fiber SFAP

```python
from muap_generator.fourier import SFAPParams, compute_sfap_from_phi_z

params = SFAPParams(
    v=4.0, fsamp=4096.0, w=256,
    len1_mm=60.0, len2_mm=60.0,
    posz_mm=0.0, z_det_mm=0.0,
    apply_z_window=False,
    enforce_delta_s_match=False,
)
t_ms, sfap, debug = compute_sfap_from_phi_z(
    phi_z, delta_s_mm=0.977, params=params,
)
```

### From FEM solver (training/generate_muap.py wrapper)

```python
from training.generate_muap import compute_muap_fourier

# z: fiber z-coordinates (mm), lead_field: FEM phi values at those coords
t_ms, muap = compute_muap_fourier(
    z, lead_field,
    conduction_velocity=4.0, fsamp=4096.0, w=256,
    len1_mm=60.0, len2_mm=60.0,
)
```

## Module Structure

```
muap_generator/
├── __init__.py              # Exports from api.py
├── api.py                   # Production API (smoothing + signed pare)
├── fourier.py               # Core Fourier pipeline
├── preprocessing.py         # Smoothing, upsampling, resampling
├── conventions.py           # Sign / timing conventions
├── adaptive_w.py            # Adaptive window-length selection
├── numerical.py             # Experimental time-domain method
├── PIPELINE.md              # This file
└── spatial_refactor/        # Spatial (time-domain) CSD@phi engine
```

## Validation Results

Validated against the analytical 4-layer cylindrical volume conductor model
(Farina & Merletti 2004) across fiber depths d = 11-31mm (muscle region).

| Pipeline | Mean r | Min r | r > 0.9 |
|----------|--------|-------|---------|
| **API (BW c=0.03 o=2, no window, no upsample)** | **0.990** | **0.942** | **21/21** |
| API (old: BW c=0.10 o=4, Hann, 2x upsample) | 0.894 | 0.761 | 11/21 |
| Fourier (raw, no smooth) | 0.821 | 0.669 | 8/21 |
| Legacy (unsigned pare) | 0.632 | — | 0/21 |

### MRI validation (edge_taper=10)

Validated on MRI-based forearm fiber potentials across 5 muscles × 6 electrodes
× 3 discretization levels (100K, 500K, 2M elements).

| Comparison | Mean r | Min r | Notes |
|------------|--------|-------|-------|
| 100K vs 2M (taper=10) | **0.9998** | 0.9997 | Per-muscle convergence |
| 100K vs 2M (taper=0) | 0.9983 | 0.9968 | Without taper |
| 500K vs 2M (taper=10) | **0.9999** | 0.9996 | Nearly identical |
| MRI roughness (taper=10) | **1.58** | — | Matches cylindrical ~1.7 |
| MRI roughness (taper=0) | 12.0 | — | Gibbs ringing from truncation |

### Sources of FEM-vs-analytical mismatch

The remaining ~5% correlation gap comes from physical model differences, not
pipeline bugs:

- **Geometry**: FEM uses finite cylinder; analytical assumes infinite
- **Source**: FEM uses Gaussian source (sigma=5mm); analytical uses point source
- **Layers**: FEM has 5 layers (cancellous+cortical bone); analytical has 4
- **Conductivities**: Slightly different tissue values

### Key fixes during validation

1. **`pare` sign convention** (resolved Apr 2026): The original Farina MATLAB
   code uses **positive** L1/L2 in the pare formula. An earlier fix incorrectly
   negated L1 (`signed_len1 = -len1_mm`), which was then validated against the
   analytical model called with negative L1=-60 — self-consistent but wrong
   convention. The correct pairing is: positive L1 in pare + positive L1 in
   MotorUnit (r=0.997 vs analytical with MATLAB convention). See
   `muap_smoothness/16_sign_convention_sort.py` for the definitive cross-test.
2. **Wrapper fiber lengths**: `compute_muap_fourier()` now accepts explicit
   `len1_mm`/`len2_mm` instead of deriving from z-array extents.
3. **`delta_s_tol`**: Changed from 1e-9 to 1e-6 to avoid spurious resampling
   from floating-point mismatch.

### Fiber-end effects

Single-fiber SFAPs show end-of-fiber spikes — these are physically real
(the AP abruptly terminates at the tendon). They are suppressed in realistic
MUAPs by: (1) multi-fiber averaging with tendon scatter (Ten1/Ten2),
(2) differential detection (SD/DD), (3) innervation zone scatter. See
`muap_smoothness/10_fiber_end_study.py` for the full investigation.

### Validation scripts

- `muap_smoothness/16_sign_convention_sort.py` — Definitive pare sign test
- `muap_smoothness/10_fiber_end_study.py` — Fiber-end spike investigation
- `muap_smoothness/03_analytical_comparison.py` — Filter optimization

## References

- Farina, D., Mesin, L., Martina, S., & Merletti, R. (2004). A Surface EMG
  Generation Model with Multilayer Cylindrical Description of the Volume
  Conductor. IEEE Trans. Biomed. Eng., 51(3), 415-426.
- Rosenfalck, P. (1969). Intra- and Extracellular Potential Fields of Active
  Nerve and Muscle Fibres. Acta Physiol. Scand., 321, 1-168.
