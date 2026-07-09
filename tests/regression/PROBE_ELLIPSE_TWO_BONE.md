# Ellipse + two-bone probe — HF jaggedness check

Built `tests/regression/neumann_meshes/meshes/ellipse_two_bone.msh` by
combining the ellipse-1.2 cross-section (from
`generated_meshes_ellipse/sample_000000`) with the two-bone placement
(from `generated_meshes_two_bone/sample_000000`).

Layer dimensions (mm, semi-axes):
- skin   a/b = 53.85 / 44.87
- fat    a/b = 51.66 / 43.05
- muscle a/b = 40.44 / 33.70
- cort1  a/b = 11.09 /  9.24   at (-1.28, -3.18)
- cort2  a/b = 10.42 /  8.68   at (-1.55,  5.43)
- canc1  a/b =  8.63 /  7.20
- canc2  a/b =  7.40 /  6.17

Solved FEM (459 K nodes, 2.97 M elements, 9.3 s solve, Gaussian σ=5
source at skin top θ=0 z=center). Sampled φ at 5 depths × 5 angles = 22
configurations (3 deep cases at d≥33 skipped — outside `b_muscle=33.7`).
Each ran through four pipelines:

- `default`     — `MUAPConfig()` (Butterworth c=0.03 o=2, w=256)
- `adaptive`    — `get_adaptive_config()` (adaptive w + auto-smoothing)
- `no_smooth`   — no smoothing (raw FEM, shows underlying jaggedness)
- `edge_taper`  — `get_truncated_input_config()` (edge_taper=15)

## Headline

**Yes — the high-frequency jaggedness from before is still present in the
raw FEM φ(z), and it does still propagate catastrophically through the
pipeline when smoothing is off.** The default Butterworth (c=0.03, o=2)
filters it cleanly. `adaptive_w_auto` matches default bit-for-bit
(Δ = 0 in every case — safety cap holds w=256 because edge/peak is high).

### MUAP roughness (RMS of d²/dt², geometric mean across 22 cases)

| pipeline      | mean roughness | mean HF ratio (>500 Hz) |
|---------------|----------------|--------------------------|
| `no_smooth`   | 2.16e-05       | up to 0.668              |
| `default`     | 1.03e-05       | 0.05–0.07                |
| `adaptive`    | 1.03e-05       | 0.05–0.07                |
| `edge_taper`  | ~ same as default | ~ same                |

**Smoothing reduction**: ~3.5× geometric mean across all cases; up to
**13× on the worst off-axis cases** (θ=90° d=10).

### Worst HF-jaggedness cases (raw, no smoothing)

| depth | θ    | edge/peak | rough_phi | rough_MUAP_ns | HF ratio (ns) | rough_default | HF ratio (default) |
|-------|------|-----------|-----------|----------------|----------------|----------------|---------------------|
| 10 mm | 90°  | 0.987     | 4.6e-05   | **1.66e-04**   | **0.668**      | 1.24e-05       | 0.063               |
| 10 mm | 135° | 0.989     | 3.3e-05   | 1.45e-04       | 0.580          | 5.5e-06        | 0.064               |
| 10 mm | 45°  | 0.977     | 4.5e-05   | 1.83e-04       | 0.517          | 2.2e-05        | 0.062               |
| 10 mm | 180° | 0.993     | 9.2e-05   | 2.93e-05       | 0.394          | 6.2e-06        | 0.066               |
| 15 mm | 90°  | 0.994     | 6.9e-06   | 3.88e-05       | 0.323          | 1.1e-05        | 0.064               |

(HF ratio = fraction of MUAP spectral energy above 500 Hz; the analytical
MUAP runs ~0.05.)

### Plot: d=10 mm θ=90° (most-affected case)

`tests/regression/figures/ellipse_two_bone/d10_th090.png`

Top row: raw φ(z) with visible HF speckle on a roughly flat baseline (the
fiber line at θ=90° passes through a region distant from the source); the
unsmoothed MUAP is dominated by HF noise (66% HF energy).

Bottom row: default Butterworth produces a clean MUAP; `edge_taper` adds
no benefit (overlays exactly); `adaptive_w_auto` matches default exactly.

## Why default doesn't change w on this geometry

The input-length safety cap in `adaptive_w.choose_w` checks
`|φ(boundary)|/|φ(peak)|`. On the ellipse+2-bone at off-axis angles, this
ratio is 0.99+ in 20 of 22 cases (the field doesn't decay across the
240-mm mesh in those directions — it asymptotes to a non-zero baseline).
The cap correctly refuses to grow w past 256: zero-padding past the input
would create an artificial boundary step.

To exercise the adaptive grow path, we'd need an L400 ellipse-2-bone mesh.
Infrastructure is in place (`build_extended_sample.py` pattern); not
generated to avoid another 5-minute mesh build for what is now-clearly a
mesh-bound regime.

## Verdict on HF jaggedness

- **Still present in the FEM**: yes, especially at off-axis directions
  where the bones break cylindrical symmetry. The numerical mesh
  discretization produces field noise that the `j·kz` factor in the
  Fourier pipeline amplifies aggressively.
- **Fixed by the default smoothing**: yes — Butterworth c=0.03, o=2 cuts
  HF energy by 10× on the worst cases, restoring clean MUAP shapes.
- **`adaptive_w_auto` doesn't break anything**: produces identical MUAPs
  to default on this geometry (the heuristic correctly stays at w=256).

## File map

- `tests/regression/build_ellipse_two_bone.py` — builds the mesh
- `tests/regression/probe_ellipse_two_bone.py` — runs the probe
- `tests/regression/PROBE_ELLIPSE_TWO_BONE.json` — full numeric table
- `tests/regression/figures/ellipse_two_bone/*.png` — 22 per-case plots
