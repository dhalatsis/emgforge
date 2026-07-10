# verification — golden cylindrical regression set

The **fixed reference the spatial engine must always reproduce.** Run after any
change to `spatial_sfap.py`:

```bash
pytest tests/synthesis                     # the gate (60 tests, ~1.5 s)
```

## What it is

`golden_cylindrical.npz` (regenerable via `build_golden.py`, ~48 KB, deterministic)
holds 12 cases. Each is
a clean **cylinder-like lead field** φ(z) + the SFAP produced by the **validated
Fourier pipeline** (`muap_generator.fourier`, Farina 2004, r=0.997 vs MATLAB) on
that same φ. A correct spatial (time-domain) engine must match each golden SFAP
at **r ≥ 0.95**, because both compute the same line-source integral
`∫ φ(z) ∂²Vm/∂z²(z−vt) dz` — one in the frequency domain, one in space/time.

Cases span: monophasic Gaussian widths (depth proxy σ=5–18 mm), biphasic
difference-of-Gaussians (realistic layered-cylinder lead-field shape), symmetric
and asymmetric L1/L2 (incl. the PM 64/144), conduction velocities 3/4/5, and NMJ
offsets posz = 0/+20/−30 (electrode-over-NMJ and the realistic offset case).

φ are clean synthetic fields that **decay to ~0 at the window edges**, so the
Fourier reference is artifact-free. (The analytical `analytical_phi_along_fibre`
extraction is deliberately NOT used — it is mutually inconsistent with the Fourier
pipeline and does not decay at the edges; that is a φ-extraction issue, not an
engine issue. Realistic FEM-φ validation is the MRI tier's separate job.)

## Status (2026-06-19)

After fixing `build_csd_matrix` (full bidirectional Vm field → numerical
`∂²/∂z²`, capturing the NMJ-junction and tendon-end sources):

```
mean r = +0.997   median = +0.997   PASS 12/12   (threshold 0.95)
```

The previous (per-half analytic, opposite-sign) construction scored mean r≈0.37
(PASS 1/12) — that was the bug the cylindrical recheck uncovered.

## Plots

`figures/golden_vs_engine.png` — all 12 cases: golden (Fourier, black) vs the
fixed engine (red), normalised + peak-aligned, r in each title. They overlay.

![golden_vs_engine](figures/golden_vs_engine.png)

`figures/before_after_fix.png` — 3 cases showing golden vs the **buggy** engine
(orange, per-half/subtract) vs the **fixed** engine (red, full-field numerical
∂²/∂z²). The buggy curve is narrow/wrong; the fixed curve tracks the golden.

![before_after_fix](figures/before_after_fix.png)

## Files
- `build_golden.py` — regenerates `golden_cylindrical.npz` (deterministic).
- `golden_cylindrical.npz` — the frozen fixture (committed).
- The gate now lives in `tests/synthesis/test_golden.py` (`pytest tests/synthesis`). It asserts signed r, alignment lag, and peak amplitude — the retired `verify_spatial.py` scored only `abs(r)` on peak-normalised traces.
- `plot_golden.py` — renders `figures/golden_vs_engine.png`.
