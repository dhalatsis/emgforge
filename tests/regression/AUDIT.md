# Integration-points audit — FEM → MUAP pipeline

Audit for Step 1.1 of the FEM worker integration plan. Maps where each
quantity that the integration plan needs to touch is currently defined.

## 1. Where `w` (pipeline window) is hardcoded

| File | Line | Symbol | Notes |
|------|------|--------|-------|
| `muap_generator/api.py` | 57 | `MUAPConfig.w = 256` | Top-level config dataclass; default flows into `_compute_muap_core`. |
| `muap_generator/fourier.py` | 174 | `SFAPParams.w = 256` | Used by `compute_sfap_from_phi_z` and `compute_muap_from_phi_matrix`. |
| `muap_generator/numerical.py` | 70 | numerical pipeline default | Older time-domain pipeline; rarely used. |
| `muap_generator/reciprocal_field_pipeline.py` | 213 | LEGACY | Not in production. |
| `muap_generator/muap_optimized.py` | 241 | LEGACY | Stale copy. |

The production code paths that read `w`:
- `_compute_muap_core` at `muap_generator/api.py:121` — pulls `config.w`,
  feeds `build_fourier_grids(w=…)`, `build_spe2_iap_spectrum(w=…)`,
  `resample_centered_line(w_out=w, …)`.
- `compute_sfap_from_phi_z` at `muap_generator/fourier.py:209` — same.
- `compute_muap_from_phi_matrix` at `muap_generator/fourier.py:301` — same.

Workstream A will let `w=None` trigger an adaptive heuristic at these three
entry points; an explicit integer preserves the old behaviour exactly.

## 2. Where φ(z) is sampled along the fibre

φ(z) comes from one of three places:

1. **Live FEM eval** — `FEMModel.evaluate_solution_at_points(points)` at
   `src/emgop/fem/solver.py:163`. Caller assembles a `(W, 3)` array of
   `(x, y, z_grid)` points on the fibre line. See
   `muap_smoothness/03_analytical_comparison.py:174-186` for the canonical
   pattern: `fiber_pts[:,2] = np.linspace(z_center − z_half, z_center + z_half, W)`
   then clipped to `[0.5, length − 0.5]`. **The number of samples equals `W`**
   which is what couples this part to the hardcoded `w=256`.

2. **NPZ dataset loading** — `extract_phi_lines(field, xs, ys)` at
   `muap_generator/preprocessing.py:211`. Pulls `field[:, y, x]` from a
   pre-voxelised `(Z, Y, X)` array. Number of z-samples = `Z` of the voxel grid.

3. **Voxel NPZ files** — `generate_muap_from_npz` at `muap_generator/api.py:249`
   loads `target[0,0]` (or `prediction[0,0]`) plus `spacing` to recover `dz`.

The fibre sampling is decoupled from the pipeline `w` in (2) and (3) because
`resample_centered_line` rescales — but (1) is tightly coupled.

## 3. Where the FEM Laplace BC is set

`src/emgop/fem/solver.py:188-217` in `FEMModel.solve_for_point`:

```python
g = fem.Constant(self.mesh, default_scalar_type(self.options["boundary_value"]))
...
L = self.source_function * v * dx + g * v * ds
```

With `boundary_value=0` (default at `src/emgop/fem/solver.py:66`), the
weak form's boundary term is identically zero, which is exactly the
homogeneous Neumann condition `∂u/∂n = 0` on all of `∂Ω`. There is no
Dirichlet constraint; the constant nullspace is removed explicitly via
`PETSc.NullSpace().create(constant=True)` in `ConstrainedLinearProblem.solve`
(`solver.py:47-58`).

`ElectrodeFEMSolver` (`src/emgop/fem/electrode_configs.py:350`) wraps
`FEMModel`. In `volumetric` return mode it just calls
`FEMModel.solve_for_point` (same Neumann BC). In `localized` return mode
(bipolar), `_solve_with_localized_ground` adds a second point sink — the
boundary condition is still homogeneous Neumann; only the source term changes.

Workstream B will paired-solve at multiple mesh extents to measure how much
the Neumann reflection contaminates φ at the fibre. Workstream C, if
triggered, will add an alternative BC flag here.

## 4. Public call-sites of the production MUAP entry points

Production functions (the ones we must not break the signature of):

- `muap_generator.api.generate_muap_from_phi` — used by `generate_muap_from_npz`,
  the smoothness study (`muap_smoothness/06_api_validation.py`), the
  parameter study scratchpads, and notebooks.
- `muap_generator.api.generate_muap_from_npz` — convenience wrapper.
- `muap_generator.fourier.compute_sfap_from_phi_z` — used by every
  `muap_smoothness/*` study and `muap_parameter_study/*`.
- `muap_generator.fourier.compute_muap_from_phi_matrix` — used internally.

`MUAPConfig` and `SFAPParams` are the two configuration surfaces. Downstream
MRI code consumes `muap_generator.api`, so we extend it with new optional
arguments rather than renaming or restructuring.

## 5. Canary categories observed from prior studies

From the smoothness study (`muap_smoothness/results_*.json`) the current
pipeline degrades in these regimes:

- **Deep electrode** (depth ≥ 25 mm on a 240 mm mesh) — field hasn't decayed
  by the edge of the 200 mm sampling extent, sampling zero-pad creates a
  step → Gibbs ringing in MUAP. Mitigation: edge_taper=15 + Butterworth
  c=0.03 o=2.
- **Long fibre** (L1 + L2 > 200 mm) — fibre + IAP envelope doesn't fit
  within the 256 mm window → wraparound.
- **Complex / off-grid geometry** — `field_to_muap_study/findings/P3_signflip.md`
  documented amplitude sign-flips on certain off-axis electrode positions.

These are the three families the test bench needs to cover. Step 1.2 will
build a bench with ~100 sanity inputs and ~100 challenging inputs spread
across these regimes.

## 6. Pipeline ↔ FEM coupling table

| Knob | Set in pipeline | Set in FEM | Coupling |
|------|-----------------|------------|----------|
| `w` (output samples) | `MUAPConfig.w` / `SFAPParams.w` | (independent, but determines `n_z_samples` when sampling FEM at the fibre) | Today: `n_z_samples = w` by convention. Workstream A: decouple — let `w` adapt while fibre sampling stays whatever FEM can deliver, with `resample_centered_line` bridging. |
| `dz` (z spacing) | `v · 1000 / fsamp` (4·1000/4096 ≈ 0.977 mm) | `np.linspace(z_min, z_max, W)` | Coupled by convention; `resample_centered_line` already handles mismatch. |
| Mesh extent | n/a | `length = factor × length_ratio` in `scripts/generate_reference_mesh.py` | Today: 240 mm fixed for cylinder dataset campaign. Workstream B needs paired 240 vs 400 mm meshes. |
| Boundary condition | n/a | `boundary_value = 0` → homogeneous Neumann | Today: hardcoded Neumann. Workstream C may add an `bc_mode` flag. |

## 7. What "the analytical baseline" means here

The reference pipeline (Farina 2004 layered cylinder, analytical) is the
vendored `emgop.analytical` package:

- `emgop.analytical.CylindricalVolumeConductor` — analytical 4-layer
  transfer function on a `(kt, kz)` grid.
- `emgop.analytical.MotorUnit` — fibre geometry container.
- `emgop.analytical.DetectionSystem` — electrode shape.
- `emgop.analytical.SignalGenerator.generate_muap` — Radon section to time.

**Sign convention**: pass `L1=+60.0, L2=+60.0` to `MotorUnit` to match
MATLAB / production conventions. Files that use `L1=-60.0` (e.g.
`03_analytical_comparison.py`) predate the sign fix and are not authoritative
on this detail.

## 8. Validation invariants (per PLAN.md)

1. **Operator-consistency**: analytical φ → production pipeline → MUAP
   should reproduce the analytical MUAP at `r ≈ 1.0`. This is the
   unbreakable invariant: any change that breaks it is a bug.
2. **Physical agreement**: FEM φ → production pipeline → MUAP should match
   analytical MUAP at `r ≥ 0.997` on the cylindrical reference cases.

Step 2 of the bench (the actual snapshot run) will baseline these numbers
on the current pipeline so we can detect regressions per change.
