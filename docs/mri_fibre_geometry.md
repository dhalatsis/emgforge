# MRI muscle fibre geometry & smoothing

How `emgforge.mri.core.fiber_directions` turns a labelled forearm segmentation into
per-muscle fibre geometry for the FEM σ tensor and for fibre sampling. Records the
smoothing strategies (and their verdicts) that were explored on the `neural-forward-emg`
`mri-morphing-disk` branch, so the knowledge lives with the code.

**Status in emgforge:** we build **centerlines only** (`sigma_mode="centerline"`, the
production default). The cross-section / morphing-disk machinery below is all present in
`fiber_directions.py` but **dormant** — see backlog item **C-04**.

---

## Three geometry layers

1. **Centerline** — `estimate_centerlines(smooth_sigma=…)`. Per-z-slice voxel centroids →
   Gaussian-smoothed (`smooth_sigma` slice-bins) → natural cubic spline. Its tangent orients
   the muscle σ tensor. Default `smooth_sigma=1.0` (archive: "stops cubic-spline overshoot";
   heavy ≥2 over-flattens). **This is what we use.**
2. **Cross-section** — `estimate_cross_sections(…)`. Ray-cast the boundary radius `R(z,θ)`
   from the centroid at `n_theta` angles per slice. Bounds the morphing disk. *Dormant.*
3. **Morphing-disk** — `fiber_tangent_morphing(...)` / `MuscleCrossSection`. Places fibres on a
   normalised disk mapped through `r_inset·R(z,θ)`, so they stay inside the muscle **and**
   follow its cross-sectional area. *Dormant.*

---

## Cross-section boundary smoothing — the five strategies

Ways to clean the blocky voxel boundary `R(z,θ)` (all are `estimate_cross_sections` args).
Compared over 20 muscles on the archive branch (`04_smoothness_options_grid.png`):

| strategy | operation | effect | trade-off |
|---|---|---|---|
| **raw** | ray-cast the raw mask | follows voxel staircase; muscles leave gaps | jagged (baseline) |
| **R-smooth** | Gaussian-smooth `R(θ,z)` after ray-cast (`smooth_sigma`, `z_smooth_sigma`, default 1.5/1.5) | rounds the outline | **shrinks inward** near concavities |
| **mask-sm** | 3-D blur the binary mask + re-threshold, *then* ray-cast (`mask_smooth_xy_mm/_z_mm`) | rounds **and keeps extent** | still leaves inter-muscle gaps |
| **dilated** | binary-dilate the mask first (`mask_dilate_iters`) | muscles grow into the void → **touch** | unconstrained → **bleeds** into neighbours |
| **dilated-cons** | dilate constrained (`constrain_dilation=True`) | Voronoi-style fill of gaps → touch, no bleed | winner: fills ~89% of gaps, ~1.5% bleed |

**Verdict:** `R-smooth` or `mask-sm` for the boundary shape (mask-sm preserves outer extent);
`dilated-cons` on top if you want muscles to touch like real anatomy. "Heavy" boundary
smoothing (θ/z σ ≥ 3) destroys real features (loses L7's two-lobe, L12's kidney-bean).

**σ-orientation sensitivity:** `sigma_sensitivity` found centerline-tangent ≈ full morphing for
MUAP *shape* (r≈1.0 median); morphing is **geometric correctness, not MUAP shape**. That is why
using centerlines alone is a defensible production simplification.

---

## Fusiform fibre architecture (why the morphing-disk matters)

Real muscle fibres are **fusiform**: concentrated at the tendon origin, fanning out through the
muscle belly, concentrating again at the insertion. Verified on the Neurodec MU-113 bundle
(`_results/mri_check/fusiform_bundle_width.png`): bundle spread ≈ **0.7 mm at both tendons →
~1.9 mm in the belly (1.64×)**, then back.

- The **morphing-disk reproduces this for free**: fibres are bounded by `r_inset·R(z,θ)`, and
  `R` shrinks where the muscle narrows (tendons) and grows in the belly — so the bundle fans
  with the muscle (matches the atlas area taper, e.g. L12 725→19 mm²).
- Our current **constant-offset** fibres (parallel copies of the centerline) keep a **constant
  width** — the flat line in that figure — so they miss the fan. This is the main geometric gap
  vs Neurodec (distinct from the ~8 ms timing/EOF gap, which is detection-side).

---

## References
- Machinery: `fiber_directions.py` — `estimate_centerlines`, `estimate_cross_sections`,
  `MuscleCrossSection`, `fiber_tangent_morphing`. Generator: `scripts/mri/build_fiber_config.py`.
- Archive figures (imported): `_results/mri_check/archive_smoothing_plots/` (see its `INDEX.md`) —
  `04_smoothness_options_grid`, `03_fiber_before_after` (85% wiggle reduction),
  `05/06_morphing_validation`, `08/09_muscle_atlas`.
- Backlog: `_refactor/TODO.md` **C-04** (fibre / muscle regularization).
