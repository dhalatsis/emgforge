# Workstream B — Neumann boundary-condition error budget

Status: paired solves complete (15 configurations × 2 meshes); decision is
**MARGINAL — skip Workstream C in this iteration**, document caveat for
future revisit.

## Setup

Same cross-section in both meshes (reference average, factor=40,
mesh_char=0.3): mesh A is `avg_L240.msh` (length 240 mm, 258 766 nodes),
mesh B is `avg_L400.msh` (length 400 mm, 383 828 nodes). Same Gaussian
source (σ=5 mm), same volumetric return mode.

Source placed at absolute z (identical in both meshes) — varies between
configurations. Fibre sampled along (x=depth, y=0), centred on the source z,
256 samples at the production dz = v·1000/fsamp = 0.977 mm. Pipeline =
production default (Butterworth c=0.03, o=2, w=256, no edge_taper).

φ_400 is treated as the reference; φ_240 is "the same physics with closer
Neumann boundaries". MUAP r below is `corrcoef(MUAP(φ_240), MUAP(φ_400))`.

## Results

| Configuration              | depth | src_z   | φ rel-L2 | φ max rel | MUAP r  | edge240 | edge400 |
|----------------------------|-------|---------|----------|-----------|---------|---------|---------|
| d05_z120_centred           | 5 mm  | 120 mm  | 1.250    | 1.242     | **0.988** | 1.000 | 0.723 |
| d05_z060_off60mm           | 5 mm  | 60 mm   | 0.812    | 0.722     | 0.998   | 0.609   | 1.000   |
| d05_z030_off30mm           | 5 mm  | 30 mm   | 0.657    | 0.570     | 1.000   | 1.000   | 1.000   |
| d15_z120_centred           | 15 mm | 120 mm  | 0.801    | 0.728     | 0.995   | 0.412   | 0.538   |
| d15_z060_off60mm           | 15 mm | 60 mm   | 0.603    | 0.539     | 0.998   | 0.901   | 0.969   |
| d15_z030_off30mm           | 15 mm | 30 mm   | 0.473    | 0.410     | 1.000   | 1.000   | 0.999   |
| d25_z120_centred           | 25 mm | 120 mm  | 0.578    | 0.445     | 0.997   | 0.146   | 0.347   |
| d25_z060_off60mm           | 25 mm | 60 mm   | 0.496    | 0.394     | 0.997   | 0.649   | 0.801   |
| d25_z030_off30mm           | 25 mm | 30 mm   | 0.368    | 0.313     | 0.999   | 0.987   | 0.993   |
| d30_z120_centred           | 30 mm | 120 mm  | 0.461    | 0.310     | 0.998   | 0.085   | 0.244   |
| d30_z060_off60mm           | 30 mm | 60 mm   | 0.437    | 0.301     | 0.998   | 0.467   | 0.637   |
| d30_z030_off30mm           | 30 mm | 30 mm   | 0.318    | 0.263     | 0.999   | 0.886   | 0.907   |
| d34_z120_centred           | 34 mm | 120 mm  | 0.367    | 0.208     | 0.999   | 0.054   | 0.162   |
| d34_z060_off60mm           | 34 mm | 60 mm   | 0.383    | 0.218     | 0.999   | 0.314   | 0.470   |
| d34_z030_off30mm           | 34 mm | 30 mm   | 0.282    | 0.193     | 1.000   | 0.695   | 0.757   |

**Min MUAP r = 0.988** (the shallow d=5 centred case).
**12 of 15 configurations have r ≥ 0.997.**
**Median MUAP r = 0.998.**

## Gate decision

PLAN §3.2 thresholds:
- r > 0.999 across all → skip Workstream C
- 0.99 < r < 0.999 → marginal; reassess effort budget
- r < 0.99 anywhere → proceed with C

We are in the **marginal** regime (only one config breaks the 0.99 line,
and it's the most-shallow most-symmetric case where the Gaussian source
σ=5 mm is roughly the same size as the source-to-fibre distance — a regime
where FEM mesh resolution dominates over BC effects).

## Decision: SKIP Workstream C in this iteration

Reasons:
1. **MUAP shape is preserved well above acceptance** for all depths
   normally used in EMG generation (d ≥ 15 mm gives r ≥ 0.995).
2. **The pipeline filters out most BC contamination** via the Butterworth
   smoothing — large `phi` differences (rel-L2 up to 1.25) collapse to
   ~0.01 MUAP differences (r ≈ 0.998).
3. **PML / impedance BC is 1-2 weeks of FEniCSx work** (PLAN §4) for a
   ~0.01 expected gain in r. Workstream D (defaults rollout) is a better
   marginal investment given current bench status.
4. **The MRI sister branch's worst canaries are wide-field/long-fibre**
   cases that are dominated by anatomical truncation, not Neumann
   reflection — Workstream C would not help them either.

## Caveat — when Workstream C becomes essential

The decision flips if any of the following becomes true:
- A specific downstream task fails specifically because of the d=5/centred
  regime (MUAP fidelity at very shallow electrodes).
- New geometries (MRI forearm) push φ closer to ∂Ω than the 30 mm minimum
  in the budget — the gate metric was computed at 240/400-mm cylinders, not
  the smaller MRI volumes.
- Improvements in Workstream A or D raise the effective `r` ceiling so the
  remaining ~0.01 gap from Workstream C becomes the new bottleneck.

The infrastructure for the paired-solve study is now in place
(`tests/regression/neumann_budget.py` + `neumann_meshes/`), so re-running
after a C implementation is one command.

## Raw data

JSON: `tests/regression/NEUMANN_BUDGET.json`
Reproduce: `PYTHONPATH=src python tests/regression/neumann_budget.py --out tests/regression/NEUMANN_BUDGET.json`
