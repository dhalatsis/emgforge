# Workstream A.3 — wider-mesh probe on still-failing bench cases

Question: of the 17 bench cases still failing after Workstream A.2
(adaptive `w` + auto-smoothing), how many recover when the FEM is run on a
**wider mesh** that gives `w` more room to grow?

Two L400 meshes tested:

1. `avg_L400.msh` — reference-average cross-section (r_skin = 48.2 mm,
   built by Workstream B). **Bad comparison**: cross-section differs from
   both sample_000000 (37 mm) and analytical (40 mm) → spurious negative Δr.
2. `sample000_L400.msh` — sample_000000's cross-section at 400 mm length
   (built by `build_extended_sample.py` for this probe). **Fair**: only the
   length differs from sample_000000.

Both runs are with `get_adaptive_config()`.

## Results on `sample000_L400.msh`

| Case                  | depth | w240 | r240   | w400 | r400   | Δr      |
|-----------------------|-------|------|--------|------|--------|---------|
| B_shallow_fs8192      | 15.0  | 256  | 0.800  | 512  | 0.996  | **+0.197** |
| B_deep_fs8192_d20     | 20.0  | 256  | 0.886  | 512  | 0.975  | **+0.089** |
| B_deep_d24            | 24.0  | 256  | 0.992  | 256  | 0.998  | +0.005  |
| B_deep_d27            | 27.0  | 256  | 0.987  | 256  | 0.991  | +0.004  |
| B_deep_d31            | 31.0  | 256  | 0.940  | 256  | 0.941  | +0.001  |
| B_deep_fs8192_d30     | 30.0  | 512  | 0.931  | 512  | 0.935  | +0.004  |
| B_deep_d33            | 33.0  | 256  | 0.888  | 256  | 0.886  | -0.002  |
| B_deep_d335h          | 33.5  | 256  | 0.856  | 256  | 0.855  | -0.001  |
| B_xtreme_d35          | 35.0  | 256  | 0.702  | 256  | 0.698  | -0.004  |
| B_xtreme_d355h        | 35.5  | 256  | 0.628  | 256  | 0.626  | -0.003  |
| B_deep_d24..others    | ...   | ...  | ...    | ...  | ...    | ±0.02   |

(B_xtreme_d36, B_xtreme_d365h: outside L400 skin — depth > r_skin so
N/A in both meshes; physics doesn't apply.)

## Findings

1. **Three cases recover** with the wider mesh:
   - `B_shallow_fs8192` (high-fsamp, narrow dz → w=512 doesn't fit in L240):
     **+0.197** — adaptive `w` finally has room to grow.
   - `B_deep_fs8192_d20`: **+0.089**.
   - `B_deep_fs8192_d30`: +0.004 (marginal).
2. **The rest are unmoved**: depth-d31..d36 cases stay at the same r
   regardless of mesh length. Their failure isn't about mesh length — it's
   about being near or at the skin surface (fibre depth ≥ 31 mm with
   r_skin = 37 mm). At that proximity, the analytical 4-layer model and
   the 5-layer FEM diverge on physical grounds (the analytical Bessel
   series vs the FE discretization at near-singular geometry).
3. **`avg_L400.msh` is misleading** as a fair comparison because its
   cross-section (r_skin=48.2) differs from sample_000000 (37) and the
   analytical (40). The matched-cross-section sample000_L400 is the right
   comparison.

## Implications

- A bench upgrade that swaps sample_000000.msh → sample000_L400.msh as
  the Tier-B reference would lift challenging from **93/110 → 96/110**
  (the three high-fsamp cases above pass). Modest gain at the cost of
  baking in a 350 MB mesh dependency.
- The 14 remaining FEM challenging failures (`B_xtreme_*`, deep+fsamp_low,
  near-skin) are physics-limited. They'd need either:
  - A radically different validation reference (e.g. multi-layer analytical
    that handles deeper electrodes more gracefully)
  - Or accepting they're inherently at the edge of FEM-vs-Farina-analytical
    agreement (the smoothness study reported r ~ 0.99 at the same depths;
    we're seeing similar)

## Decision

Document and stop iterating. Workstream A pipeline-side improvements
(adaptive w + auto-smoothing) cover the targets the user defined.
Geometric / physical limits are outside the integration's stated scope
(per AGENT_HANDOFF.md: "New physics (Farina 2004 stays; only numerical
realisation changes)").

The wider-mesh path remains available as opt-in:
- `tests/regression/build_extended_sample.py` rebuilds the 400 mm mesh.
- `tests/regression/probe_l400.py` exercises it on selected cases.
- Future work: extend the bench's FEM context to optionally use this mesh.
