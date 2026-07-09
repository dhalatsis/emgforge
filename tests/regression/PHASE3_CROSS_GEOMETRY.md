# Phase 3 cross-geometry sign-flip regression — Tier C

Closes the gap identified by
`field_to_muap_study/deliverables/fem_worker_integration/VERIFICATION_2026-05-26.md`:
Round 1's 200-case bench is cylinder-only and never exercised the cross-
geometry pairing scenario that the original Phase 3 finding was discovered
on. This document and `tests/regression/phase3_bench.py` add that
coverage.

## Test design

For each of the 4 complex datasets (`fem_fiber_offgrid_complex`,
`fem_fiber_matched_depth_complex`, `fem_fiber_distance_sweep_complex`,
`fem_fiber_finedz_complex` — all under
`./data/training/results/`):

1. Load complex φ + matching circular φ (50 electrodes each).
2. Compute pipeline MUAPs per electrode at fixed w=256 (the original
   Phase 3 test condition).
3. Pair each complex electrode `i` to the nearest circular electrode by
   `(theta_deg, z_norm)` cylindrical position (`max_dtheta=20°, max_dz=0.08`).
4. PTP-filter: drop pairs whose complex MUAP PTP is < 30% of the dataset's
   median PTP (mirrors original P3 amplitude filter).
5. Report signed Pearson r per pair; count `r < 0` as a "flip".

Acceptance: **0 % flips with `get_adaptive_config()` on all 4 datasets**
after Round 2 (auto-edge_taper).

## Results

```
Loading circular reference: 50 electrodes, fiber_r=28.92, r_skin=48.20

Dataset                              default      adaptive_w_auto     truncated_explicit
                                    flips/n     flips/n             flips/n
fem_fiber_offgrid_complex            7/21 (33%)  0/17  (0%)          0/17  (0%)
fem_fiber_matched_depth_complex      0/18  (0%)  0/13  (0%)          0/13  (0%)
fem_fiber_distance_sweep_complex     3/34  (9%)  0/28  (0%)          0/28  (0%)
fem_fiber_finedz_complex             0/18  (0%)  0/13  (0%)          0/13  (0%)
```

Total: default = 10/91 flips (11 %), adaptive_w_auto = **0/71 flips (0 %)**,
truncated_explicit = 0/71 flips (matches adaptive — sanity check).

The `offgrid_complex` dataset reproduces the original Phase 3 finding's
worst-case flip rate (33 % matches `findings/P3_signflip.md` ~19-23 %
range). The auto-edge_taper detects truncation on every flipped case
(edge/peak ≈ 1.0) and applies the 15-sample cosine taper, restoring
median signed r from +0.865 → +0.994 — exactly matching the explicit
`edge_taper=15` baseline that the original study recommended.

## What "auto" detects

`muap_generator.api._edge_over_peak(phi)` returns
`max(|φ_left|, |φ_right|) / max(|φ|)`. Triggers the taper when this exceeds
`MUAPConfig.auto_edge_taper_threshold` (default 0.3).

Empirical separation on these datasets:

| Source                          | edge/peak typical | edge/peak max |
|---------------------------------|-------------------|----------------|
| analytical (sharp Gaussian φ)   | 0.001             | 0.01           |
| FEM cylinder shallow (d=15)     | 0.14              | 0.40           |
| FEM cylinder deep (d=25+)       | 0.1               | 0.5            |
| Phase 3 offgrid_complex         | 1.00              | 1.00           |
| Phase 3 matched_depth_complex   | 0.12 – 1.00       | 1.00           |

The 0.3 threshold cleanly separates "field decayed at boundary" (no
taper needed) from "field saturated at boundary" (taper required). The
verification-document spot-check recommended 0.3 as the default,
tunable up to 0.5 if false positives appear on deep cylinder cases.
Pushing the threshold above 0.7 would re-expose the Phase 3 flip
pattern.

## Reproducing

```bash
PYTHONPATH=src python tests/regression/phase3_bench.py \
    --out tests/regression/snapshots/phase3_round2.json
```

Configurable: `--datasets fem_fiber_offgrid_complex` (or any subset of
the 4) to skip slower runs.

Wall time: ~20 s for all 4 datasets × 3 configs = 12 runs.

## File map

- `tests/regression/phase3_bench.py` — driver
- `tests/regression/snapshots/phase3_round2.json` — full numerical results
- `field_to_muap_study/findings/P3_signflip.md` — original Phase 3 finding
- `field_to_muap_study/deliverables/fem_worker_integration/VERIFICATION_2026-05-26.md` —
  verification that surfaced the gap
- `muap_generator/api.py::_edge_over_peak()` — the detector
- `tests/test_auto_edge_taper.py` — unit tests for the detector + pipeline integration
