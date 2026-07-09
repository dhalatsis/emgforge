# FEM worker integration — final summary (autonomous run)

Branch: `fem-worker-integration`. Mission per `AGENT_HANDOFF.md`: take the
field_to_muap_study findings and bake them into the production FEM → MUAP
pipeline so it handles arbitrary geometries without per-dataset manual tuning.

## Outcome

| Metric                       | Baseline  | After all workstreams | Δ        |
|------------------------------|-----------|----------------------|----------|
| **Sanity pool**              | 90/90 ✓   | 90/90 ✓              | 0        |
| **Challenging pool**         | 81/110    | **93/110**            | **+12**  |
| **Operator-consistency (Tier A challenging at r≥0.99)** | 9/22 | **22/22** | **+13**  |
| Default-config Δr per case   | n/a       | 0.0000 ± 0.0000      | none     |

Backwards compatibility: every existing `MUAPConfig()` / `SFAPParams()` call
behaves exactly as before. The improvements are opt-in via
`get_adaptive_config()` or `MUAPConfig(w=None, smoothing_method="auto")`.

## What landed

### Step 1 — Audit + regression bench (commit `671e3c1`)

- `tests/regression/AUDIT.md`: integration-points memo
- `tests/regression/cases.py`: 200 deterministic test cases (90 sanity,
  110 challenging) — Tier A (analytical φ for operator consistency) and
  Tier B (FEM φ for physical agreement) tiers
- `tests/regression/bench.py`: snapshot/compare runner
- `tests/regression/analytical_ref.py`: extracts analytical φ from the
  Farina 2004 SignalGenerator innards for Tier A
- `tests/regression/snapshots/baseline_default.json` (+ edge_taper baseline)

### Workstream A — adaptive `w` (commit `b0c572c`)

- `muap_generator/adaptive_w.py` — heuristic + input-length safety cap
- Nonlinear-LS `A·exp(-|z|/λ) + B` decay fit handles real Neumann-nullspace
  baselines
- `MUAPConfig.w` accepts `None` → adaptive selection
- 18 unit tests in `tests/test_adaptive_w.py`

### Workstream B — Neumann BC error budget (commit `c93c845`)

- Two reference meshes built: `avg_L240.msh` (240 mm) and `avg_L400.msh`
  (400 mm), same average cross-section
- `tests/regression/neumann_budget.py` runs paired solves at 15
  configurations (5 depths × 3 source-z positions)
- min MUAP r = 0.988, median = 0.998 → **marginal** → **SKIP Workstream C**
  this iteration (PML/impedance BC deferred)
- Re-activation criteria in `tests/regression/NEUMANN_BUDGET.md`

### Workstream A.2 — HF-aware "auto" smoothing (commit `2d73edf`)

- `MUAPConfig.smoothing_method = "auto"` — HF energy fraction > 1e-10
  → Butterworth, else pass-through
- Clean 5-orders-of-magnitude separation: analytical HF ≈ 1e-30, FEM HF
  ≈ 1e-5
- `get_adaptive_config()` updated to default to `"auto"` smoothing
- 8 unit tests in `tests/test_auto_smoothing.py`
- **Recovered all 22 Tier A challenging cases at r ≈ 1.0** (operator
  consistency restored)

### Workstream D — defaults rollout (commit `5aea1cc`)

- `get_adaptive_config()` / `get_truncated_input_config()` presets added
- `muap_generator/PIPELINE.md` preset table updated
- Final regression: **Δr = 0** for `MUAPConfig()` default

### Workstream A.3 probe — wider-mesh ceiling (commit `f393a2c`)

- Built `sample000_L400.msh` (sample_000000 cross-section, 400 mm length)
- Probed the 17 still-failing cases: **3 recover** (all high-fsamp cases
  where `dz` is small enough that even `w=512` doesn't fit in L240)
- The other 14 are physics-limited (fibre depth ≥ 31 mm with r_skin = 37 mm
  — at-or-near the skin surface, where 5-layer FEM and 4-layer analytical
  diverge inherently)
- Infrastructure in `build_extended_sample.py` + `probe_l400.py` for
  future activation; bench default mesh unchanged

## What's still open (deliberately deferred)

1. **Workstream C (PML / impedance BC)** — gate metric was marginal
   (min r = 0.988); deferred until either MRI canaries demonstrate need
   or downstream tasks specifically fail in the shallow-d=5 regime.
2. **L400 default for Tier B** — would lift challenging from 93 → 96
   but bakes in a 350 MB mesh dependency. Deferred.
3. **Physics-limited cases** (d ≥ 31 mm at r_skin ≈ 37 mm) — outside
   scope per `AGENT_HANDOFF.md` ("New physics stays").
4. **MRI canary integration** — sister branch (memory:
   `mri-morphing-disk-canaries`) has 12 wide-field cases. They will benefit
   when the MRI agent's MUAP path opts into `get_adaptive_config()`.

## How to use the new pipeline

```python
from muap_generator import get_adaptive_config, generate_muap_from_phi

cfg = get_adaptive_config()           # adaptive w + auto smoothing
result = generate_muap_from_phi(phi_mat, dz_mm=dz, config=cfg)
```

Backwards-compatible — `MUAPConfig()` still defaults to w=256, Butterworth.

## Regression bench commands

```bash
# Baseline + post-change snapshots
PYTHONPATH=src python tests/regression/bench.py snapshot \
    --config default --out tests/regression/snapshots/post_change.json
PYTHONPATH=src python tests/regression/bench.py compare \
    --reference tests/regression/snapshots/baseline_default.json \
    --new tests/regression/snapshots/post_change.json

# Other configs available: edge_taper, no_smoothing, adaptive_w,
# adaptive_w_taper, adaptive_w_no_smoothing, adaptive_w_auto

# Workstream B re-run
PYTHONPATH=src python tests/regression/neumann_budget.py \
    --out tests/regression/NEUMANN_BUDGET.json
```

## Unit tests

```bash
PYTHONPATH=src python -m pytest tests/test_adaptive_w.py tests/test_auto_smoothing.py -v
# 26 tests, all passing
```
