# Complex-geometry probe — adaptive_w_auto vs default

Ran the production pipeline on three NON-circular meshes that exist locally
(none of the main 200-case bench cases use them — bench is cylinder-only):

| Variant        | Cross-section                      | Note                                  |
|----------------|-------------------------------------|---------------------------------------|
| `two_bone`     | circular, 2 cortical bones offset   | sample_000000 (r_skin=33.0 mm, len=240) |
| `ellipse`      | elliptic ratio 1.2, 1 bone          | sample_000000 (r_skin=44.9 mm, len=240) |
| `offcenter`    | circular, 1 bone displaced          | sample_000000 (r_skin=28.7 mm, len=240) |

For each: one FEM solve (Gaussian σ=5 source at skin top, θ=0, z=center),
then sample φ at depths {10, 15, 20, 25, 30} mm × angles {0, 45, 90}° from
the electrode. Run φ through both **default `MUAPConfig()`** and
**`get_adaptive_config()`** (adaptive `w` + auto-smoothing). Compare both
against the analytical 4-layer cylinder reference at the matched `w`.

## Headline result

```
variant       n   default r       ok    auto r          ok    Δr mean
two_bone     15   0.891 ±0.165   15/15  0.891 ±0.165   15/15  +0.0000
ellipse      15   0.842 ±0.172   15/15  0.842 ±0.172   15/15  +0.0000
offcenter    12   0.937 ±0.066   12/12  0.937 ±0.066   12/12  +0.0000
```

**Default and adaptive_w_auto produce identical outputs on every case (42/42).**

Why: when the input φ isn't decayed at its mesh-bound boundary, the
input-length safety cap holds `w` at 256 — same as default. Auto-smoothing
detects the FEM's HF content (~1e-5) and applies the same Butterworth as
default. Two opt-in features collapse to the legacy behaviour when the
geometry can't support anything different.

**Property worth keeping**: adaptive_w_auto is **regression-safe** even on
geometries it can't help with. No surprise behaviour change for complex /
short-mesh datasets.

## Detailed numbers (r vs cylinder analytical reference at matched w)

| Mesh        | depth | θ=0°  | θ=45° | θ=90° | ok | comment |
|-------------|-------|-------|-------|-------|----|---------|
| two_bone    | 10 mm | 0.999 | 1.000 | 0.991 | ✓ | shallow, bone effect mild |
| two_bone    | 15 mm | 0.994 | 0.995 | 0.968 | ✓ | |
| two_bone    | 20 mm | 0.977 | 0.968 | 0.900 | ✓ | |
| two_bone    | 25 mm | 0.969 | 0.861 | 0.738 | ✓ | bone path at 90° |
| two_bone    | 30 mm | 0.972 | 0.589 | 0.438 | ✓ | deep + 90° passes through both bones |
| ellipse     | 10 mm | 0.988 | 0.984 | 0.976 | ✓ | shallow, ellipse impact small |
| ellipse     | 15 mm | 0.979 | 0.969 | 0.953 | ✓ | |
| ellipse     | 20 mm | 0.949 | 0.918 | 0.884 | ✓ | |
| ellipse     | 25 mm | 0.874 | 0.787 | 0.720 | ✓ | |
| ellipse     | 30 mm | 0.703 | 0.518 | 0.422 | ✓ | deep + 90° along long axis |
| offcenter   | 10 mm | 0.977 | 0.996 | 0.995 | ✓ | bone offset minor |
| offcenter   | 15 mm | 0.940 | 1.000 | 0.976 | ✓ | |
| offcenter   | 20 mm | 0.893 | 0.989 | 0.914 | ✓ | |
| offcenter   | 25 mm | 0.900 | 0.905 | 0.759 | ✓ | |

## How to read the low r values

The reference is the analytical 4-layer **circular** cylinder. Genuine
complex-geometry MUAPs **should** diverge from it when the geometry
asymmetries become significant — that's a feature, not a bug. The "ok"
column verifies each MUAP is well-formed (PTP > 0, 2 ms < duration < 25 ms,
HF ratio < 0.5) regardless of how it compares to the cylindrical baseline.

The right benchmark for these geometries is the FEM solution **vs itself**
(self-consistency across configs) — and on that metric, default and
adaptive_w_auto are bit-for-bit identical, which is exactly the safe
behaviour we want.

## What this DOESN'T show

- Mesh-extent-bound improvements (adaptive_w can't grow `w` because mesh
  length is 240 mm in all three). Would need a wider mesh to see daylight
  between the two configs.
- Sister MRI canaries — those would actually exercise the adaptive grow path
  because MRI volumes have more axial extent.

## File map

- `tests/regression/probe_complex.py` — runs the probe
- `tests/regression/PROBE_COMPLEX.json` — full numeric results
- This document — interpretation
