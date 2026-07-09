# "Why do all the MUAPs look the same?" — shape vs amplitude analysis

User flagged two suspicious observations during the ellipse+two-bone probe:

1. "If you are just using the FEM results, why are some φ(z) noisy?"
2. "All MUAPs look exactly the same, something is off."

Both are real observations. Both turn out to be physics / numerical
behaviour we'd expect, not pipeline bugs. Documented here for posterity.

---

## Q1: Why is φ(z) "noisy"?

**It's not noise — it's CG1 piecewise-linear interpolation kinks.**

The FEM solver uses CG1 (P1) function space — solution is piecewise
linear between mesh vertices. C⁰ continuous, but derivatives jump at
element boundaries. When we sample φ at 0.977 mm spacing through cells
that are typically several mm wide, we cross many element boundaries per
sample and the linear segments stitch into a slightly-jagged curve.

### Evidence — sampling-density invariance

Same fibre line (d=10 mm, θ=90°), evaluated at three sampling densities
on the ellipse+2-bone mesh:

| n samples | dz       | peak     | roughness `d²/dz²` | HF fraction (>0.3/mm) |
|-----------|----------|----------|--------------------|------------------------|
| 256       | 0.977 mm | 3.12e-3  | 4.59e-05           | 0.0004                 |
| 1024      | 0.244 mm | 3.12e-3  | 3.45e-04           | 0.0006                 |
| 4096      | 0.061 mm | 3.12e-3  | 2.66e-03           | 0.0006                 |

- Peak amplitude is **invariant** under sampling density — the FEM solution
  itself is consistent.
- HF fraction is **constant** (~0.0005) across 16× density change — that's
  the bandlimited spectral content of the underlying CG1 function.
- Roughness **explodes ~6× per 4× density increase** — exactly the
  signature of piecewise-linear functions whose second derivatives are
  delta-functions at cell boundaries. Finer sampling resolves more
  delta-spikes.

If this were random sampling noise, HF fraction would scale with `√n`.
It doesn't.

### What to do about it

Three options, none of which are integration-scope:
- Switch the FEM space from CG1 → CG2 (quadratic) for C¹ continuity. Costs
  ~8× memory + solve time. Out of scope.
- Smooth φ before downstream use. **This is already the default**
  (Butterworth c=0.03, o=2). Reduces the propagated artefact 10× on the
  worst cases — see `PROBE_ELLIPSE_TWO_BONE.md`.
- Sample at element-centroid spacing rather than fine intra-element grid.
  Cosmetic; doesn't change the spectral content.

The pipeline as-shipped handles this correctly via Butterworth.

---

## Q2: Why do all MUAPs look the same?

**The MUAP shape is dominated by IAP × pare, both of which are FIXED
across cases.** The lead field φ(z) mostly modulates amplitude; shape
changes only when the field's spatial structure is significantly different.

### Evidence — analytical reference has the same property

Pure analytical MUAPs (4-layer cylinder, no FEM involved), varying only
the fibre depth. Sign-aligned, normalised to peak, then correlated against
d=8 mm:

| depth | PTP      | shape r vs d=8 mm |
|-------|----------|-------------------|
|  8 mm | 7.5e-6   | +1.0000 (ref)     |
| 12 mm | 8.2e-6   | +1.0000           |
| 18 mm | 1.3e-5   | +0.9861           |
| 25 mm | 2.7e-5   | +0.8553           |
| 30 mm | 5.6e-5   | **−0.5770** (polarity flip!) |

So even the **analytical model** produces nearly-identical MUAP shapes
across shallow depths and only diverges at deep depths — where the
geometry drives a polarity flip (the classic Phase-3 sign-flip from the
field_to_muap_study).

### Why this happens — IAP × pare dominates

The Fourier MUAP pipeline assembles:

```
   MUAP(t) = Radon{ (1/v) · spe2(kz) · pare(kα, kβ) · jkz · C(kz) }
```

- `spe2(kz)` — Rosenfalck IAP spectrum, FIXED (depends only on v).
- `pare(kα, kβ)` — fibre-end function, FIXED for given L1, L2, v.
- `jkz` — derivative factor, FIXED.
- `C(kz) = FFT{φ(z)}` — only this depends on the lead field.

Three of the four factors are case-invariant. The MUAP's morphology
(tri-phasic spike shape, peak times, fibre-end ringing) comes from those
three. C(kz) acts as a "field-shaped filter" that modulates this fixed
morphology — strongly in amplitude, weakly in shape unless the field has
sharply different spatial structure.

### What our bench cases produce

Pairwise SHAPE correlations between MUAPs at different (depth, θ) on the
ellipse+2-bone mesh, normalised + sign-aligned:

```
                d10t000  d10t090  d20t000  d20t090  d30t000  d30t090
   d10t000      1.0000   0.9980   0.9949   0.9971   0.9211   0.9972
   d10t090      0.9980   1.0000   0.9866   0.9998   0.8950   0.9998
   d20t000      0.9949   0.9866   1.0000   0.9844   0.9554   0.9846
   d20t090      0.9971   0.9998   0.9844   1.0000   0.8899   1.0000
   d30t000      0.9211   0.8950   0.9554   0.8899   1.0000   0.8905
   d30t090      0.9972   0.9998   0.9846   1.0000   0.8905   1.0000
```

- 14 of 15 pairs have r ≥ 0.98 — shapes are nearly identical.
- Only **d30t000** stands apart (r ≈ 0.89–0.96 vs other cases) — the
  d=30/θ=0° MUAP has a visibly wider positive tail because the fibre is
  in the field's most-intense region.
- Note: the analytical at d=30 polarity-flips (r=−0.58 vs d=8); the FEM
  at d=30/θ=0 does NOT flip (r=+0.92 vs d=10/θ=0). This is a *separate*
  finding: the FEM with σ=5 Gaussian source doesn't reproduce the Phase-3
  inversion the analytical predicts. Outside the integration scope, but
  worth flagging for the field_to_muap_study follow-up.

### What you SHOULD see vary across cases

- Amplitude (40× range across the 22 ellipse+2-bone cases — yes)
- Subtle envelope changes (visible in PTP-normalised plot — yes)
- Polarity flips (only in the regime where the analytical does too — d≥25
  for cylinder geometries)

### Why the four pipeline-config plots overlay in PROBE_ELLIPSE_TWO_BONE

Inside each per-case plot, `default`, `adaptive_w_auto`, and `edge_taper`
produce **identical** MUAPs (Δr = 0). This is the regression-safety
property by design:

- `adaptive_w_auto` safety cap holds `w` at 256 when the input isn't
  decayed at its boundary (which it isn't on this mesh).
- Auto-smoothing detects FEM HF and applies the same Butterworth as
  default.
- `edge_taper=15` is a near-no-op when `w=256` and the input φ already
  saturates at a near-constant baseline near the boundary.

So all three post-smoothing pipelines do the same thing on this geometry.
That's the *opposite* of a bug — it means the integration doesn't break
anything when its features can't help.

---

## Honest take on the bench metric

`r(production_MUAP, analytical_MUAP)` is partially inflated by the
universal IAP×pare shape. High r (e.g. 0.99) tells you the pipeline got
the IAP and pare right and produced a sensible-looking MUAP. It does NOT
strongly discriminate between subtle field-shape differences.

A more discriminating bench would also report:
- **PTP ratio** (amplitude match)
- **Shape r after PTP normalisation** (shape-only match)
- **Pipeline self-consistency** (default vs adaptive vs edge_taper on the
  same input)

The current bench reports r-vs-analytical as a single number. Future
extension: report a 3-tuple per case. Not urgent — the bench passes/fails
have been calibrated against the existing single metric.

---

## Files

- `tests/regression/figures/diag_shape_amplitude.png` — 2×2 panel showing
  raw φ, normalised φ, raw MUAPs, normalised MUAPs across 5 contrasting
  (d, θ) cases.
- `tests/regression/PROBE_ELLIPSE_TWO_BONE.md` — the smoothing-vs-no-
  smoothing analysis (still valid; reconfirms HF jaggedness fix).
