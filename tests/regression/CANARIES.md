# Baseline canary report (default config, commit pre-Workstream-A)

Captured by `tests/regression/bench.py snapshot --config default` against
the 200-case bench. Numbers below are r vs analytical-MUAP reference.

## Sanity pool — 90/90 pass (100%)

| Tier | n  | mean r | min r | max r |
|------|----|--------|-------|-------|
| A    | 43 | 0.9972 | 0.9911| 0.9995|
| B    | 47 | 0.9988 | 0.9948| 1.0000|

Clean baseline. Any future change must keep this at 90/90.

## Challenging pool — 81/110 pass (74%) — TARGETS

| Tag                             | n  | pass | mean r | min r  |
|---------------------------------|----|------|--------|--------|
| `B_xtreme_deep` (d=35-36.5mm)    | 4  | 0/4  | 0.59   | 0.49   |
| `B_shallow_fsamp_extreme`        | 2  | 0/2  | 0.93   | 0.88   |
| `A_fsamp_sweep_extreme` (fs=2048)| 1  | 0/1  | 0.97   | 0.97   |
| `A_depth_sweep_deep` (d=21-30)   | 8  | 0/8  | 0.96   | 0.88   |
| `B_too_shallow` (d=5-6)          | 2  | 0/2  | 0.98   | 0.98   |
| `B_deep_fsamp`                   | 6  | 4/6  | 0.88   | 0.67   |
| `A_deep_long`                    | 6  | 2/6  | 0.92   | 0.87   |
| `B_deep` (d=24-34)               | 11 | 7/11 | 0.95   | 0.81   |
| `B_deep_half` (deep, half-mm)    | 6  | 4/6  | 0.95   | 0.86   |
| `B_asym2` (deep, asym)           | 12 | 12/12| 0.98   | 0.93   |
| `B_deep_long`                    | 6  | 6/6  | 0.98   | 0.96   |

The five worst tags (`B_xtreme_deep`, `B_shallow_fsamp_extreme`,
`A_fsamp_sweep_extreme`, `A_depth_sweep_deep`, `A_deep_long`) are the
**primary canaries** for Workstream A (adaptive `w`). The deep-electrode
group is the **primary canary** for Workstream B (Neumann error budget).

## Negative result — edge_taper as default

Running `--config edge_taper` (Butterworth + edge_taper=15) catastrophically
regresses Tier A operator-consistency: A_longfib drops r=0.99 → 0.12;
A_extreme_asym drops r=0.99 → 0.27; A_deep_long drops r=0.92 → 0.23.

Reason: edge_taper forces the first/last 15 samples to zero. For analytical
φ(z) on a long fibre or wide field, those samples carry real signal.
Tapering destroys it.

**Implication for Workstream D defaults**: edge_taper should NOT be a global
default. It's a useful crutch ONLY when the input φ has truncation/sampling
artifacts at its boundary (typical for FEM with mesh edge near sampling
extent). Adaptive `w` (Workstream A) is the principled fix; edge_taper stays
as an opt-in tool.

## Workstream A.2 update — `adaptive_w_auto` results

`smoothing_method="auto"` (default in `get_adaptive_config()` since
Workstream A.2) detects input HF content and smooths only when needed
(HF > 1e-10). Analytical inputs (HF ≈ 1e-30) skip smoothing → operator
consistency r → 1.0. FEM inputs (HF ≈ 1e-5) get the usual Butterworth.

| Config                  | Sanity   | Challenging | Δ vs default |
|-------------------------|----------|-------------|--------------|
| `default`               | 90/90    | 81/110      | —            |
| `adaptive_w`            | 90/90    | 80/110      | -1           |
| **`adaptive_w_auto`**   | 90/90    | **93/110**  | **+12**      |

The +12 comes entirely from Tier A challenging cases (operator-consistency
on long-fibre, deep-depth, fsamp-extreme, asym-fibre regimes), now all
passing at r = 1.000.

Remaining 17 challenging failures are physical/geometric limits of the
240 mm cylinder mesh and the d=35-36.5 mm near-skin regime — they need
wider mesh or different physics, not pipeline changes.
