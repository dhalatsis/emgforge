# Workstream A — adaptive `w` results

Status: heuristic + safety cap implemented; tested; doesn't regress sanity;
modest gains on Tier A challenging when smoothing is off. Bench-limited
by the 240 mm sample_000000 mesh.

## Bench snapshots compared

| Config                       | Sanity (Tier A / Tier B) | Challenging (Tier A / Tier B) |
|------------------------------|--------------------------|-------------------------------|
| `default`                    | 43/43, 47/47 = 90/90 ✓   |  9/22, 72/88 = 81/110 (74%)   |
| `edge_taper`                 | 40/43, 47/47 = 87/90 ✗   |  0/22, 56/88 = 56/110 (51%)   |
| `adaptive_w`                 | 43/43, 47/47 = 90/90 ✓   |  9/22, 71/88 = 80/110 (73%)   |
| `adaptive_w_taper`           | 43/43, 47/47 = 90/90 ✓   |  9/22, 58/88 = 67/110 (61%)   |
| `adaptive_w_no_smoothing`    | 43/43,  1/47 = 44/90 ✗   | **22/22**, 21/88 = 43/110     |

## Findings

1. **Operator-consistency is recovered to r=1.0** by `adaptive_w_no_smoothing`
   on every challenging Tier A case (the analytical→pipeline→analytical loop
   becomes a perfect identity at all depths up to d=30 mm). The two-sided
   nonlinear-LS field-decay fit handles the analytical's DC-baseline cleanly.

2. **Smoothing trades operator consistency for FEM resilience.** Default
   Butterworth (c=0.03, o=2) keeps Tier B sanity at 100% (FEM φ has
   high-frequency numerical noise that smoothing removes) but costs ~0.01
   on Tier A operator consistency. Net trade is positive — keep as default.

3. **Adaptive `w` alone is roughly neutral on the bench.** Tier A
   challenging passes don't change (smoothing is the bottleneck, not `w`).
   Tier B challenging loses one case (B_longfib_d20_L80) because the FEM
   only provides 240 mm of φ and the heuristic correctly refuses to grow
   `w` past what the data supports (input-length cap).

4. **`edge_taper` should not be a global default.** Tapers `φ(boundary) → 0`
   even when the data carries real signal there. Tier A `A_longfib` r drops
   0.99 → 0.12 under `--config edge_taper`. Useful only when input φ is
   known to be truncated mid-decay.

5. **The mesh is the bottleneck** on the bench's challenging Tier B pool.
   `B_xtreme_deep` (d=35..36.5 mm) has the fibre at the cylinder skin
   surface — physics gets singular. `B_deep` (d=24..34) needs wider mesh
   or better BC to recover. Both are Workstream B/C territory.

## Sister-branch coordination

The MRI sister worktree (memory `mri-morphing-disk-canaries`) logged 12
canary cases where φ peaks at the data-extent edge — exactly the
wide-field/long-fibre regime adaptive `w` is for. Those will benefit when
the MRI agent's MUAP path picks up the new `MUAPConfig(w=None)` default.

## Recommended pipeline defaults (carried forward to Workstream D)

```python
MUAPConfig(
    w=None,                           # adaptive
    w_min=256, w_max=1024,            # safety clamps
    smoothing_method="butterworth",   # kept for FEM resilience
    butterworth_cutoff=0.03, butterworth_order=2,
    edge_taper=0,                     # NOT a default
)
```
