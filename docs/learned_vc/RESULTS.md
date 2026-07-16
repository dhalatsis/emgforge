# Learned VC — results so far (cylinder, dev scale)

**Run:** 2026-07-16 · branch `feat/learned-vc` · worktree `/home/dc23/projects/emgforge-nf`
**Scope:** Phases 0–2 complete on the cylinder. MRI (Phases 3–7) not started, by design —
Gate 2 had to pass first.

## Headline

| gate | question | result |
|---|---|---|
| **0b** | is a surrogate worth it on speed? | ✅ **fired** — no. See below. |
| **1** | is the MUAP scoreboard reproducible? | ✅ **PASS** — byte-identical rerun |
| **2** | can a learned VC reproduce cylinder MUAPs? | ✅ **PASS** — median MUAP corr **+0.991** |

## Gate 0b — the speed case for a surrogate collapses

`00_baseline_fem_timing.py`, WR, 127k fibre query points:

| | measured |
|---|---|
| `solve_for_point` / electrode | 0.39 s |
| `compute_closest_entity` (point→cell) | **7.17 s — 99.5% of the cost** |
| `uh.eval` | 0.036 s |

`cell_ids` depends only on `(mesh, points)`, never on the solution ⇒ cacheable for a fixed query
set. **Verified byte-identical φ (max|Δ| = 0.00e+00) at 165× faster sampling → 7.6 s → 0.43 s per
electrode (~18×).**

**Consequence:** the prior **75×** surrogate speedup was measured against an FEM redundantly
re-locating the same points. Against a cached FEM, a 0.15 s net is **~3×**. The learned VC must
be justified by **differentiability / mesh-free deployment / generalisation**, *not* speed.
**This is an open decision for a human.** (Silver lining: dataset gen 259 min → 15 min.)

*Also a free ~18× for the existing pipeline — see `HANDOFF.md`.*

## Gate 1 — the frozen MUAP benchmark

`muap_bench_cyl.npz`: **27 configs** (9 electrodes × 3 MU depths), 125 s. Rerun reproduces every
array byte-for-byte. Difficulty gradient **8.7 µV → 0.14 µV**. Verified the weak configs are real
MUAPs (3–7 zero-crossings), not FEM noise; their long durations are the correct deep-source
signature (broad φ smears the MUAP).

## Gate 2 — the learned VC on the cylinder

64 electrodes × 20k pts (646 s) → MLP+Fourier, 331 K params, 215 s on GPU → scored on the 27
**held-out** configs:

| | φ rel-L2 | MUAP corr (median) | ≥0.94 | p2p med | Δjag med | Gate 2 |
|---|---|---|---|---|---|---|
| **MLP + Fourier** | **0.125** | **+0.991** | **20/27** | **1.01** | −0.00006 | **PASS ✅** |
| SIREN | 0.276 | +0.883 | 9/27 | 0.84 | +0.00008 | FAIL ❌ |

### The two findings that matter more than the pass

**1. The scoreboard earned its keep immediately.** All 7 MLP failures localise to one physical
regime — the **superficial MU (depth 8) directly under the electrode**, i.e. the *strongest*
signals — where amplitude is under-predicted **5–8×** (`p2p ratio 0.13–0.22`). Suspect: the
`asinh` target compression (added to stop near-source points dominating the loss) over-corrected.
**φ rel-L2 = 0.125 is one averaged number that could never have revealed this.** The MUAP
benchmark localised it to a regime in a single table. *This is the Phase-1 thesis, demonstrated.*

**2. We did NOT reproduce the prior SIREN finding — honestly.** Prior work: SIREN wins pointwise,
loses on MUAPs (jaggedness ⇒ amplified by SFAP ∝ φ″). Here SIREN lost on **both**, because it
**overfits 128×** at 64 electrodes (train 0.00019 / val 0.02445). Its Δjaggedness ≈ 0 — at this
scale it's **overfitting, not oscillating**. Prior used **2016** electrodes; its best small-data
variant was **SIREN+relcoords**, untested here. ⇒ **"MLP+Fourier is better" is unsettled beyond
N=64.** Retest at full scale on the cluster.

## What's proven about the pipeline

- **FEM → dataset → GPU training → MUAP scoreboard** works end-to-end, with benchmark electrodes
  genuinely held out (generalisation, not a fit check).
- **The cluster split is verified:** `emgforge.synthesis` + `metrics` import **no FEM stack**
  (dolfinx/mpi4py/petsc4py absent), while `emgforge.mri.core.fem_solver` pulls all three ⇒
  dataset-gen stays local, **training *and* MUAP scoring are cluster-portable**. Protect that.

## Next (in order)

1. **Human decision:** with speed out (Gate 0b), is the goal differentiability, mesh-free
   deployment, or cross-subject generality? This changes what success means before MRI.
2. Fix the near-field amplitude bug (gentler/learned compression, per-decade loss weighting, or
   log|φ| + sign head) → re-run Gate 2; target ≥0.94 in 27/27, not 20/27.
3. **E2.2 data-scaling** 64 → 256 → 2048 + **SIREN+relcoords** — the first cluster job.
4. **E2.4** multi-electrode grid coherence (propagation/IZ intact across the array).
5. Only then Phase 3 (MRI).

## Artifacts

`_results/neural_field/`: `muap_bench_cyl.npz` (the frozen scoreboard) · `cyl_elec_64.npz`
(dataset, 4.9 MB) · `cyl_{mlp,siren}.pt` · `score_cyl_{mlp,siren}.npz` · `phase2_final.png`
