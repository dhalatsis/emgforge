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

**Consequence (as first stated):** against a cached FEM a 0.15 s net is **~3×** ⇒ the speed case
looked dead. *Also a free ~18× for the existing pipeline — see `HANDOFF.md`.*

### ⚠️ CORRECTION — Gate 0b part 2: the cache breaks, and the speed case comes back

`07_break_the_cache.py`. The "~3×" was measured on **WR (10.5 k nodes, 0.39 s solve) with fixed
anatomy** — too narrow a claim. Caching only holds while **(mesh, query points)** are fixed:

| regime | FEM / sample | vs 0.15 s net |
|---|---|---|
| A · fixed anatomy, **small** mesh (WR 10.5 k) | 0.43 s | ~3× ← the original measurement |
| A · fixed anatomy, **big** mesh (cyl 194 k) | **7.75 s** | **52×** |
| B · **varying anatomy** (cache void) | **89.2 s** | **594×** |

**Two things break it:**
1. **Mesh size.** The cache removes point-location, so the *solve* dominates — and it scales with
   the mesh (WR 0.39 s vs cylinder **7.74 s**). The 3× was an artifact of a small mesh.
2. **Changing the anatomy** — the real one. New mesh ⇒ cache void, **plus** costs a fixed-anatomy
   sweep amortises to zero: **mesh generation 56.8 s** (uncacheable, dominant), σ re-assembly
   14.8 s, cell_ids 7.8 s, solve 9.8 s ⇒ **89.2 s for one sample of a new anatomy**.
3. (subtler) pennation **rotates the fibres** ⇒ query points move ⇒ cache void even on a fixed mesh.

**Revised conclusion:** caching kills the surrogate's speed case *for repeated electrodes on one
fixed, small mesh* — **and nowhere else**. The regimes that matter scientifically — pennation/fat
sweeps (prior Exp 2–3, where anatomy IS the condition), cross-subject, and **inverse problems**
(geometry is the optimisation variable ⇒ re-mesh every iteration, cache never applies) — are
**594×**. Speed and differentiability now point the same way: toward anatomy conditioning
(option D).

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

---

# Gate 4 — MRI learned VC (2026-07-17): PASS, and it settles the target question

`mri_elec_256_near_muscle` (256 electrodes, 218 train / 38 val, muscle-only + hybrid
near-field) → MLP+Fourier → scored on the 27 held-out frozen MRI benchmark configs
(21/27 detectable, 0.04–36 µV):

| target | φ rel-L2 | median r | ≥0.94 | **amp-wtd mean** | detectable min r / p2p |
|---|---|---|---|---|---|
| φ·(r+r₀) | 0.116 | +0.995 | **27/27** | +0.989 | 0.957 / 1.08 |
| **raw** | **0.081** | **+0.997** | **27/27** | **+0.995** | 0.977 / **1.03** |
| asinh | 0.082 | +0.993 | **27/27** | +0.995 | **0.985** / 1.09 |

## The finding: the targets converged — the whole investigation was data starvation

| tallest-peak p2p | cylinder (55 VC, starved) | MRI (218 VC, muscle-only) |
|---|---|---|
| asinh | **0.19** ✗ | **1.09** ✅ |
| φ·(r+r₀) | 0.97 | 1.08 |
| raw | 0.42 | 1.03 |
| **spread** | **5×** | **~6%** |

**`asinh` — the target blamed for the entire near-field amplitude bug — passes at 0.995 with
the best worst-case (0.985).** It was never wrong; the *data* was. Six training runs of target
engineering (asinh → φ·r → φ·(r+r₀) → raw → |φ|¹ → |φ|²) were all compensating for a dataset
that sampled **0.1% near-field** and **73% wrong tissue**.

**Winner: `raw`** — no transform at all. Best φ *and* best amp-weighted score. With the right
data, the clever priors are unnecessary.

> ### ⚠️ RETRACTION (same day, see the N-sweep below)
> **"Best φ" is not a supported claim.** Retraining the *identical* N=256 `raw` config gave
> φ rel-L2 **0.1097** where this run gave **0.0809** — 36% apart, from cuDNN/cuBLAS
> non-determinism alone (a fixed seed does not pin it). `raw` beat `asinh` by 2%
> (0.0809 vs 0.0824). **That margin is inside the noise; the φ ranking was never real.**
>
> What survives, and is strengthened: the amp-weighted MUAP score is stable to **0.1%**
> across the same retrain (0.995 → 0.996), so `raw` ≈ `asinh` at 0.995 is a genuine tie, and
> `phir0`'s 0.989 may be a real small deficit. The **convergence** conclusion — the targets
> are indistinguishable because the data, not the objective, was the problem — gets *stronger*
> if the residual differences are noise.

**This pre-answers the cluster N-sweep** at 256 electrodes: the prediction was that targets
converge with enough data and the investigation resolves to "we were data-starved". Demonstrated.

## What actually mattered (in order)

1. **Sample the right domain** — muscle only (73%/86% of points were in tissue no fibre occupies)
2. **Sample the right regime** — hybrid near-field (uniform-in-volume gave 0.1% near-field)
3. **Enough VC solutions** — 55 → 218
4. **Judge on MUAPs, amplitude-weighted** — φ error is anti-correlated with MUAP quality
5. *(the target/loss choice — irrelevant once 1–4 are right)*

---

# The MRI N-sweep (2026-07-17) — free, and it settles two things

The 256-electrode dataset's electrodes are i.i.d. random, so a **prefix is a valid smaller
draw**: the whole curve costs no new FEM. The frozen benchmark is held out by construction at
every N. All runs in `fenicsx-env` (see `ENVIRONMENTS.md` — this matters, `scifem` shifts
rel-L2 ~6%).

| N | φ rel-L2 | amp-wtd r | ≥0.94 | worst detectable r | p2p | gate |
|---|---|---|---|---|---|---|
| 32 | 0.336 | 0.938 | 17/27 | 0.808 | 1.02 | **FAIL** |
| 64 | 0.126 | 0.989 | 24/27 | 0.769 | 1.16 | PASS |
| **128** | **0.097** ← *best φ* | 0.994 | 26/27 | 0.899 | 1.06 | PASS |
| **256** | 0.110 | **0.996** | **27/27** | **0.975** | **1.01** | PASS |

## 1. φ error is the wrong referee — demonstrated twice, both by accident

**(a) φ is non-monotonic in N.** It ranks **N=128 above N=256**, while *every* MUAP measure
says 256 is strictly better (27/27 vs 26/27, worst case 0.975 vs 0.899, p2p 1.01 vs 1.06).
**Selecting on φ picks the worse model given more data.**

**(b) φ is ~300× less stable than the MUAP score.** Retraining the identical N=256 config:

| | Gate 4 run | rerun | spread |
|---|---|---|---|
| φ rel-L2 | 0.0809 | 0.1097 | **+36%** |
| amp-weighted MUAP r | 0.995 | 0.996 | **+0.1%** |
| worst detectable r | 0.977 | 0.975 | 0.2% |

Two models whose φ error differs by a third produce indistinguishable MUAPs. *This is the
Phase-1 thesis (SFAP ∝ ∫Vm·φ″, so φ error is not the quantity of interest) confirmed by
evidence nobody designed an experiment to collect.* It also retroactively justifies the
frozen benchmark's existence.

## 2. The headline saturates at N=64; the TAIL does not

amp-weighted mean: 0.989 → 0.994 → 0.996 (flat after 64). Worst detectable config:
**0.769 → 0.899 → 0.977 — climbing monotonically, not saturated at 256.**

So "is more N worth it?" has *two different answers depending on which number you read*, and
the mean is the misleading one. N=64 even has a **worse** worst case than N=32 (0.769 vs
0.808) while its weighted mean looks far better. **Gate 4's amp-weighted mean alone is an
insufficient gate — the worst detectable config must be reported with it.**

⇒ Extending to 1024 is justified (~45 min of local CPU). Not because the mean needs it —
because the tail might.
