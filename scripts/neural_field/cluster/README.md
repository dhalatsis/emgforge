# Cluster scaling — the big-scale experiments

## Why scale is the open question

Every conclusion so far was measured at **55–218 VC solutions**, i.e. ~3–10% of the closed
Exp 1's **2016 electrodes**. At that size we are characterising **inductive bias, not
capacity** — several rankings could invert. Specifically unsettled:

| finding | measured at | risk at scale |
|---|---|---|
| `φ·(r+r₀)` > `raw` > `asinh` | 55 VC (cylinder, 24× range) | the factorisation's edge came partly from *substituting for missing data* — with 30× more, `raw` may catch up |
| SIREN overfits 128× → loses | 55 VC | overfitting is a **small-data** artifact; SIREN may win at 2048 (prior work's claim) |
| `\|φ\|^α` helps monotonically | 55 VC | a re-weighting patch for scarcity — may become unnecessary |

**The whole point of scaling: find out which of these were real and which were artifacts of N.**

## The split (verified, protect it)

| stage | needs | runs |
|---|---|---|
| FEM dataset generation | dolfinx + mesh + segmentation | **local** (CPU) |
| training | torch only + `.npz` | **cluster** (GPU) |
| **MUAP scoring** | `emgforge.synthesis` — **no FEM stack** | **either** ✅ |

Verified: `emgforge.synthesis` + `metrics` import **no** dolfinx/mpi4py/petsc4py, while
`emgforge.mri.core.fem_solver` pulls all three. So the cluster can rank checkpoints by **MUAP
quality in situ**, not by val loss — which matters, because φ error is *anti-correlated* with
MUAP quality (measured 4×).

**Rule: no FEM import may leak into the trainer or the scorer.**

## Cost (measured, not estimated)

| | solve | 2048-electrode dataset |
|---|---|---|
| cylinder (194k nodes) | 7.7 s | **~4.5 h** |
| **WR / MRI (10.5k nodes)** | **0.39 s** | **~25 min** ⭐ |

Hybrid near-field sampling adds only ~0.3 s/electrode. **The MRI dataset is 20× cheaper — do
the scaling study there**, not on the cylinder.

## The artifact to ship

Self-contained, no mesh / no dolfinx / no MRI on the cluster:

```
dataset.npz        points, electrodes, phi, near_points, near_phi   (muscle-only)
muap_bench_*.npz   the frozen scoreboard (configs + truth + properties)
config.yaml        arch / target / wpow / epochs
```

Back: `checkpoint.pt` + the MUAP score table.

## The race to run

| axis | values |
|---|---|
| N (VC solutions) | 64 → 256 → 1024 → **2048** (the data-scaling curve is the headline) |
| target | `raw`, `φ·(r+r₀)`, `asinh`, `raw+\|φ\|¹` |
| arch | MLP+Fourier, SIREN, **SIREN+relcoords** (untested, was prior work's best small-data) |

That is 4 × 4 × 3 = 48 runs; each ~200 s–20 min on one GPU. **Prioritise the N-sweep for
`φ·(r+r₀)` and `raw`** — if they converge at 2048, the target question is closed and the
factorisation was only ever a small-data crutch.

## TODO before submitting (needs human input)

- [ ] scheduler — slurm? something else?
- [ ] how data reaches the node (scp / shared FS / object store?)
- [ ] is torch+CUDA provisioned, or do we ship an env spec?
- [ ] GPU count / walltime limits
- [ ] then: `submit.sh` + `env.yaml` (torch only — **never** dolfinx)
