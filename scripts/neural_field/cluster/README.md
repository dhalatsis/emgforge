# The cluster job: anatomy-conditioning (fat × pennation) + MRI electrode sweep

**Operational contract: `HANDOFF.md`. The DAG as code: `experiment.py` (run it to print).**
This file is the *why*.

## Why this one IS a cluster job (when the electrode sweeps were not)

An earlier version of this file argued — correctly — that the **electrode sweeps don't need a
cluster**: fixed anatomy means the FEM cache holds, a 2048-electrode MRI sweep is ~15 min, and
the whole target/SIREN/N-scaling question got settled locally for free (see
`../../docs/learned_vc/RESULTS.md`). That still stands.

What changed is the *experiment*. Cross-**fat** and cross-**pennation** are **varying anatomy** —
every `r_fat` is a new mesh, every pennation a new σ-assembly + solve. That is Regime B (Gate
0b): the cell-id cache is void and meshing dominates, ~**594× the surrogate's net cost per
sample**. It is also embarrassingly parallel across the 36 anatomies. That combination — genuinely
expensive, trivially parallel — is exactly what a CPU cluster is for. The GPU training that
follows is cheap; it rides along on the GPU nodes.

So the split you asked for is the right one: **FEM generation fans out on CPU, training on GPU,
shared filesystem.**

## One env, not the three-way split

The earlier "the env is the real cost" objection was about the MRI meshing path (pytetwild). This
experiment **meshes cylinders via gmsh** and reuses the cached WR mesh for the MRI axis, so there
is **no pytetwild** — the whole pipeline runs in a single `fenicsx-env` (dolfinx 0.7.3 + gmsh 4.13
+ torch 2.3.1). `environment.yml` reproduces it. That removes the main provisioning risk.

## The matrix (see `experiment.py`)

| axis | grid | mechanism |
|---|---|---|
| fat | `r_fat ∈ {36,38,40,42,44,46}` mm | new gmsh mesh per value (the cost driver) |
| pennation | `{0,5,10,15,20,25}°` | rotates the muscle σ tensor (same mesh, new solve) |
| electrode | 96 per anatomy | source position |
| MRI electrode | N ∈ {256,512,1024,2048} | single WR anatomy, cheap solves |

**Held-out benchmark anatomies** sit at half-steps interior to the fat/pennation grid
(interpolation, not extrapolation); `experiment.py` asserts the train grid never collides with
them. Models are judged on **MUAP reconstruction** (Gate 5: amp-weighted r ≥ 0.94 on held-out
anatomies), **never on φ error** — φ is non-monotonic and 36%-noisy (RESULTS.md).

## Settled design rules (do not re-litigate)

- Judge on the MUAP benchmark, reporting the amp-weighted mean **and** the worst detectable
  config. The gate statistic, not the model, decides most verdicts.
- Never rank by φ.
- `r_fat > r_muscle (=35)` is required (guarded); pennation is a σ-rotation with z-aligned
  benchmark fibres (`21_build_bench_anatomy.py` header).
- **No FEM import may leak into the trainer or scorer** — the GPU stages stay dolfinx-free
  (verified). This is what lets them run on GPU nodes at all.

## Future direction (NOT this job)

**Leave-one-subject-out cross-subject** (WR/DH/AG/Kostia) is unblocked — the label scheme is
shared (FCU = label 8), meshing a new subject works (~69 s), and AG's internal-fat-as-muscle bug
is fixed. It is a *different* scientific claim (across real subjects, not synthetic anatomy) and a
*different* env (MRI meshing needs `scifem`/pytetwild). Parked deliberately; pick it up after the
anatomy grid if the conditioning generalises. Four subjects can falsify cross-subject transfer,
not establish it.
