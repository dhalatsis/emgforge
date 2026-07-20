# Cluster handoff — anatomy-conditioning + MRI electrode experiment

**For the agent taking this to the cluster.** The physics is built, tested locally, and
scheduler-agnostic. Your job is the cluster-specific glue: finish the submit templates, install
the env, and launch. This doc is the contract.

## What the experiment does

Train ONE learned volume-conductor `φ = f(query_xyz, electrode_xyz, r_fat, pennation)` and test
whether it interpolates to fat/pennation anatomies held out from training. Plus the MRI
cross-electrode sweep as a second axis. Models are judged on **MUAP reconstruction** against a
frozen FEM benchmark, never on φ error (φ error is non-monotonic and 36%-noisy — see
`../../docs/learned_vc/RESULTS.md`).

Two anatomy axes on the cylinder:
- **fat** (`r_fat` ∈ {36,38,40,42,44,46} mm) — each is a NEW gmsh mesh (the cost driver).
- **pennation** (∈ {0,5,10,15,20,25}°) — rotates the muscle σ tensor (same mesh, new solve).

Held-out benchmark anatomies sit at half-steps interior to the grid (interpolation, not
extrapolation): fat {37,39,41,43} × pennation {2.5,7.5,12.5,17.5}. `experiment.py` asserts the
train grid never collides with them.

## The environment — one env does everything

`environment.yml` → `fenicsx-env`: **dolfinx 0.7.3 + gmsh 4.13 + torch 2.3.1/cu121, numpy 1.26**.
- **No pytetwild needed** (cylinders mesh via gmsh; MRI reuses the cached WR mesh).
- The `numpy<2` / `torch<2.6` pins are load-bearing: numpy 2 removed `ndarray.ptp`, torch 2.6
  flipped `torch.load(weights_only)` — both broke the scorer once. Do not bump them.
- For reproducibility on HPC, an Apptainer image built from this env is the safest route
  (the dolfinx/petsc/mpich stack is fragile across nodes).

## The split (verified — protect it)

| stage | needs | node |
|---|---|---|
| FEM dataset + benchmark gen | dolfinx + gmsh | **CPU** — the bottleneck, embarrassingly parallel |
| training + MUAP scoring | torch + numpy only (NO dolfinx) | **GPU** |

`emgforge.synthesis` imports no dolfinx, so the GPU stages are FEM-free. **Never let a dolfinx
import leak into `22_train_conditioned.py` / `23_score_conditioned.py` / `03`/`13`.**

## How to run it

Everything routes through one scheduler-agnostic dispatcher:

```
python scripts/neural_field/cluster/run_job.py --list                 # the 52-job DAG + array sizes
python scripts/neural_field/cluster/run_job.py --where cpu --index $I  # one CPU (FEM) job
python scripts/neural_field/cluster/run_job.py --where gpu --index $I  # one GPU (torch) job
python scripts/neural_field/cluster/run_job.py --name train_cyl        # one job by name
```

- Each job writes a `_results/neural_field/_cluster_state/<name>.done` sentinel on the shared FS.
- Idempotent: a job with a sentinel is skipped (safe to requeue).
- A GPU job checks its dependency sentinels; if a gen job hasn't finished it exits **rc=3** —
  requeue after the CPU stage completes, or use a scheduler dependency.

### Submit (fill in `<<...>>` in the templates)

`submit.pbs.template` / `submit.slurm.template` each define **stage 1 (CPU array)**; copy to a
`.gpu` variant with the GPU header block (shown at the bottom of each template) for **stage 2**.

```
# PBS
JID=$(qsub submit.pbs.cpu)
qsub -W depend=afterokarray:${JID}[] submit.pbs.gpu

# SLURM
JID=$(sbatch --parsable submit.slurm.cpu)
sbatch --dependency=afterok:${JID} submit.slurm.gpu
```

What you must fill in: queue/partition names, account, `walltime`, and the **env-activate line**
(`module load …` or `conda activate fenicsx-env` or `source …/activate`). Array sizes come from
`run_job.py --list` (currently `0-41` CPU, `0-9` GPU; they auto-track if you edit the grid).

## Cost (measured locally, one core)

- cylinder mesh build: **~49 s** (shared across the 6 pennations of a given fat → 6 builds).
- cylinder solve: **~13 s/electrode** (incl. near-field sampling), 194k-node mesh.
- 6×6 grid × 96 electrodes ≈ **36 × 96 × 13 s ≈ 12 CPU-hours**, fully parallel over 36 array
  tasks → minutes of wall-clock on the cluster.
- MRI solves are ~0.39 s each; the 2048 set is ~15 min on one core.
- GPU training: ~5–15 min per model.

## Validate ONE job before the full array

```
export PYTHONPATH=src
python scripts/neural_field/cluster/run_job.py --name gen_cyl_f38_p10   # a CPU job (~20 min)
python scripts/neural_field/cluster/run_job.py --name bench_cyl         # the benchmark (~10 min)
# then, after some gen jobs + bench exist:
python scripts/neural_field/cluster/run_job.py --name train_cyl         # GPU
python scripts/neural_field/cluster/run_job.py --name score_cyl         # GPU -> the result table
```

`score_cyl` prints the anatomy-benchmark table and PASS/FAIL (Gate 5: amp-weighted MUAP r ≥ 0.94
on held-out anatomies). `score_mri_<N>` does the same for the electrode sweep.

## What is NOT built (deliberately — decide if you need them)

- Submit scripts are **templates**, not runnable as-is (queue/account/env unknown to me).
- No multi-GPU / distributed training (models are small — one GPU is plenty).
- Pennation is modelled as **σ-tensor rotation only**; benchmark fibres stay z-aligned (see the
  header of `21_build_bench_anatomy.py`). Coupling fibre geometry to pennation is a documented
  extension, not done.
- The MRI axis conditions on electrode only (single anatomy); it does not share the cylinder's
  fat/pennation conditioning.
