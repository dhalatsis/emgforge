# Environments — which conda env runs which stage, and why it matters

Discovered the hard way on 2026-07-17: **no single local env runs the whole pipeline.**
Two stages failed under the "obvious" env before this was pinned down. Read this before
running anything, and before costing a cluster port.

| stage | script | env | hard requirement |
|---|---|---|---|
| mesh generation (per subject) | `emgforge.mri.core.build_mesh` | **`scifem`** | `pytetwild` (absent from `fenicsx-env`) |
| FEM dataset generation | `12_gen_mri_dataset.py` | **`fenicsx-env`** | dolfinx; reads a *cached* `.msh`, so no pytetwild |
| MUAP benchmark build | `11_build_muap_bench_mri.py` | **`fenicsx-env`** | dolfinx (geometry only) |
| training | `03_train_cyl.py` | either (**use `fenicsx-env`**) | torch only |
| MUAP scoring | `04/13_score_bench*.py` | **`fenicsx-env`** | see below |

```
fenicsx-env   numpy 1.26.4   torch 2.3.1+cu121   dolfinx 0.7.3   cuda True   <- Gate 1-4 ran here
scifem        numpy 2.2.6    torch 2.10.0+cu128  dolfinx 0.8.0   cuda True   <- pytetwild only
```

## The two failures, so nobody re-derives them

Running the scorer under `scifem` fails twice, and **both failures are impossible in the env
Gate 4 actually used** — which is how we know Gate 1–4 were produced under `fenicsx-env`:

1. `torch.load` — torch ≥2.6 flipped `weights_only` to `True`, rejecting the numpy arrays
   (`co`/`cs`/`val_elec`) in our own checkpoints. Fixed: explicit `weights_only=False`.
2. `ndarray.ptp()` — removed in numpy 2.0. Fixed: `np.ptp(x)` everywhere (numerically
   identical, verified).

Both are now version-agnostic, so the scripts run in either env. **But that does not make
results comparable across envs.** Any number compared against the Gate 1–4 baselines must be
produced under `fenicsx-env`, or the comparison is confounded by a torch minor version.

## Meshes: WR's recipe is unrecoverable (2026-07-17)

`save_metadata` never recorded the build flags, and `target_z` turns out not to be the lever:

| DH build | nodes | tets |
|---|---|---|
| `--target-z 1.0` (default) | 21311 | 101663 |
| `--target-z 1.5` (WR's z spacing) | 19393 | 93695 |
| **forearm_WR.msh (actual)** | **10570** | **42315** |

Resampling z 6.0→1.5 vs 6.0→1.0 moves the count 9% — nowhere near WR's 2× gap. WR's coarseness
came from `--edge-length`/`--surface-faces` (back-solving the tet ratio: edge_length ≈ 0.026 vs
the 0.02 default), and those were never stored. **WR's mesh cannot be reproduced.**

Consequence for cross-subject: do **not** try to match WR. Build all four subjects on one
*recorded* recipe (defaults, ~20k nodes) as a separate set with its own MUAP benchmark, and
leave the existing WR mesh + `muap_bench_mri.npz` untouched as the fixed-anatomy Gate 1–4
result. Two benchmarks, nothing invalidated. Otherwise mesh resolution is confounded with
subject identity.

## Implication for scaling

This three-way split is an argument *against* a cluster port, not for it. Reproducing it on a
remote node means provisioning dolfinx 0.7.3 + numpy<2 + torch<2.6 *and* a pytetwild env — and
we just spent an afternoon getting it right on a machine we control. Meanwhile the measured
costs say every experiment fits on this workstation (12 cores, 62 GB, one 2080 Ti). See
`scripts/neural_field/cluster/README.md`.
