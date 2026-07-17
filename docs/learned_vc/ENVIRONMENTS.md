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

## Implication for scaling

This three-way split is an argument *against* a cluster port, not for it. Reproducing it on a
remote node means provisioning dolfinx 0.7.3 + numpy<2 + torch<2.6 *and* a pytetwild env — and
we just spent an afternoon getting it right on a machine we control. Meanwhile the measured
costs say every experiment fits on this workstation (12 cores, 62 GB, one 2080 Ti). See
`scripts/neural_field/cluster/README.md`.
