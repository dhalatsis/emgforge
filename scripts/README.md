# scripts/

CLI entry points for the cylindrical-pipeline data generation flow. Run from repo root with `PYTHONPATH=src` (or use the shell wrappers `pipeline.sh` / `run.sh`).

## Numbered pipeline (current, active)

| Script | Purpose |
|---|---|
| `00_generate_meshes.py` | Generate a batch of parameterized 5-layer concentric-cylinder meshes via Gmsh OpenCASCADE (LHS sampling over geometry params). Writes `.msh` + metadata `.json`. |
| `01_solve_fem.py` | FEM solve of the conductivity equation on one mesh with a configurable electrode (volumetric / localized return, `opposite` / `distal` / `cylindrical` ground modes). Outputs `u.npy`, `source.npy`, `summary.json`. Needs `mpirun -n 1`. |
| `02_merge_manifests.py` | Aggregate per-sample metadata into a single manifest. |
| `03_voxelize.py` | Sample the FEM `u` onto a regular voxel grid (default 48×48×96 or 96×96×192), with the conductivity tensor and mask. Outputs `.npz`. |
| `06_generate_pointcloud_electrode.py` | Generate a point-cloud dataset by sweeping electrode position over a single mesh (the current preferred dataset format — scattered points + interior + boundary, with PINN-ready source field + normals). |
| `06b_generate_pointcloud_pinn_only.py` | PINN-only variant of `06_` (collocation points + boundary only, no FEM labels). |
| `07_generate_pointcloud_pennation.py` | Same but sweeping muscle fibre pennation angle. |
| `08_generate_pointcloud_fat.py` | Same but sweeping fat-layer thickness. |
| `generate_reference_mesh.py` | Single-shot reference-mesh generator (called by the FEM-worker regression suite). |

The numbering reflects pipeline order: **mesh → solve → voxelize / pointcloud-extract**. `06_`, `07_`, `08_` are alternative endings for different sweep axes; they don't run together.

## `legacy/` — superseded by the pointcloud era

| Script | Replaced by |
|---|---|
| `legacy/04_generate_electrode_dataset.py` | `06_generate_pointcloud_electrode.py` (voxel-grid → pointcloud format) |
| `legacy/05_generate_pennation_dataset.py` | `07_generate_pointcloud_pennation.py` |

These produce the **voxel-grid** datasets that drove the neural-field surrogate experiments (Exp 1–3 in `training/neural_field/`). The pointcloud format used by `06_`–`08_` is the current preferred input.

## Shell wrappers (at repo root)

| Wrapper | Purpose |
|---|---|
| `../pipeline.sh` | End-to-end one-sample driver: mesh → fem → voxel, with all electrode-config flags. |
| `../run.sh` | Batch wrapper around `pipeline.sh`, 4 concurrent samples (range 100–500 by default). |

## See also

- `pbs_scripts/` — cluster (PBS) job templates that wrap these scripts.
- `docs/pipeline/RUNNING.md` — full usage examples + parameter reference.
- `docs/pipeline/FEM.md` — the FEM problem statement and electrode configurations.
- `docs/pipeline/GEOMETRY_GENERATION.md` — the parametric-cylinder design and LHS sampling.
