# scripts/

CLI entry points, split by category (run from repo root; `emgforge` + `neural_field`
are pip-installed editable, so no `PYTHONPATH` needed):

- **`scripts/`** — the **emgforge** forward-model pipeline: parametric mesh generation + FEM
  solve (produces the volume-conductor solution).
- **`scripts/neural_field/`** — the **neural_field** ML/dataset pipeline: samples that FEM
  solution onto voxel grids / point clouds for neural-field & PINN training. Consumes the
  emgforge outputs; never the reverse.

## emgforge — forward-model pipeline

| Script | Purpose |
|---|---|
| `00_generate_meshes.py` | Generate a batch of parameterized 5-layer concentric-cylinder meshes via Gmsh OpenCASCADE (LHS sampling over geometry params). Writes `.msh` + metadata `.json`. |
| `01_solve_fem.py` | FEM solve of the conductivity equation on one mesh with a configurable electrode (volumetric / localized return, `opposite` / `distal` / `cylindrical` ground modes). Outputs `u.npy`, `source.npy`, `summary.json`. Needs `mpirun -n 1`. |
| `02_merge_manifests.py` | Aggregate per-sample metadata into a single manifest. |
| `generate_reference_mesh.py` | Single-shot reference-mesh generator (called by the FEM-worker regression suite). |
| `mri/build_fiber_config.py` | Per-muscle fibre config (PCA + centerline) for the MRI fibre-aligned σ. |

## neural_field — ML/dataset pipeline (`scripts/neural_field/`)

Consume the emgforge FEM outputs (`u`, `source`) → training datasets.

| Script | Purpose |
|---|---|
| `neural_field/03_voxelize.py` | Sample the FEM `u` onto a regular voxel grid (default 48×48×96 or 96×96×192), with the conductivity tensor and mask. Outputs `.npz`. |
| `neural_field/06_generate_pointcloud_electrode.py` | Point-cloud dataset by sweeping electrode position over a mesh (current preferred format — scattered interior + boundary points, PINN-ready source field + normals). |
| `neural_field/06b_generate_pointcloud_pinn_only.py` | PINN-only variant of `06_` (collocation + boundary only, no FEM labels). |
| `neural_field/07_generate_pointcloud_pennation.py` | Same, sweeping muscle fibre pennation angle. |
| `neural_field/08_generate_pointcloud_fat.py` | Same, sweeping fat-layer thickness. |

Pipeline order across the two categories: **mesh → solve** (emgforge) **→ voxelize /
pointcloud-extract** (neural_field). `06_`/`07_`/`08_` are alternative sweep endings; they
don't run together.

## `legacy/` — superseded by the pointcloud era

| Script | Replaced by |
|---|---|
| `legacy/04_generate_electrode_dataset.py` | `neural_field/06_generate_pointcloud_electrode.py` |
| `legacy/05_generate_pennation_dataset.py` | `neural_field/07_generate_pointcloud_pennation.py` |

These produced the **voxel-grid** datasets that drove the neural-field surrogate experiments
(the external `neural_field/` training). The pointcloud format is the current preferred input.

## Shell wrappers (at repo root)

| Wrapper | Purpose |
|---|---|
| `../pipeline.sh` | End-to-end one-sample driver: mesh → fem → voxel. |
| `../run.sh` | Batch wrapper around `pipeline.sh`. |

## See also
- `docs/pipeline/RUNNING.md` — usage examples + parameter reference.
- `docs/pipeline/FEM.md` — the FEM problem statement and electrode configurations.
- `docs/pipeline/GEOMETRY_GENERATION.md` — the parametric-cylinder design and LHS sampling.
