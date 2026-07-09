# Running the Pipeline

This note documents how to run the mesh → FEM → voxel pipeline, plus common parameters to vary.

## Quick Start (single sample)

```bash
cd /path/to/emgforge
export PYTHONPATH=src

# Pick one sample (assumes mesh + meta already exist)
SAMPLE=sample_000000
OUT_ROOT=${DATA_ROOT:-./data}/generated_meshes

python scripts/01_solve_fem.py \
  --mesh "$OUT_ROOT/meshes/${SAMPLE}.msh" \
  --meta "$OUT_ROOT/metadata/${SAMPLE}.json" \
  --out_dir "$OUT_ROOT/fem/${SAMPLE}" \
  --source_mode gaussian \
  --source_sigma 3

python scripts/03_voxelize.py \
  --meta "$OUT_ROOT/metadata/${SAMPLE}.json" \
  --fem_summary "$OUT_ROOT/fem/${SAMPLE}/summary.json" \
  --fem_u "$OUT_ROOT/fem/${SAMPLE}/u.npy" \
  --out "$OUT_ROOT/voxel/${SAMPLE}.npz" \
  --nx 48 --ny 48 --nz 96 \
  --sigma_vox 1.5
```

## End-to-end using `pipeline.sh`

```bash
./pipeline.sh \
  --steps fem,voxel \
  --sample sample_000000 \
  --out-root ${DATA_ROOT:-./data}/generated_meshes \
  --nx 48 --ny 48 --nz 96 \
  --sigma-vox 1.5 \
  --source-mode gaussian \
  --source-sigma 3 \
  --source-cyl '1.,0.,0.5' \
  --source-layer skin \
  --source-margin 0.0 \
  --use-fem-source 1
```

`pipeline.sh` supports:
- `--steps` (mesh | fem | voxel | combinations)
- `--source-cyl` for fixed normalized cylindrical source placement
- `--source-layer` (skin | muscle) and `--source-margin`
- `--source-mode` (gaussian | point)
- `--source-sigma` for Gaussian FEM source width
- `--nx --ny --nz`, `--sigma-vox` for voxel grid + source smoothing

## Mesh generation

Mesh parameters and sampling ranges are defined in `scripts/00_generate_meshes.py`:
- `sample_parameters(...)` contains the normalized ranges for radii/thicknesses.
- CLI flags control output count, scaling, and mesh sizing.

Example:
```bash
python scripts/00_generate_meshes.py \
  --n 100 \
  --seed 0 \
  --factor 40 \
  --length_ratio_min 6 --length_ratio_max 6 \
  --mesh_char_factor 0.2 \
  --refine_on skin \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes
```

### Chunked mesh generation (PBS / parallel)

For HPC runs you can generate disjoint chunks of a single global sample set by using:

- `--n`: global total \(N\) samples (e.g. 500)
- `--start_idx`: start index into \([0, N)\)
- `--count`: number of meshes to generate from that start

Example: generate meshes 100–149 (50 meshes) out of a global 500-sample set:

```bash
python scripts/00_generate_meshes.py \
  --n 500 --seed 1 \
  --start_idx 100 --count 50 \
  --factor 40 \
  --length_ratio_min 6 --length_ratio_max 6 \
  --mesh_char_factor 0.2 \
  --refine_on skin \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes \
  --verbose
```

Each chunk writes its own manifest: `manifest_<start>_<end>.csv` under `--out_dir`.

## FEM solving (`scripts/01_solve_fem.py`)

Key parameters:
- `--source_mode gaussian|point`
- `--source_sigma` (Gaussian std in mesh units)
- `--source_cyl r_norm,theta_deg,z_norm` (normalized cylindrical source placement)
- `--source_layer skin|muscle`
- `--source_margin` (mesh units)

Example (fixed source in muscle for testing):
```bash
python scripts/01_solve_fem.py \
  --mesh ${DATA_ROOT:-./data}/generated_meshes/meshes/sample_000000.msh \
  --meta ${DATA_ROOT:-./data}/generated_meshes/metadata/sample_000000.json \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes/fem/sample_000000 \
  --source_mode gaussian \
  --source_sigma 3 \
  --source_cyl '0.6,0,0.5' \
  --source_layer muscle \
  --source_margin 0.5
```

Outputs per sample:
- `u.npy` (solution vector)
- `source.npy` (FEM source vector, Gaussian mode)
- `summary.json` (source info + stats)
- `plots/` (optional slices if `--plots` is used)

## Voxelization (`scripts/03_voxelize.py`)

Key parameters:
- `--nx --ny --nz`: grid resolution
- `--sigma-vox`: Gaussian source width (voxel units)
- `--fem_u`: pass FEM solution for target `u`
- `--fem_summary`: provides `source_point`, and `fem_source` path when available
- `--fem_source`: (optional) sample FEM source onto grid

Example:
```bash
python scripts/03_voxelize.py \
  --meta ${DATA_ROOT:-./data}/generated_meshes/metadata/sample_000000.json \
  --fem_summary ${DATA_ROOT:-./data}/generated_meshes/fem/sample_000000/summary.json \
  --fem_u ${DATA_ROOT:-./data}/generated_meshes/fem/sample_000000/u.npy \
  --fem_source ${DATA_ROOT:-./data}/generated_meshes/fem/sample_000000/source.npy \
  --out ${DATA_ROOT:-./data}/generated_meshes/voxel/sample_000000.npz \
  --nx 48 --ny 48 --nz 96 \
  --sigma_vox 1.5
```

The output `.npz` contains:
- `sigma` (z,y,x,6), `source`, `mask`, optional `u`, optional `source_fem`
- `grid_shape`, `spacing`, `r_skin`, `length`, `source_point`

## QA notebook

`notebooks/voxel_viz.ipynb` includes:
- slice visualizations
- `u(z)` comparison across samples at a fixed normalized cylindrical point

Update `VOX_DIR`, `R_NORM`, and `THETA_DEG` in the notebook QA cell before running.
