# Cluster Migration Notes

## Recent Development Summary

### FEM Pipeline Enhancements (v2.0.0)

**Electrode Configuration**
- Bipolar electrode support with `--return_mode localized`
- Ground placement modes: `opposite`, `distal`, `cylindrical`
- Native FEniCSx point source (no scifem dependency)

**Pennation Angle Support**
- Conductivity tensor rotation via Rodrigues' formula
- CLI: `--pennation_angle` in `scripts/01_solve_fem.py`
- Rotates muscle fiber orientation relative to longitudinal axis

### Point Cloud Generation Module

New `src/emgop/pointcloud/` module for neural field and PINN training data.

**Scripts:**
| Script | Purpose |
|--------|---------|
| `06_generate_pointcloud_electrode.py` | Electrode position sweep on single mesh |
| `07_generate_pointcloud_pennation.py` | Pennation angle sweep on single mesh |

**Sampling Strategies:**
- `uniform` - Random sampling in cylinder
- `near_electrode` - Concentrated near electrode(s)
- `stratified_tissue` - Equal per tissue layer
- `surface_biased` - Higher density near skin
- `mixed` - Combination of above

**Output Format (`.npz`):**
```
points:           (N, 3)   - query coordinates
u:                (N,)     - FEM solution
sigma:            (N, 6)   - conductivity [xx,yy,zz,xy,xz,yz]
tissue_labels:    (N,)     - tissue ID (0-4)
source_field:     (N,)     - source term f(x) for PINN
boundary_points:  (M, 3)   - boundary coords for PINN
boundary_normals: (M, 3)   - outward normals
electrode_position: (3,)   - source electrode
```

---

## Cluster Deployment Plan

### Environment Setup

```bash
# Create conda environment
conda create -n fenicsx-env python=3.11
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich petsc4py gmsh

# Clone and setup
git clone <emgforge-repo-url>
cd emgforge
export PYTHONPATH=src
```

### Dataset Generation Commands

**Electrode Sweep (recommended: 500-1000 samples)**
```bash
python scripts/06_generate_pointcloud_electrode.py \
    --mesh /path/to/sample_000000.msh \
    --meta /path/to/sample_000000.json \
    --out_dir ./datasets/electrode_sweep \
    --n_electrodes 500 \
    --n_points 10000 \
    --n_boundary 2000 \
    --sampling_strategy near_electrode \
    --seed 42
```

**Pennation Sweep (recommended: 20-50 angles)**
```bash
python scripts/07_generate_pointcloud_pennation.py \
    --mesh /path/to/sample_000000.msh \
    --meta /path/to/sample_000000.json \
    --out_dir ./datasets/pennation_sweep \
    --n_angles 20 \
    --angle_min 0 --angle_max 45 \
    --n_points 10000 \
    --n_boundary 2000 \
    --sampling_strategy uniform \
    --seed 42
```

### PBS Infrastructure

Dedicated PBS scripts in `pbs_scripts/pointcloud/`:

```bash
cd pbs_scripts/pointcloud

# Single reference mesh with 128 electrodes (8x16 grid)
./submit_electrode.sh

# Custom electrode density
N_ELECTRODES=64 GRID_Z=8 GRID_THETA=8 ./submit_electrode.sh

# Multi-mesh mode (500 meshes)
N_MESHES=500 MESHES_PER_TASK=5 MESH_NAME="" ./submit_electrode.sh
```

**Reference Mesh Generation:**
```bash
conda activate fenicsx-env
python scripts/generate_reference_mesh.py --verbose
```

Creates `reference_avg.msh` with average layer dimensions:
- Skin radius: 48.2 mm, Length: 240 mm
- Suitable for neural field training on fixed geometry

### Resource Estimates

| Dataset | Samples | Points/Sample | Time/Sample | Total Time |
|---------|---------|---------------|-------------|------------|
| Electrode sweep (128) | 128 | 10k+2k | ~60s | ~2-3h |
| Electrode sweep (500) | 500 | 10k | ~10s | ~1.5h |
| Pennation sweep | 20 | 10k | ~35s | ~12min |

*Note: Grid electrodes take longer due to bipolar config FEM solve complexity.*

---

## File Structure

```
emgforge/
├── src/emgop/
│   ├── fem/           # FEM solver, rotation, electrode config
│   ├── pointcloud/    # Point cloud generation module
│   ├── voxel/         # Voxelization (legacy, still functional)
│   └── meshing/       # Mesh generation
├── scripts/
│   ├── 01_solve_fem.py
│   ├── 06_generate_pointcloud_electrode.py
│   └── 07_generate_pointcloud_pennation.py
├── docs/
│   ├── pennation_angle.md
│   ├── native_point_source.md
│   └── cluster_migration.md
└── pbs_scripts/       # HPC job templates
```

## Next Steps

1. Transfer mesh files to cluster storage
2. Run small test batch (10 samples) to verify setup
3. Generate full electrode sweep dataset
4. Generate pennation sweep dataset
5. Train neural field model: `u(x,y,z | electrode_pos)` or `u(x,y,z | pennation_angle)`
