#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or CLI flags)
PYTHONPATH_DEFAULT="src"
STEPS_DEFAULT="all"    # all | mesh | fem | voxel | mesh,fem | fem,voxel | mesh,voxel
N_MESH=1
SEED=0
OUT_ROOT="${DATA_ROOT:-./data}/generated_meshes"
NX=48
NY=48
NZ=96
SIGMA_VOX=1.5
VOX_SUBDIR="voxel"
SOURCE_MODE="gaussian"
SOURCE_SIGMA=3
USE_FEM_SOURCE=0
SOURCE_CYL="1., 0., 0.5"
SOURCE_LAYER="skin"
SOURCE_MARGIN=0.0
# Electrode configuration (new in FEM v2.0.0)
RETURN_MODE="volumetric"
GROUND_MODE="opposite"
GROUND_CYL=""
GROUND_RADIUS=5.0
GROUND_POINTS=256
ELECTRODE_INSET=4.0
NATIVE_POINT_SOURCE=1
REFINE_ON="skin"
MESH_CHAR=0.2
FACTOR=40.0
LENGTH_RATIO_MIN=6.0
LENGTH_RATIO_MAX=6.0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --steps STEPS          Steps to run: all|mesh|fem|voxel or comma-list (default: $STEPS_DEFAULT)
  --sample SAMPLE        Sample id (e.g., sample_000000) (required)
  --out-root PATH        Base output dir (default: $OUT_ROOT)
  --nx N                 Voxel nx (default: $NX)
  --ny N                 Voxel ny (default: $NY)
  --nz N                 Voxel nz (default: $NZ)
  --sigma-vox S          Voxel Gaussian sigma in voxels (default: $SIGMA_VOX)
  --vox-subdir NAME      Voxel output subdir under out-root (default: $VOX_SUBDIR)
  --source-mode MODE     gaussian|point (default: $SOURCE_MODE)
  --source-sigma S       Gaussian sigma in mesh units (FEM) (default: $SOURCE_SIGMA)
  --use-fem-source 0/1   If 1, pass FEM source vector to voxelizer when available (default: $USE_FEM_SOURCE)
  --source-cyl SPEC      Normalized r_norm,theta_deg,z_norm for source (comma-separated). If set, overrides angle/eps placement.
  --source-layer LAYER   skin|muscle (clamp radial placement when source-cyl is used) (default: $SOURCE_LAYER)
  --source-margin M      Margin in mesh units from layer boundaries when source-cyl is used (default: $SOURCE_MARGIN)
  --return-mode MODE     volumetric|localized - current return mode (default: $RETURN_MODE)
  --ground-mode MODE     opposite|distal|cylindrical - ground placement for localized mode (default: $GROUND_MODE)
  --ground-cyl SPEC      r_norm,theta_deg,z_norm for cylindrical ground mode (comma-separated)
  --ground-radius R      Ground electrode sampling radius in mesh units (default: $GROUND_RADIUS)
  --ground-points N      Number of ground electrode sampling points (default: $GROUND_POINTS)
  --electrode-inset D    Inset from boundaries for electrode placement (default: $ELECTRODE_INSET)
  --native-point-source  1=native point source (scifem-free, default), 0=scifem (must be installed) (default: $NATIVE_POINT_SOURCE)
  --mesh-char F          Mesh.CharacteristicLengthFactor (default: $MESH_CHAR)
  --refine-on MODE       skin|all_interfaces (default: $REFINE_ON)
  --factor F             radius factor for mesh params (default: $FACTOR)
  --lr-min F             length_ratio_min (default: $LENGTH_RATIO_MIN)
  --lr-max F             length_ratio_max (default: $LENGTH_RATIO_MAX)
  --n-mesh N             number of meshes to generate (default: $N_MESH)
  --seed N               seed (default: $SEED)
  -h, --help             Show this help

Env:
  PYTHONPATH (default: $PYTHONPATH_DEFAULT)

EOF
}

# Parse args
STEPS="$STEPS_DEFAULT"
SAMPLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --nx) NX="$2"; shift 2;;
    --ny) NY="$2"; shift 2;;
    --nz) NZ="$2"; shift 2;;
    --sigma-vox) SIGMA_VOX="$2"; shift 2;;
    --vox-subdir) VOX_SUBDIR="$2"; shift 2;;
    --source-mode) SOURCE_MODE="$2"; shift 2;;
    --source-sigma) SOURCE_SIGMA="$2"; shift 2;;
    --use-fem-source) USE_FEM_SOURCE="$2"; shift 2;;
    --source-cyl) SOURCE_CYL="$2"; shift 2;;
    --source-layer) SOURCE_LAYER="$2"; shift 2;;
    --source-margin) SOURCE_MARGIN="$2"; shift 2;;
    --return-mode) RETURN_MODE="$2"; shift 2;;
    --ground-mode) GROUND_MODE="$2"; shift 2;;
    --ground-cyl) GROUND_CYL="$2"; shift 2;;
    --ground-radius) GROUND_RADIUS="$2"; shift 2;;
    --ground-points) GROUND_POINTS="$2"; shift 2;;
    --electrode-inset) ELECTRODE_INSET="$2"; shift 2;;
    --native-point-source) NATIVE_POINT_SOURCE="$2"; shift 2;;
    --mesh-char) MESH_CHAR="$2"; shift 2;;
    --refine-on) REFINE_ON="$2"; shift 2;;
    --factor) FACTOR="$2"; shift 2;;
    --lr-min) LENGTH_RATIO_MIN="$2"; shift 2;;
    --lr-max) LENGTH_RATIO_MAX="$2"; shift 2;;
    --n-mesh) N_MESH="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SAMPLE" ]]; then
  echo "Error: --sample is required (e.g., sample_000000)"; exit 1;
fi

export PYTHONPATH="${PYTHONPATH:-$PYTHONPATH_DEFAULT}"

# Normalize steps
IFS=',' read -ra STEPS_ARR <<< "$STEPS"
WANT_MESH=0; WANT_FEM=0; WANT_VOX=0
for s in "${STEPS_ARR[@]}"; do
  case "$s" in
    all) WANT_MESH=1; WANT_FEM=1; WANT_VOX=1;;
    mesh) WANT_MESH=1;;
    fem) WANT_FEM=1;;
    voxel) WANT_VOX=1;;
    meshfem|mesh,fem) WANT_MESH=1; WANT_FEM=1;;
    femvoxel|fem,voxel) WANT_FEM=1; WANT_VOX=1;;
    meshvoxel|mesh,voxel) WANT_MESH=1; WANT_VOX=1;;
    *) echo "Unknown step token: $s"; exit 1;;
  esac
done

# Paths
MESH_DIR="$OUT_ROOT/meshes"
META_DIR="$OUT_ROOT/metadata"
FEM_DIR="$OUT_ROOT/fem/$SAMPLE"
VOX_DIR="$OUT_ROOT/$VOX_SUBDIR"
MANIFEST="$OUT_ROOT/manifest.csv"
MESH_PATH="$MESH_DIR/${SAMPLE}.msh"
META_PATH="$META_DIR/${SAMPLE}.json"
VOX_OUT="$VOX_DIR/${SAMPLE}.npz"
FEM_SUMMARY="$FEM_DIR/summary.json"
FEM_U="$FEM_DIR/u.npy"
FEM_SOURCE="$FEM_DIR/source.npy"

run_mesh() {
  python scripts/00_generate_meshes.py \
    --n "$N_MESH" \
    --seed "$SEED" \
    --factor "$FACTOR" \
    --length_ratio_min "$LENGTH_RATIO_MIN" \
    --length_ratio_max "$LENGTH_RATIO_MAX" \
    --mesh_char_factor "$MESH_CHAR" \
    --refine_on "$REFINE_ON" \
    --out_dir "$OUT_ROOT" \
    --limit 1 \
    --verbose
}

run_fem() {
  cmd=(python scripts/01_solve_fem.py
    --mesh "$MESH_PATH"
    --meta "$META_PATH"
    --out_dir "$FEM_DIR"
    --plots
    --source_mode "$SOURCE_MODE"
    --source_sigma "$SOURCE_SIGMA"
    --return_mode "$RETURN_MODE"
    --ground_mode "$GROUND_MODE"
    --ground_radius "$GROUND_RADIUS"
    --ground_points "$GROUND_POINTS"
    --electrode_inset "$ELECTRODE_INSET"
  )
  if [[ -n "$SOURCE_CYL" ]]; then
    cmd+=(--source_cyl "$SOURCE_CYL" --source_layer "$SOURCE_LAYER" --source_margin "$SOURCE_MARGIN")
  fi
  if [[ -n "$GROUND_CYL" ]]; then
    cmd+=(--ground_cyl "$GROUND_CYL")
  fi
  if [[ "$NATIVE_POINT_SOURCE" -eq 1 ]]; then
    cmd+=(--native_point_source)
  else
    cmd+=(--no-native_point_source)
  fi
  "${cmd[@]}"
}

run_voxel() {
  cmd=(python scripts/neural_field/03_voxelize.py
    --meta "$META_PATH"
    --fem_summary "$FEM_SUMMARY"
    --fem_u "$FEM_U"
    --out "$VOX_OUT"
    --nx "$NX" --ny "$NY" --nz "$NZ"
    --sigma_vox "$SIGMA_VOX"
  )
  if [[ "$USE_FEM_SOURCE" -eq 1 && -f "$FEM_SOURCE" ]]; then
    cmd+=(--fem_source "$FEM_SOURCE")
  fi
  "${cmd[@]}"
}

# Execute
if [[ "$WANT_MESH" -eq 1 ]]; then
  echo "[mesh] generating mesh + meta -> $MESH_PATH"
  run_mesh
fi

if [[ "$WANT_FEM" -eq 1 ]]; then
  echo "[fem] solving FEM -> $FEM_DIR"
  run_fem
fi

if [[ "$WANT_VOX" -eq 1 ]]; then
  echo "[voxel] voxelizing -> $VOX_OUT (subdir=$VOX_SUBDIR grid=${NX}x${NY}x${NZ})"
  run_voxel
fi

echo "Done. Steps: mesh=$WANT_MESH fem=$WANT_FEM voxel=$WANT_VOX"