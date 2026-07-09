#!/bin/bash
#
# Run FEM residual troubleshooting tests
#
# Usage:
#   ./run_all.sh [mesh_path] [meta_path]
#
# Example:
#   ./run_all.sh ${DATA_ROOT:-./data}/generated_meshes/meshes/sample_000001.msh \
#                ${DATA_ROOT:-./data}/generated_meshes/metadata/sample_000001.json

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Set up environment
export PYTHONPATH="${PROJECT_ROOT}/src"

# Default mesh and metadata paths
MESH="${1:-${DATA_ROOT:-./data}/generated_meshes/meshes/sample_000001.msh}"
META="${2:-${DATA_ROOT:-./data}/generated_meshes/metadata/sample_000001.json}"

echo "============================================================"
echo "FEM Residual Troubleshooting"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "Mesh: ${MESH}"
echo "Meta: ${META}"
echo "============================================================"
echo ""

# Check if mesh exists
if [ ! -f "${MESH}" ]; then
    echo "ERROR: Mesh file not found: ${MESH}"
    echo ""
    echo "To generate a test mesh, run:"
    echo "  python scripts/00_generate_meshes.py --n 1 --seed 0 --factor 40 --out_dir ./test_meshes/"
    exit 1
fi

# Run the test script
# Using mpirun -n 1 for single-process FEniCSx (required for point source evaluation)
echo "Running residual formulation tests..."
echo ""

if command -v mpirun &> /dev/null; then
    mpirun -n 1 python tests/fem_residual_debug/test_residual_formulations.py \
        --mesh "${MESH}" \
        --meta "${META}" \
        --electrode-radius 3.0 \
        --n-points 256
else
    python tests/fem_residual_debug/test_residual_formulations.py \
        --mesh "${MESH}" \
        --meta "${META}" \
        --electrode-radius 3.0 \
        --n-points 256
fi

echo ""
echo "============================================================"
echo "Tests complete."
echo "============================================================"
