cd "$(dirname "$0")"
export PYTHONPATH=src
max_jobs=4
n=0

# Electrode configuration (new in FEM v2.0.0)
# Set these to override defaults:
#   RETURN_MODE=localized        # volumetric (default) or localized (bipolar)
#   GROUND_MODE=opposite         # opposite, distal, or cylindrical
#   GROUND_CYL='1.0,180,0.5'     # r_norm,theta_deg,z_norm (for cylindrical mode)
#   NATIVE_POINT_SOURCE=0        # Use scifem point source (default is native, no scifem)

for i in $(seq 100 500); do
  SAMPLE=$(printf "sample_%06d" "$i")
  ./pipeline.sh \
    --steps fem,voxel \
    --sample "$SAMPLE" \
    --out-root "${DATA_ROOT:-./data}/forward_model_datasets/v1_medium" \
    --nx 48 --ny 48 --nz 96 \
    --sigma-vox 1.5 \
    --source-mode gaussian \
    --source-sigma 3 \
    --source-cyl '1.,0.,0.5' \
    --source-layer skin \
    --source-margin 0.0 \
    --use-fem-source 0 \
    --return-mode "${RETURN_MODE:-volumetric}" \
    --ground-mode "${GROUND_MODE:-opposite}" \
    ${GROUND_CYL:+--ground-cyl "$GROUND_CYL"} \
    --ground-radius "${GROUND_RADIUS:-5.0}" \
    --ground-points "${GROUND_POINTS:-256}" \
    --electrode-inset "${ELECTRODE_INSET:-4.0}" \
    --native-point-source "${NATIVE_POINT_SOURCE:-1}" &
  (( ++n % max_jobs == 0 )) && wait
done
wait