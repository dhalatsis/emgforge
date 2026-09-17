#!/usr/bin/env python3
"""Generate a per-muscle fibre config for the fibre-aligned FEM conductivity.

The FEM muscle conductivity is transversely isotropic (σ∥ along the fibre, σ⊥
across, 5:1). By default the MRI solver aligns that tensor with the limb z-axis
everywhere; this script produces the per-muscle fibre directions that let the
solver align σ with each muscle's *own* fibres instead (``sigma_mode="centerline"``).

Recipe (the archive-settled one):
    MuscleFiberModel(seg) → estimate_fibers("pca") → estimate_centerlines() → save_config()

  * PCA of each muscle label's voxel cloud → principal axis = fibre direction.
  * Per-z-slice centroids → a smoothed centerline, so the σ tangent follows the
    muscle along its length (centerline mode).
  * Bone / fat / skin labels are classified out; only muscle labels get a direction.

This builds centerlines only (the production default). The cross-section boundary
smoothing (raw / R-smooth / mask-sm / dilated / dilated-cons) and the fusiform
morphing-disk are available in ``fiber_directions`` but not built here — see
``docs/mri_fibre_geometry.md`` (backlog TODO C-04).

Usage:
    python scripts/mri/build_fiber_config.py --seg forearm_seg.nii.gz \
        --out forearm_WR_fibers.json

Then point the solver at it:
    MRIFEMModel(mesh, fiber_config="forearm_WR_fibers.json", sigma_mode="centerline")

Requires the ``[mri]`` extra (nibabel).
"""
import argparse

from emgforge.mri.core.fiber_directions import MuscleFiberModel


def build(seg: str, out: str, min_slices: int = 3, smooth_sigma: float = 1.0,
          method: str = "pca") -> None:
    fm = MuscleFiberModel(seg)
    fm.estimate_fibers(method=method)
    fm.estimate_centerlines(min_slices=min_slices, smooth_sigma=smooth_sigma)
    fm.save_config(out)

    mus = [m for m in fm.muscles.values() if m.tissue_type == "muscle"]
    n_cl = sum(1 for m in mus if m.centerline is not None)
    n_bone = sum(1 for m in fm.muscles.values() if m.tissue_type == "bone")
    print(f"{len(mus)} muscles ({n_cl} with centerlines), {n_bone} bone labels → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a per-muscle fibre config (PCA + centerline).")
    ap.add_argument("--seg", required=True, help="labelled NIfTI segmentation (.nii.gz)")
    ap.add_argument("--out", required=True, help="output fibre config (.json)")
    ap.add_argument("--min-slices", type=int, default=3,
                    help="min z-slices for a centerline (else global direction only)")
    ap.add_argument("--smooth-sigma", type=float, default=1.0,
                    help="centerline waypoint smoothing in slice bins (settled: 1.0)")
    ap.add_argument("--method", default="pca",
                    choices=["pca", "endpoints", "harmonic"],
                    help="fibre-direction estimator (default: pca). 'harmonic' "
                         "uses the masked-Laplace streamline field.")
    a = ap.parse_args()
    build(a.seg, a.out, a.min_slices, a.smooth_sigma, a.method)


if __name__ == "__main__":
    main()
