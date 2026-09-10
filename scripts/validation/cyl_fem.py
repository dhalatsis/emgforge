"""Cylinder FEM lead fields for the validation suite — solve once, cache, reuse.

Every check that needs an FEM φ(z) on the 4-layer cylinder (analytical radii
10/35/38/40, Farina 2004 conductivities) reads it from one cache built here.
The cylinder is z-invariant and rotationally symmetric, so ONE reciprocal solve
(electrode at θ=0, z=L/2) serves every fibre depth, angle and axial offset:
we sample full-length fibre lines and let each consumer window them.

Cache layout (``_results/validation/fem_cache/cyl_lines.npz``):

  z_abs           (Nz,)            absolute z of the line samples (mm)
  radii, thetas   (Nr,), (Nth,)    fibre radial positions (mm) / angles (deg)
  phi             (Nr, Nth, Nz)    φ along each line, electrode at (θ=0, z=ZE)
  phi_e_th30      (Nr, Nth, Nz)    same lines, electrode rotated to θ=+30°
  phi_e_z140      (Nr, Nth, Nz)    same lines, electrode shifted to z=ZE+20
  recip_*         reciprocity pair (see build_cache)
  fat/*           fat-thickness sweep on the anat meshes (fat 2/4/6/8 mm)

Run:  python scripts/validation/cyl_fem.py        (≈10–15 min, 8 solves)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESH_CACHE = ROOT / "_results/sanity/fem_cache"          # meshes shared with the sanity suite
CACHE = ROOT / "_results/validation/fem_cache/cyl_lines.npz"

# --- geometry (= emgforge.analytical / tests.regression.analytical_ref) ---
R_BONE, R_MUSCLE, R_FAT, R_SKIN, LENGTH = 10.0, 35.0, 38.0, 40.0, 240.0
ZE = LENGTH / 2.0                    # primary electrode axial position
DZ_LINE = 0.5                        # line sampling step (mm); consumers re-grid
RADII = np.array([33.0, 30.0, 27.0, 25.0, 20.0, 15.0])      # fibre radial pos (mm)
THETAS = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 180.0])
FAT_MM = [2, 4, 6, 8]                # anat meshes: muscle 35, fat 35+f, skin 35+f+2
SOURCE_SIGMA = 5.0                   # Gaussian electrode source (validated recipe)


def _geo(r_fat=R_FAT, r_skin=R_SKIN):
    from emgforge.fem.geometry import ParametricGeometry
    return ParametricGeometry(r_bone=R_BONE, r_muscle=R_MUSCLE, r_fat=r_fat,
                              r_skin=r_skin, length=LENGTH)


def _model(msh: Path, json: Path, geo, source_sigma=SOURCE_SIGMA):
    from emgforge.fem import FEMModel
    from emgforge.fem.conductivity import TissueTable
    geo.build(msh, json, char_length=0.3)
    return FEMModel(str(msh), gdim=3, build_conductivity_map=True, point_source=False,
                    source_sigma=source_sigma, conductivity=TissueTable.analytical())


SIGMA1 = CACHE.parent / "cyl_lines_sigma1.npz"


def sigma1_lines(force=False):
    """The primary solve repeated with a σ=1 mm electrode source (the default σ=5 mm blob
    reaches through the 2 mm skin + 3 mm fat into the muscle). Same lines as ``phi``."""
    if SIGMA1.exists() and not force:
        return np.load(SIGMA1)["phi"]
    geo = _geo()
    model = _model(MESH_CACHE / "cyl_10_35_38_40.msh", MESH_CACHE / "cyl_10_35_38_40.json", geo, 1.0)
    phi = sample_lines(model, model.solve_for_point(geo.electrode_on_skin(0.0, ZE)), geo, RADII, THETAS)
    np.savez_compressed(SIGMA1, phi=phi)
    return phi


def z_line():
    return np.arange(0.5, LENGTH - 0.5 + 1e-9, DZ_LINE)


def sample_lines(model, uh, geo, radii, thetas):
    """φ on every (radius, θ) fibre line, full length. Returns (Nr, Nth, Nz)."""
    z = z_line()
    out = np.zeros((len(radii), len(thetas), len(z)))
    for i, r in enumerate(radii):
        for j, th in enumerate(thetas):
            x0, y0 = geo.fibre_xy_radial(r, th)
            pts = np.column_stack([np.full(len(z), x0), np.full(len(z), y0), z])
            out[i, j] = model.evaluate_solution_at_points(pts, uh=uh)
    return out


def build_cache(force=False):
    if CACHE.exists() and not force:
        return dict(np.load(CACHE))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    store = dict(z_abs=z_line(), radii=RADII, thetas=THETAS, ze=ZE)

    geo = _geo()
    t0 = time.time()
    model = _model(MESH_CACHE / "cyl_10_35_38_40.msh", MESH_CACHE / "cyl_10_35_38_40.json", geo)
    print(f"[model] {time.time()-t0:.0f}s", flush=True)

    def solve(theta, z):
        t0 = time.time()
        uh = model.solve_for_point(geo.electrode_on_skin(theta, z))
        print(f"[solve θ={theta:g} z={z:g}] {time.time()-t0:.0f}s", flush=True)
        return uh

    uh0 = solve(0.0, ZE)
    store["phi"] = sample_lines(model, uh0, geo, RADII, THETAS)
    # symmetry partners: rotate the electrode / shift it axially
    store["phi_e_th30"] = sample_lines(model, solve(30.0, ZE), geo, RADII, THETAS)
    store["phi_e_z140"] = sample_lines(model, solve(0.0, ZE + 20.0), geo, RADII, THETAS)
    # reciprocity: source at a fibre point, read at the electrode; vs uh0 at that point
    fib_pt = np.array([*geo.fibre_xy_radial(30.0, 0.0), ZE])
    el_pt = geo.electrode_on_skin(0.0, ZE)
    uh_f = model.solve_for_point(fib_pt)
    store["recip_fibre_to_elec"] = model.evaluate_solution_at_points(el_pt[None], uh=uh_f)[0]
    store["recip_elec_to_fibre"] = model.evaluate_solution_at_points(fib_pt[None], uh=uh0)[0]
    # a short transverse profile of the electrode's field at the fibre depth (for the
    # anisotropy / lateral-extent checks): φ across x at r=30, z=ZE plane, y-offsets
    yoff = np.arange(-30, 30.01, 1.0)
    x0, _ = geo.fibre_xy_radial(30.0, 0.0)
    pts = np.column_stack([np.full(len(yoff), x0), yoff, np.full(len(yoff), ZE)])
    store["trans_y"] = yoff
    store["trans_phi"] = model.evaluate_solution_at_points(pts, uh=uh0)
    del model, uh0, uh_f

    # fat-thickness sweep (anat meshes): fibre fixed at r=30 (5 mm under the muscle
    # surface) and, separately, fixed 10 mm below the skin
    for f in FAT_MM:
        rf, rs = R_MUSCLE + f, R_MUSCLE + f + 2.0
        g = _geo(rf, rs)
        t0 = time.time()
        m = _model(MESH_CACHE / f"anat/cyl_fat{int(rf)}_cl0.3.msh",
                   MESH_CACHE / f"anat/cyl_fat{int(rf)}_cl0.3.json", g)
        uh = m.solve_for_point(g.electrode_on_skin(0.0, ZE))
        print(f"[fat {f} mm model+solve] {time.time()-t0:.0f}s", flush=True)
        lines = sample_lines(m, uh, g, np.array([30.0, rs - 10.0]), np.array([0.0, 10.0, 20.0]))
        store[f"fat{f}_phi"] = lines           # (2 radii, 3 thetas, Nz)
        store[f"fat{f}_r_skin"] = rs
        del m, uh

    np.savez_compressed(CACHE, **store)
    print("wrote", CACHE, flush=True)
    return dict(np.load(CACHE))


RECIP = CACHE.parent / "reciprocity_interior.npz"


def reciprocity_interior(force=False):
    """Reciprocity between two INTERIOR points (Gaussian sources fully inside the
    mesh, unlike a skin electrode whose blob is clipped): φ_A(B) must equal φ_B(A).
    A = (r=30, θ=0), B = (r=30, θ=30), both at z=ZE. Cached; 2 solves."""
    if RECIP.exists() and not force:
        d = np.load(RECIP)
        return float(d["a_at_b"]), float(d["b_at_a"])
    geo = _geo()
    model = _model(MESH_CACHE / "cyl_10_35_38_40.msh", MESH_CACHE / "cyl_10_35_38_40.json", geo)
    A = np.array([*geo.fibre_xy_radial(30.0, 0.0), ZE])
    B = np.array([*geo.fibre_xy_radial(30.0, 30.0), ZE])
    a_at_b = model.evaluate_solution_at_points(B[None], uh=model.solve_for_point(A))[0]
    b_at_a = model.evaluate_solution_at_points(A[None], uh=model.solve_for_point(B))[0]
    np.savez(RECIP, a_at_b=a_at_b, b_at_a=b_at_a)
    return float(a_at_b), float(b_at_a)


# ---------------------------------------------------------------------------
# consumer helpers
# ---------------------------------------------------------------------------
def window(phi_line, z_abs, centre_mm, nz, dz):
    """Re-grid a full-length line onto ``nz`` samples at ``dz`` centred on ``centre_mm``
    (index nz//2 → centre), the convention ``compute_sfap_spatial`` expects."""
    z = (np.arange(nz) - nz // 2) * dz + centre_mm
    return np.interp(z, z_abs, phi_line, left=phi_line[0], right=phi_line[-1])


if __name__ == "__main__":
    build_cache(force="--force" in sys.argv)
