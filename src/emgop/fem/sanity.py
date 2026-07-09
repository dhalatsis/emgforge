from __future__ import annotations

import json
import math
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import dolfinx
from dolfinx import fem
from petsc4py.PETSc import ScalarType as default_scalar_type
from ufl import dx

from .constants import GROUP_NAMES
from .solver import FEMModel
from .electrode_configs import ElectrodeFEMSolver, compute_ground_position

# Optional dependency (only needed for plotting)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# MPI globals used for diagnostics
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size


def load_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)


def estimate_r_skin_from_mesh(mesh: dolfinx.mesh.Mesh) -> float:
    xy = mesh.geometry.x[:, :2]
    r = np.sqrt((xy ** 2).sum(axis=1))
    return float(r.max())


def choose_source_point_near_surface(
    r_skin: float,
    length: float,
    thickness_skin: Optional[float] = None,
    angle_rad: float = 0.0,
    z_frac: float = 0.5,
    eps_mode: str = "skin_frac",
    eps_value: float = 0.15,
) -> np.ndarray:
    """
    Place a single source close to the outer surface at z = z_frac*L.
    eps_mode:
      - "skin_frac": eps = eps_value * thickness_skin
      - "abs":      eps = eps_value (in mesh units)
      - "r_frac":   eps = eps_value * r_skin
    """
    if eps_mode == "skin_frac":
        if thickness_skin is None:
            raise ValueError("eps_mode=skin_frac requires thickness_skin")
        eps = max(eps_value * thickness_skin, 0.2)
    elif eps_mode == "abs":
        eps = float(eps_value)
    elif eps_mode == "r_frac":
        eps = float(eps_value) * r_skin
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")

    r_src = r_skin - eps
    z_src = z_frac * length
    x_src = r_src * math.cos(angle_rad)
    y_src = r_src * math.sin(angle_rad)
    return np.array([[x_src, y_src, z_src]], dtype=np.float64)


def source_point_from_cyl_normalized(
    meta: dict,
    r_norm: float,
    theta_deg: float,
    z_norm: float,
    layer: str = "skin",
    margin: float = 0.0,
) -> np.ndarray:
    """
    Map normalized cylindrical coords to world coords, clamped to a target layer band.
    layer: "skin" (full radius) or "muscle" (between cortical and muscle radii).
    margin: optional inward/outward margin (mesh units) to stay away from interfaces.
    """
    gp = meta["geometry_params"]
    r_cort = float(gp["radius_cort_bone"])
    r_musc = float(gp["radius_muscle"])
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])
    shape = gp.get("shape", "circle")

    theta = np.deg2rad(theta_deg)
    z = z_norm * length

    # For elliptical geometry, compute angle-dependent radii
    if shape == "ellipse":
        from emgop.pointcloud.geometry import _ellipse_radius, _taper_scale
        a_skin = float(gp["a_skin"])
        b_skin = float(gp["b_skin"])
        r_skin_local = _ellipse_radius(a_skin, b_skin, theta)

        a_musc = float(gp.get("a_muscle", r_musc))
        b_musc = float(gp.get("b_muscle", r_musc))
        r_musc_local = _ellipse_radius(a_musc, b_musc, theta)

        a_cort = float(gp.get("a_cort_bone", r_cort))
        b_cort = float(gp.get("b_cort_bone", r_cort))
        r_cort_local = _ellipse_radius(a_cort, b_cort, theta)
    else:
        r_skin_local = r_skin
        r_musc_local = r_musc
        r_cort_local = r_cort

    # Apply taper if present
    z_profile = gp.get("z_profile", "constant")
    if z_profile == "taper":
        from emgop.pointcloud.geometry import _taper_scale
        scale = _taper_scale(gp, z, length)
        r_skin_local *= scale
        r_musc_local *= scale
        r_cort_local *= scale

    # base target from normalized radius
    r_target = r_norm * r_skin_local

    if layer == "muscle":
        r_lo = r_cort_local + margin
        r_hi = r_musc_local - margin
    else:  # "skin" or default
        r_lo = 0.0 + margin
        r_hi = r_skin_local - margin

    if r_hi <= r_lo:
        r_hi = r_lo + 1e-6  # avoid degenerate band

    # clamp into band
    r_target = max(r_lo, min(r_target, r_hi))

    x = r_target * np.cos(theta)
    y = r_target * np.sin(theta)
    return np.array([[x, y, z]], dtype=np.float64)


def make_probe_points_from_meta(meta: dict, z: float, angles=(0.0, math.pi / 2, math.pi)) -> np.ndarray:
    gp = meta["geometry_params"]
    r_canc = float(gp["radius_canc_bone"])
    r_cort = float(gp["radius_cort_bone"])
    r_musc = float(gp["radius_muscle"])
    r_fat = float(gp["radius_fat"])
    r_skin = float(gp["radius_skin"])

    radii = [
        0.0,
        0.5 * r_canc,
        0.5 * (r_canc + r_cort),
        0.5 * (r_cort + r_musc),
        0.5 * (r_musc + r_fat),
        0.5 * (r_fat + r_skin),
        0.95 * r_skin,
    ]

    pts = []
    for r in radii:
        for th in angles:
            pts.append([r * math.cos(th), r * math.sin(th), z])
    return np.array(pts, dtype=np.float64)


def compute_mean_u(model: FEMModel, uh: fem.Function) -> float:
    u1 = fem.Constant(model.mesh, default_scalar_type(1.0))
    vol = fem.assemble_scalar(fem.form(u1 * dx))
    mean_u = fem.assemble_scalar(fem.form(uh * u1 * dx)) / vol
    return float(mean_u)


def _save_slice_plot(data2d: np.ndarray, extent: Tuple[float, float, float, float], title: str, out_path: Path):
    if plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(data2d, origin="lower", extent=extent, aspect="auto")
    plt.title(title)
    plt.colorbar(im, shrink=0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_slices(
    model: FEMModel,
    uh: fem.Function,
    out_dir: Path,
    r_skin: float,
    length: float,
    source_point: np.ndarray,
    res_xy: int = 220,
    res_xz: int = 240,
):
    """
    Saves slice plots (XY, shifted XY, XZ, YZ, YZ near source) for sanity checks.
    """
    if plt is None:
        return

    if SIZE != 1 and RANK == 0:
        print("[warn] Plotting with MPI>1 may give incomplete slices. Prefer mpirun -n 1 for plotting.")

    x_src = float(source_point[0, 0])
    z_src = float(source_point[0, 2])

    xs = np.linspace(-r_skin, r_skin, res_xy)
    ys = np.linspace(-r_skin, r_skin, res_xy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, z_src)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vals = model.evaluate_solution_at_points(pts, uh=uh).reshape(X.shape)

    mask = (np.sqrt(X**2 + Y**2) <= r_skin)
    vals_masked = vals.copy()
    vals_masked[~mask] = np.nan
    _save_slice_plot(vals_masked, (xs.min(), xs.max(), ys.min(), ys.max()), f"XY at z={z_src:.3f}", out_dir / "plots" / "slice_xy.png")

    z_far = max(0.05 * length, min(0.95 * length, z_src + 0.10 * length))
    Zf = np.full_like(X, z_far)
    ptsf = np.stack([X.ravel(), Y.ravel(), Zf.ravel()], axis=1)
    valsf = model.evaluate_solution_at_points(ptsf, uh=uh).reshape(X.shape)
    valsf[~mask] = np.nan
    _save_slice_plot(valsf, (xs.min(), xs.max(), ys.min(), ys.max()), f"XY at z={z_far:.3f}", out_dir / "plots" / "slice_xy_far.png")

    zs = np.linspace(0.0, length, res_xz)
    X2, Z2 = np.meshgrid(xs, zs, indexing="xy")
    Y2 = np.zeros_like(X2)
    pts2 = np.stack([X2.ravel(), Y2.ravel(), Z2.ravel()], axis=1)
    vals2 = model.evaluate_solution_at_points(pts2, uh=uh).reshape(X2.shape)
    mask2 = (np.abs(X2) <= r_skin)
    vals2[~mask2] = np.nan
    _save_slice_plot(vals2, (xs.min(), xs.max(), zs.min(), zs.max()), "XZ at y=0", out_dir / "plots" / "slice_xz.png")

    ys2 = ys
    Y3, Z3 = np.meshgrid(ys2, zs, indexing="xy")
    X3 = np.zeros_like(Y3)
    pts3 = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)
    vals3 = model.evaluate_solution_at_points(pts3, uh=uh).reshape(Y3.shape)
    mask3 = (np.abs(Y3) <= r_skin)
    vals3[~mask3] = np.nan
    _save_slice_plot(vals3, (ys2.min(), ys2.max(), zs.min(), zs.max()), "YZ at x=0", out_dir / "plots" / "slice_yz.png")

    x_near = max(-0.95 * r_skin, min(0.95 * r_skin, 0.8 * x_src))
    X4 = np.full_like(Y3, x_near)
    pts4 = np.stack([X4.ravel(), Y3.ravel(), Z3.ravel()], axis=1)
    vals4 = model.evaluate_solution_at_points(pts4, uh=uh).reshape(Y3.shape)
    vals4[~mask3] = np.nan
    _save_slice_plot(vals4, (ys2.min(), ys2.max(), zs.min(), zs.max()), f"YZ at x={x_near:.3f}", out_dir / "plots" / "slice_yz_near.png")


def run_one(
    mesh_path: Path,
    meta_path: Optional[Path],
    out_dir: Path,
    plots: bool,
    angle: float,
    eps_mode: str,
    eps_value: float,
    source_mode: str = "gaussian",
    source_sigma: float = 0.1,
    source_cyl: Optional[Tuple[float, float, float]] = None,
    source_layer: str = "skin",
    source_margin: float = 0.0,
    # Electrode configuration parameters (new in v2.0.0)
    return_mode: str = "volumetric",
    ground_mode: str = "opposite",
    ground_cyl: Optional[Tuple[float, float, float]] = None,
    ground_radius: float = 5.0,
    ground_points: int = 256,
    electrode_inset: float = 4.0,
    use_native_point_source: bool = False,
    # Pennation angle for muscle fiber orientation
    pennation_angle: float = 0.0,
):
    # Import version info for summary
    from . import __version__ as fem_version

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(meta_path) if meta_path is not None and meta_path.exists() else None

    # Extract geometry params for source placement (needed before model creation for localized mode)
    if meta is not None:
        gp = meta["geometry_params"]
        r_skin = float(gp["radius_skin"])
        L = float(gp["length"])
        t_skin = float(gp.get("thickness_skin", 0.0))
    else:
        # We need a temporary model to get mesh dimensions
        temp_model = FEMModel(str(mesh_path), gdim=3, point_source=False, build_conductivity_map=False)
        r_skin = estimate_r_skin_from_mesh(temp_model.mesh)
        L = float(temp_model.height)
        t_skin = None
        del temp_model

    # Compute source point position
    if source_cyl is not None and meta is not None:
        r_norm, theta_deg, z_norm = source_cyl
        src = source_point_from_cyl_normalized(
            meta,
            r_norm=r_norm,
            theta_deg=theta_deg,
            z_norm=z_norm,
            layer=source_layer,
            margin=source_margin,
        )
    else:
        src = choose_source_point_near_surface(
            r_skin=r_skin,
            length=L,
            thickness_skin=t_skin,
            angle_rad=angle,
            z_frac=0.5,
            eps_mode=eps_mode,
            eps_value=eps_value,
        )

    # Ground position for localized mode (computed before solve)
    ground_position = None
    if return_mode == "localized":
        if meta is None:
            raise ValueError("Localized return mode requires metadata (--meta)")
        ground_position = compute_ground_position(
            meta,
            src[0],  # source_point as 1D array
            ground_mode=ground_mode,
            ground_cyl=ground_cyl,
            inset=electrode_inset,
        )

    # Create model and solve based on return mode
    if return_mode == "localized":
        # Use ElectrodeFEMSolver for bipolar configuration
        solver = ElectrodeFEMSolver(
            str(mesh_path),
            return_mode="localized",
            ground_position=ground_position,
            ground_radius=ground_radius,
            ground_points=ground_points,
            use_native_point_source=use_native_point_source,
            gdim=3,
            build_conductivity_map=True,
        )
        model = solver.model  # For compatibility with downstream code
        # Apply pennation angle if specified
        if pennation_angle != 0.0:
            model.apply_pinnation(pennation_angle)
        # Default source electrode radius based on electrode_inset
        source_radius = electrode_inset
        uh = solver.solve(src[0], source_radius=source_radius, source_points=ground_points)
    else:
        # Volumetric mode (existing behavior)
        model = FEMModel(
            str(mesh_path),
            gdim=3,
            point_source=(source_mode == "point"),
            point_electrode=False,
            build_conductivity_map=True,
            source_sigma=source_sigma,
        )
        # Apply pennation angle if specified
        if pennation_angle != 0.0:
            model.apply_pinnation(pennation_angle)
        uh = model.solve_for_point(src)
        solver = None

    zmid = 0.5 * L
    if meta is not None:
        probe_pts = make_probe_points_from_meta(meta, z=zmid)
    else:
        radii = [0.0, 0.25 * r_skin, 0.5 * r_skin, 0.75 * r_skin, 0.95 * r_skin]
        angles = (0.0, math.pi / 2, math.pi)
        probe_pts = np.array([[r * math.cos(a), r * math.sin(a), zmid] for r in radii for a in angles], dtype=np.float64)

    probe_vals = model.evaluate_solution_at_points(probe_pts, uh=uh)

    u1 = fem.Constant(model.mesh, default_scalar_type(1.0))
    vol = fem.assemble_scalar(fem.form(u1 * dx))
    mean_u = fem.assemble_scalar(fem.form(uh * u1 * dx)) / vol

    finite = bool(np.all(np.isfinite(probe_vals)))
    vmin = float(np.min(probe_vals)) if probe_vals.size else float("nan")
    vmax = float(np.max(probe_vals)) if probe_vals.size else float("nan")
    vabs = float(np.max(np.abs(probe_vals))) if probe_vals.size else float("nan")

    # fem_source path is only valid for volumetric mode (localized uses point sources)
    has_fem_source = return_mode == "volumetric" and hasattr(model, "source_function") and model.source_function is not None

    summary = {
        "mesh": str(mesh_path),
        "meta": str(meta_path) if meta_path is not None else None,
        "mpi_size": SIZE,
        "source_point": src[0].tolist(),
        "source_mode": source_mode,
        "source_sigma": source_sigma if source_mode == "gaussian" else None,
        "fem_source": str(out_dir / "source.npy") if has_fem_source else None,
        "source_cyl": source_cyl if source_cyl is not None else None,
        "source_layer": source_layer,
        "source_margin": source_margin,
        "r_skin": r_skin,
        "length": L,
        "mean_u": mean_u,
        "probe": {"finite": finite, "vmin": vmin, "vmax": vmax, "vabs_max": vabs, "n": int(probe_vals.size)},
        # Electrode configuration (v2.0.0+)
        "return_mode": return_mode,
        "ground_mode": ground_mode if return_mode == "localized" else None,
        "ground_position": ground_position.tolist() if ground_position is not None else None,
        "ground_cyl": ground_cyl if ground_cyl is not None else None,
        "ground_radius": ground_radius if return_mode == "localized" else None,
        "ground_points": ground_points if return_mode == "localized" else None,
        "electrode_inset": electrode_inset if return_mode == "localized" else None,
        "use_native_point_source": use_native_point_source,
        "pennation_angle": pennation_angle,
        "fem_version": fem_version,
    }

    if RANK == 0:
        np.save(out_dir / "u.npy", np.asarray(uh.vector.array))
        if hasattr(model, "sigma_anisotropic"):
            np.save(out_dir / "sigma.npy", np.asarray(model.sigma_anisotropic.vector.array))
        np.save(out_dir / "probes_points.npy", probe_pts)
        np.save(out_dir / "probes_values.npy", probe_vals)
        # Save source function (only available for volumetric mode)
        if has_fem_source:
            np.save(out_dir / "source.npy", np.asarray(model.source_function.vector.array))

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if plots:
            plot_slices(model, uh, out_dir, r_skin=r_skin, length=L, source_point=src)

        print("\n" + "=" * 80)
        print("FEM sanity run complete")
        print(f"mesh: {mesh_path}")
        if meta_path is not None:
            print(f"meta: {meta_path}")
        print(f"out:  {out_dir}")
        print(f"source: {summary['source_point']}")
        print(f"mean(u): {mean_u:.6e}")
        print(f"probes: finite={finite} vmin={vmin:.6e} vmax={vmax:.6e} |v|max={vabs:.6e}")
        if plots and plt is None:
            print("[warn] matplotlib not available; plots were skipped.")
        elif plots:
            print(f"plots: {out_dir / 'plots'}")


def run_manifest_entry(
    manifest_path: Path,
    index: int,
    out_root: Path,
    plots: bool,
    angle: float,
    eps_mode: str,
    eps_value: float,
    source_mode: str = "gaussian",
    source_sigma: float = 0.1,
    source_cyl: Optional[Tuple[float, float, float]] = None,
    source_layer: str = "skin",
    source_margin: float = 0.0,
    # Electrode configuration parameters (new in v2.0.0)
    return_mode: str = "volumetric",
    ground_mode: str = "opposite",
    ground_cyl: Optional[Tuple[float, float, float]] = None,
    ground_radius: float = 5.0,
    ground_points: int = 256,
    electrode_inset: float = 4.0,
    use_native_point_source: bool = False,
    # Pennation angle for muscle fiber orientation
    pennation_angle: float = 0.0,
):
    rows = load_manifest(manifest_path)
    if index < 0 or index >= len(rows):
        raise SystemExit(f"--index out of range: {index} (manifest has {len(rows)} rows)")

    row = rows[index]
    mesh_path = Path(row.get("mesh_file", row.get("mesh", "")))
    meta_path = Path(row.get("meta_file", row.get("meta", ""))) if row.get("meta_file") or row.get("meta") else None

    sample_id = row.get("sample_id", f"index_{index:06d}")
    out_dir = out_root / sample_id

    run_one(
        mesh_path,
        meta_path,
        out_dir,
        plots,
        angle,
        eps_mode,
        eps_value,
        source_mode=source_mode,
        source_sigma=source_sigma,
        source_cyl=source_cyl,
        source_layer=source_layer,
        source_margin=source_margin,
        return_mode=return_mode,
        ground_mode=ground_mode,
        ground_cyl=ground_cyl,
        ground_radius=ground_radius,
        ground_points=ground_points,
        electrode_inset=electrode_inset,
        use_native_point_source=use_native_point_source,
        pennation_angle=pennation_angle,
    )


__all__ = [
    "load_manifest",
    "load_meta",
    "estimate_r_skin_from_mesh",
    "choose_source_point_near_surface",
    "make_probe_points_from_meta",
    "compute_mean_u",
    "plot_slices",
    "run_one",
    "run_manifest_entry",
]

