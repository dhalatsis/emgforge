from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

from emgop.fem.constants import CONDUCTIVITY
from emgop.fem.solver import FEMModel
from dolfinx import fem, geometry, io
from mpi4py import MPI


def _grid_centers_normalized(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns normalized grid center coordinates (x', y', z') with:
      x', y' in [-1, 1], z' in [0, 1]
    Shapes are (nx,), (ny,), (nz,).
    """
    xs = np.linspace(-1.0 + 1.0 / nx, 1.0 - 1.0 / nx, nx)
    ys = np.linspace(-1.0 + 1.0 / ny, 1.0 - 1.0 / ny, ny)
    zs = np.linspace(0.0 + 0.5 / nz, 1.0 - 0.5 / nz, nz)
    return xs, ys, zs


def _conductivity_tensor_channels(mask: np.ndarray, tissue_mask: np.ndarray, anis_ratio: float) -> np.ndarray:
    """
    Build 6-channel symmetric conductivity tensor (xx, yy, zz, xy, xz, yz)
    for each voxel. mask is boolean inside; tissue_mask is int labels 0..4.
    """
    sigma = np.zeros(mask.shape + (6,), dtype=np.float32)

    # channel order: xx, yy, zz, xy, xz, yz
    def set_iso(tissue_id: int, value: float):
        m = mask & (tissue_mask == tissue_id)
        sigma[m, 0] = value
        sigma[m, 1] = value
        sigma[m, 2] = value
        # off-diagonals stay zero

    set_iso(0, CONDUCTIVITY["Cancellous Bone"])
    set_iso(1, CONDUCTIVITY["Cortical Bone"])
    set_iso(3, CONDUCTIVITY["Fat"])
    set_iso(4, CONDUCTIVITY["Skin"])

    # Muscle anisotropic diag
    m_musc = mask & (tissue_mask == 2)
    if np.any(m_musc):
        musc = CONDUCTIVITY["Muscle"]
        sigma[m_musc, 0] = musc[0, 0]
        sigma[m_musc, 1] = musc[1, 1]
        sigma[m_musc, 2] = musc[2, 2]

    return sigma


def _evaluate_u_on_grid(mesh_path: Path, u_vec: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Evaluate a CG1 solution vector on voxel centers.
    X, Y, Z are broadcastable (nx, ny, nz) arrays in world coords.
    Returns u_grid with shape (nx, ny, nz).
    """
    mesh, _, _ = io.gmshio.read_from_msh(str(mesh_path), MPI.COMM_WORLD, gdim=3)
    V = fem.functionspace(mesh, ("CG", 1))
    u = fem.Function(V)
    u.vector.array[:] = u_vec

    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    midpoints = geometry.create_midpoint_tree(mesh, mesh.topology.dim, np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))

    Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
    pts = np.stack([Xb.ravel(), Yb.ravel(), Zb.ravel()], axis=1)
    cell_ids = geometry.compute_closest_entity(tree, midpoints, mesh, pts).squeeze()
    vals = u.eval(pts, cell_ids)
    u_grid = np.asarray(vals).reshape(Xb.shape)
    return u_grid


def voxelize_geometry_and_source(
    meta: Dict,
    source_point: np.ndarray,
    grid_shape: Tuple[int, int, int] = (96, 96, 192),
    gaussian_sigma_vox: float = 1.5,
    fem_u_path: Optional[Path] = None,
    mesh_override: Optional[Path] = None,
    fem_source_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Voxelize geometry (conductivity tensor) and source (Gaussian blob with mean-zero)
    onto a canonical normalized grid that always contains the cylinder.

    Returns arrays with axes ordered (z, y, x, channel) for tensors and (z, y, x) for scalars.
    Channels for conductivity: [xx, yy, zz, xy, xz, yz]
    """
    nx, ny, nz = grid_shape
    xs, ys, zs = _grid_centers_normalized(nx, ny, nz)

    gp = meta["geometry_params"]
    shape = gp.get("shape", "circle")
    dx_bone = float(gp.get("bone_offset_x", 0.0))
    dy_bone = float(gp.get("bone_offset_y", 0.0))
    bone_count = int(gp.get("bone_count", 1))

    z_profile = gp.get("z_profile", "constant")
    s0 = float(gp.get("taper_scale_z0", 1.0))
    s1 = float(gp.get("taper_scale_z1", 1.0))
    if z_profile == "constant":
        s0, s1 = 1.0, 1.0

    r_canc = float(gp["radius_canc_bone"])
    r_cort = float(gp["radius_cort_bone"])
    r_musc = float(gp["radius_muscle"])
    r_fat = float(gp["radius_fat"])
    r_skin = float(gp["radius_skin"])
    length = float(gp["length"])

    # z-dependent scale on voxel centers (shape 1,1,nz)
    z_scale = (s0 + (s1 - s0) * zs)[None, None, :]

    # World coordinates for grid centers (shape: (nx, ny, nz) after broadcasting)
    if shape == "ellipse":
        a_skin = float(gp["a_skin"])
        b_skin = float(gp["b_skin"])
        a_max = max(a_skin * s0, a_skin * s1)
        b_max = max(b_skin * s0, b_skin * s1)
        X = xs[:, None, None] * a_max  # (nx,1,1)
        Y = ys[None, :, None] * b_max  # (1,ny,1)
    else:
        r_max = max(r_skin * s0, r_skin * s1)
        X = xs[:, None, None] * r_max  # (nx, 1, 1)
        Y = ys[None, :, None] * r_max  # (1, ny, 1)
    Z = zs[None, None, :] * length  # (1, 1, nz)

    # Broadcast radial distance to full 3D grid
    # Broadcast to full 3D grids for classification
    Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)

    # Tissue labels: 0=canc,1=cort,2=muscle,3=fat,4=skin, -1 outside
    tissue = np.full((nx, ny, nz), -1, dtype=np.int8)

    if shape == "ellipse":
        a_musc = float(gp["a_muscle"])
        b_musc = float(gp["b_muscle"])
        a_fat = float(gp["a_fat"])
        b_fat = float(gp["b_fat"])
        a_skin = float(gp["a_skin"])
        b_skin = float(gp["b_skin"])

        a_canc = float(gp["a_canc_bone"])
        b_canc = float(gp["b_canc_bone"])
        a_cort = float(gp["a_cort_bone"])
        b_cort = float(gp["b_cort_bone"])

        # outer layers centered at origin
        e_skin = (Xb / (a_skin * z_scale)) ** 2 + (Yb / (b_skin * z_scale)) ** 2
        e_fat = (Xb / (a_fat * z_scale)) ** 2 + (Yb / (b_fat * z_scale)) ** 2
        e_musc = (Xb / (a_musc * z_scale)) ** 2 + (Yb / (b_musc * z_scale)) ** 2

        tissue[e_skin <= 1.0] = 4
        tissue[e_fat <= 1.0] = 3
        tissue[e_musc <= 1.0] = 2

        # bones: single-bone uses (dx_bone,dy_bone). two-bone uses explicit centers and sizes.
        if bone_count == 2 and "bone1_center_x" in gp and "bone2_center_x" in gp:
            x1 = float(gp["bone1_center_x"])
            y1 = float(gp["bone1_center_y"])
            x2 = float(gp["bone2_center_x"])
            y2 = float(gp["bone2_center_y"])

            a_canc2 = float(gp["a_canc_bone_2"])
            b_canc2 = float(gp["b_canc_bone_2"])
            a_cort2 = float(gp["a_cort_bone_2"])
            b_cort2 = float(gp["b_cort_bone_2"])

            X1 = Xb - x1
            Y1 = Yb - y1
            X2 = Xb - x2
            Y2 = Yb - y2

            e_cort1 = (X1 / (a_cort * z_scale)) ** 2 + (Y1 / (b_cort * z_scale)) ** 2
            e_canc1 = (X1 / (a_canc * z_scale)) ** 2 + (Y1 / (b_canc * z_scale)) ** 2
            e_cort2 = (X2 / (a_cort2 * z_scale)) ** 2 + (Y2 / (b_cort2 * z_scale)) ** 2
            e_canc2 = (X2 / (a_canc2 * z_scale)) ** 2 + (Y2 / (b_canc2 * z_scale)) ** 2

            tissue[(e_cort1 <= 1.0) | (e_cort2 <= 1.0)] = 1
            tissue[(e_canc1 <= 1.0) | (e_canc2 <= 1.0)] = 0
        else:
            Xs = Xb - dx_bone
            Ys = Yb - dy_bone
            e_cort = (Xs / (a_cort * z_scale)) ** 2 + (Ys / (b_cort * z_scale)) ** 2
            e_canc = (Xs / (a_canc * z_scale)) ** 2 + (Ys / (b_canc * z_scale)) ** 2
            tissue[e_cort <= 1.0] = 1
            tissue[e_canc <= 1.0] = 0
    else:
        r0 = np.sqrt(Xb**2 + Yb**2)
        tissue[r0 <= (r_skin * z_scale)] = 4
        tissue[r0 <= (r_fat * z_scale)] = 3
        tissue[r0 <= (r_musc * z_scale)] = 2

        if bone_count == 2 and "bone1_center_x" in gp and "bone2_center_x" in gp:
            x1 = float(gp["bone1_center_x"])
            y1 = float(gp["bone1_center_y"])
            x2 = float(gp["bone2_center_x"])
            y2 = float(gp["bone2_center_y"])
            r_canc2 = float(gp["radius_canc_bone_2"])
            r_cort2 = float(gp["radius_cort_bone_2"])

            r1 = np.sqrt((Xb - x1) ** 2 + (Yb - y1) ** 2)
            r2 = np.sqrt((Xb - x2) ** 2 + (Yb - y2) ** 2)
            tissue[(r1 <= (r_cort * z_scale)) | (r2 <= (r_cort2 * z_scale))] = 1
            tissue[(r1 <= (r_canc * z_scale)) | (r2 <= (r_canc2 * z_scale))] = 0
        else:
            rb = np.sqrt((Xb - dx_bone) ** 2 + (Yb - dy_bone) ** 2)
            tissue[rb <= (r_cort * z_scale)] = 1
            tissue[rb <= (r_canc * z_scale)] = 0

    mask_inside = tissue >= 0

    # Conductivity tensor
    sigma = _conductivity_tensor_channels(mask_inside, tissue, anis_ratio=5.0)

    # Source: Gaussian normalized to integral +1 then subtract uniform sink to make integral 0
    sx, sy, sz = source_point
    dx = X - sx
    dy = Y - sy
    dz = Z - sz
    dist2 = dx * dx + dy * dy + dz * dz

    # Use grid spacing for sigma
    if shape == "ellipse":
        hx = 2 * (a_max) / nx
        hy = 2 * (b_max) / ny
    else:
        hx = 2 * (r_max) / nx
        hy = 2 * (r_max) / ny
    hz = length / nz
    dV = hx * hy * hz
    sigma_phys = gaussian_sigma_vox * (hx + hy + hz) / 3.0

    blob = np.exp(-dist2 / (2.0 * sigma_phys * sigma_phys)) * mask_inside
    blob_sum = float(blob.sum() * dV)
    if blob_sum > 0:
        blob /= blob_sum

    volume = float(mask_inside.sum() * dV)
    source = blob - (mask_inside.astype(np.float32) * (1.0 / volume))

    # Optional FEM solution sampling onto grid
    u_grid = None
    if fem_u_path is not None:
        mesh_path = mesh_override if mesh_override is not None else Path(meta["mesh_file"])
        u_vec = np.load(fem_u_path)
        u_grid = _evaluate_u_on_grid(mesh_path, u_vec, X, Y, Z)

    # Optional FEM source sampling (Gaussian only)
    source_from_fem = None
    if fem_source_path is not None:
        mesh_path = mesh_override if mesh_override is not None else Path(meta["mesh_file"])
        s_vec = np.load(fem_source_path)
        source_from_fem = _evaluate_u_on_grid(mesh_path, s_vec, X, Y, Z)

    # Reorder axes to (z, y, x, ...)
    sigma_zyx = np.transpose(sigma, (2, 1, 0, 3))
    source_zyx = np.transpose(source, (2, 1, 0))
    mask_zyx = np.transpose(mask_inside.astype(np.uint8), (2, 1, 0))

    return {
        "sigma": sigma_zyx,
        "source": source_zyx.astype(np.float32),
        "mask": mask_zyx,
        **({"u": np.transpose(u_grid, (2, 1, 0)).astype(np.float32)} if u_grid is not None else {}),
        **({"source_fem": np.transpose(source_from_fem, (2, 1, 0)).astype(np.float32)} if source_from_fem is not None else {}),
        "grid_shape": np.array(grid_shape, dtype=np.int32),
        "spacing": np.array([hx, hy, hz], dtype=np.float32),
        "r_skin": np.array([r_skin], dtype=np.float32),
        "length": np.array([length], dtype=np.float32),
        "source_point": np.array(source_point, dtype=np.float32),
    }


def load_meta(meta_path: Path) -> Dict:
    with open(meta_path, "r") as f:
        return json.load(f)


__all__ = ["voxelize_geometry_and_source", "load_meta"]

