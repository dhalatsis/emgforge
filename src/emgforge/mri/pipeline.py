"""One chain from a labelled MRI segmentation to interference EMG on an HD-EMG grid.

The stages, in the order ``scripts/run_pipeline.py`` runs (and times) them. Each is a
plain function over package objects; none adds physics — every stage calls the module
that owns it:

  1. :func:`build_mesh`             segmentation → tetrahedral mesh with tissue tags
                                    (``mri.core.build_mesh``)
  2. :func:`build_fibre_config`     PCA fibre direction + smoothed centreline per muscle
                                    → the fibre-aligned conductivity config
                                    (``mri.core.fiber_directions``)
  3. ``fm.estimate_cross_sections`` ray-cast muscle boundaries → the morphing-disk frame
  4. :func:`poisson_bed`            the muscle's fibre bed (``mri.core.muscle_fiber_bed``)
  5. :func:`henneman_pool`          the motor-unit pool (``mri.core.motor_unit_pool``)
  6. :func:`build_volume_conductor` σ tensor per cell (fibre-aligned), skin shell,
                                    assembly (``mri.core.fem_solver.MRIFEMModel``)
  7. :func:`grid_electrodes`        a regular M×N grid ray-cast onto the skin over the muscle
  8. :func:`solve_grid_leadfields`  one reciprocity solve per electrode, φ along every fibre
  9. :func:`condition_leadfields`   the recipe's 3-monopole fit of φ, once per (electrode, fibre)
 10. :func:`muap_tensor`            per-MU direct line-source synthesis on the grid
                                    (``synthesis.field_to_muap`` with :func:`production_config`)
 11. ``emgforge.Simulator``         pool + twitch + drive → EMG, force, spikes

What this module adds is the glue that used to live in scripts and figure code (grid
placement, per-unit fibre beds, the parallel tensor build) and two factorings that make a
cold run cheap without changing a number:

* the mesh-cell lookup of the fibre sample points is done once for all electrodes
  (``MRIFEMModel.locate_points`` → ``evaluate_solution_at_points(cells=…)``);
* the monopole fit of φ(z) — which depends on the electrode and the fibre but not on the
  motor unit — is done once per pair (:func:`condition_leadfields`) instead of once per
  unit that shares the fibre; the per-unit synthesis then runs the identical recipe with
  ``denoise="none"`` on the conditioned field. :func:`verify_conditioning` checks the two
  paths agree to the last bit.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
from emgforge.synthesis.preprocessing import denoise_field_n

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_SEG = DATA / "forearm_WR_segmentation.nii.gz"      # the committed WR forearm
LABELS_JSON = DATA / "pd_lab_labels.json"
FCU = 8                                                    # the label every released result used

# the FCU innervation zone used by every released pool (scripts/mri/sample_mu_pool.py)
IZ_FRAC_FCU = 0.305
IZ_JITTER = 0.02


def muscle_name(label: int) -> str:
    """Human name of a WR segmentation label (``pd_lab_labels.json``), or ``label<n>``."""
    try:
        with open(LABELS_JSON) as fh:
            return json.load(fh)["common_labels"][str(int(label))]["name"]
    except (OSError, KeyError, ValueError):
        return f"label{int(label)}"


# --------------------------------------------------------------------------- 1. mesh
def build_mesh(seg, out_msh, *, target_z: float = 1.5, edge_length: float = 0.03,
               surface_faces: int = 50000, smooth_iters: int = 10) -> dict:
    """Segmentation → tetrahedral mesh with tissue tags (the ``build_mesh`` recipe).

    Resample z to ``target_z`` mm, marching cubes on the tissue mask, Laplacian
    smoothing + decimation to ``surface_faces`` triangles, fTetWild tetrahedralisation at
    ``edge_length`` (fraction of the bounding-box diagonal), one tissue tag per
    tetrahedron from the label under its centroid. Written as MSH 2.2 with a physical
    group per tissue plus ``<stem>_metadata.json`` recording the recipe.

    ``edge_length=0.03`` gives ~47 k tetrahedra on the WR forearm — the resolution of the
    released ``forearm_WR.msh`` (42 k; its exact recipe was not recorded). The CLI default
    of ``build_mesh.py`` (0.02) gives ~100 k.
    """
    from emgforge.mri.core import build_mesh as bm

    out_msh = Path(out_msh)
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    seg_data, vs = bm.load_and_resample(str(seg), target_z)
    v, f = bm.extract_surface(seg_data, vs)
    n_mc = int(len(f))
    v, f = bm.smooth_and_decimate(v, f, target_faces=surface_faces, smooth_iters=smooth_iters)
    cwd = os.getcwd()
    os.chdir(out_msh.parent)          # fTetWild drops a __tracked_surface.stl into the cwd
    try:
        tv, tc = bm.tetrahedralize(v, f, edge_length)
    finally:
        os.chdir(cwd)
    side = out_msh.parent / "__tracked_surface.stl"
    if side.exists():
        side.unlink()
    seg_labels, tissue_ids = bm.assign_labels(tv, tc, seg_data, vs)
    # The ASCII writer is the one every validated mesh came from: the gmsh-API path adds
    # nodes to a discrete entity before creating it and raises whenever gmsh is importable.
    bm._export_msh_ascii(tv, tc, tissue_ids, seg_labels, str(out_msh))
    meta_path = out_msh.with_name(out_msh.stem + "_metadata.json")
    bm.save_metadata(str(meta_path), str(seg), seg_data, vs, tv, tc, tissue_ids, seg_labels)
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["build_params"] = dict(target_z=float(target_z), edge_length=float(edge_length),
                                surface_faces=int(surface_faces), smooth_iters=int(smooth_iters))
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return dict(mesh=str(out_msh), metadata=str(meta_path),
                seg_shape_resampled=list(seg_data.shape), voxel_size_mm=[float(x) for x in vs],
                n_labels=int(len(np.unique(seg_data[seg_data > 0]))),
                n_surface_faces_raw=n_mc, n_surface_vertices=int(len(v)), n_surface_faces=int(len(f)),
                n_vertices=int(len(tv)), n_tets=int(len(tc)),
                tissue_tets=meta["tissue_type_counts"], build_params=meta["build_params"])


# --------------------------------------------------------------------------- 2. fibre config
def build_fibre_config(seg, out_json, *, min_slices: int = 3, smooth_sigma: float = 1.0,
                       method: str = "pca"):
    """Per-muscle fibre direction + centreline → the fibre-aligned σ config
    (the recipe of ``scripts/mri/build_fiber_config.py``). Returns ``(fibre_model, info)``;
    the model still needs ``estimate_cross_sections()`` before it can seed a fibre bed."""
    from emgforge.mri.core.fiber_directions import MuscleFiberModel

    fm = MuscleFiberModel(str(seg))
    fm.estimate_fibers(method=method)
    fm.estimate_centerlines(min_slices=min_slices, smooth_sigma=smooth_sigma)
    fm.save_config(str(out_json))
    mus = [m for m in fm.muscles.values() if m.tissue_type == "muscle"]
    info = dict(config=str(out_json), n_labels=len(fm.muscles), n_muscles=len(mus),
                n_centerlines=sum(m.centerline is not None for m in mus),
                n_bone=sum(m.tissue_type == "bone" for m in fm.muscles.values()),
                n_fat=sum(m.tissue_type == "fat_skin" for m in fm.muscles.values()),
                seg_shape=list(fm.seg_data.shape), voxel_size_mm=[float(x) for x in fm.voxel_size])
    return fm, info


# --------------------------------------------------------------------------- 4/5. bed + pool
def poisson_bed(fm, muscle: int, *, density: float = 4.0, min_fibers: int = 30, seed: int = 0):
    """The straight (morphing-disk) Poisson-disk fibre bed of one muscle — the pool's bed."""
    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds

    beds = build_muscle_beds(fm, density=density, method="poisson", labels=[muscle],
                             min_fibers=min_fibers, seed=seed)
    if muscle not in beds:
        raise ValueError(f"label {muscle} ({muscle_name(muscle)}) yields no fibre bed "
                         f"(fewer than {min_fibers} fibres or no centreline/cross-section)")
    return beds[muscle]


def henneman_pool(bed, n_mu: int, *, size_min: int = 5, size_max: int = 400, seed: int = 0):
    """Henneman pool on a bed: exponential sizes in ``[size_min, min(size_max, N)]``,
    territories grown around random anchor fibres, index = size rank = recruitment order."""
    from emgforge.mri.core.motor_unit_pool import sample_henneman_pool

    N = len(bed.r_norms)
    return sample_henneman_pool(bed, n_mu=n_mu, size_min=size_min, size_max=min(size_max, N), seed=seed)


def bed_arc_geometry(bed):
    """Per-fibre arc-length step and total arc length (the AP runs along the curve)."""
    seg = np.linalg.norm(np.diff(bed.paths, axis=1), axis=2)
    return seg.mean(axis=1), seg.sum(axis=1)


def mu_synth_bed(bed, mu, arc_dz, L_fib, *, iz_frac: float = IZ_FRAC_FCU,
                 iz_jitter: float = IZ_JITTER, v: float = 4.0) -> FibreBed:
    """The synthesis FibreBed of one motor unit: NMJ at fraction ``iz_frac`` (+ a per-fibre
    ``N(0, iz_jitter)`` scatter, seeded by the unit index) of each fibre's arc length,
    ``posz = (Lp − Ld)/2`` so the array centre stays the φ sampling centre."""
    idx = mu.fiber_idxs
    rng = np.random.default_rng(int(mu.idx))
    izf = np.clip(iz_frac + rng.normal(0.0, iz_jitter, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1.0 - izf) * L_fib[idx]
    return FibreBed.from_arrays(dz_mm=arc_dz[idx], len1_mm=Lp, len2_mm=Ld,
                                posz_mm=(Lp - Ld) / 2.0, v=v)


# --------------------------------------------------------------------------- 6. volume conductor
def build_volume_conductor(mesh, fibre_config, seg, *, skin_shell_mm: float = 1.5,
                           sigma_mode: str = "centerline", **options):
    """``MRIFEMModel`` on the mesh: per-cell σ from the tissue tag, muscle tensors rotated
    onto each muscle's centreline tangent, a ``skin_shell_mm`` skin layer, KSP set up."""
    from emgforge.mri.core.fem_solver import MRIFEMModel

    return MRIFEMModel(str(mesh), fiber_config=str(fibre_config), nifti_path=str(seg),
                       skin_shell_mm=skin_shell_mm, sigma_mode=sigma_mode, **options)


# --------------------------------------------------------------------------- 7. electrodes
def exterior_triangles(fem):
    """(n_facets, 3, 3) coordinates of the exterior (skin) mesh triangles."""
    # boundary facets bound exactly one cell (same identification as MRIFEMModel._apply_skin_shell)
    top = fem.mesh.topology
    top.create_connectivity(2, 3); top.create_connectivity(2, 0)
    f2c, f2v = top.connectivity(2, 3), top.connectivity(2, 0)
    nf = top.index_map(2).size_local
    tri = [f2v.links(fi) for fi in range(nf) if len(f2c.links(fi)) == 1]
    return fem.mesh.geometry.x[np.array(tri)]


def _ray_outermost_hit(tri, o, d):
    """Möller–Trumbore: distance to the outermost exterior triangle along ray o + t d."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d[None, :], e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det); inv[ok] = 1.0 / det[ok]
    s = o[None, :] - v0
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (d[None, :] * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
    return float(t[hit].max()) if hit.any() else np.nan


def seg_slice_index(fm, z_mm: float) -> int:
    return int(np.clip(round(z_mm / fm.voxel_size[2]), 0, fm.seg_data.shape[2] - 1))


def limb_centre(fm, z_mm: float):
    """Area centroid (x, y) of all tissue (label > 0) on the axial segmentation slice at z —
    unbiased by mesh-vertex density (a mesh-vertex mean is skewed toward the dense skin)."""
    kz = seg_slice_index(fm, z_mm)
    vox = np.argwhere(fm.seg_data[:, :, kz] > 0)
    return vox.mean(axis=0) * fm.voxel_size[:2]


def surface_contour(tri, centre_xy, z_mm: float, theta_deg: np.ndarray):
    """Skin-surface points (K, 3) of the mesh at height z along rays at θ (deg) from centre_xy."""
    o = np.array([centre_xy[0], centre_xy[1], z_mm])
    pts = np.zeros((len(theta_deg), 3))
    for k, th in enumerate(np.radians(theta_deg)):
        d = np.array([np.cos(th), np.sin(th), 0.0])
        t = _ray_outermost_hit(tri, o, d)
        pts[k] = o + t * d
    return pts


def grid_electrodes(fem, fm, bed, *, m: int = 5, n: int = 5, ied_mm: float = 10.0,
                    zc_frac: float = 0.5, theta_half_deg: float = 45.0, theta_step_deg: float = 0.25):
    """Regular ``m × n`` skin grid over the bed's muscle: ``m`` rows along the arm (z,
    ``ied_mm`` apart) and ``n`` columns around the arm at ``ied_mm`` *arc length* along the
    skin contour, centred on the ray from the limb axis (tissue centroid at the grid centre
    z = ``zc_frac`` of the mesh) through the muscle centroid — which is also the skin point
    nearest the muscle. One fixed axis for all rows, so the only row-to-row xy drift is the
    true skin slant. Returns ``(elec_xyz (m, n, 3), info)``."""
    tri = exterior_triangles(fem)
    X = fem.mesh.geometry.x
    z_lo, z_hi = float(X[:, 2].min()), float(X[:, 2].max())
    zc = z_lo + zc_frac * (z_hi - z_lo)
    cxy = limb_centre(fm, zc)
    mus_ang = float(np.degrees(np.arctan2(bed.centroid_xy[1] - cxy[1], bed.centroid_xy[0] - cxy[0])))
    th = mus_ang + np.arange(-theta_half_deg, theta_half_deg + 0.01, theta_step_deg)
    elec = np.zeros((m, n, 3))
    for i in range(m):
        z = zc + (i - (m - 1) / 2) * ied_mm
        P = surface_contour(tri, cxy, z, th)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        s -= np.interp(mus_ang, th, s)                        # arc length 0 at the muscle direction
        half = (n - 1) / 2 * ied_mm
        if half > min(-s.min(), s.max()):
            raise ValueError(f"a {n}-column grid at {ied_mm} mm needs ±{half:.0f} mm of skin arc; the "
                             f"±{theta_half_deg}° contour spans {-s.min():.0f}/{s.max():.0f} mm — raise theta_half_deg")
        for j in range(n):
            target = (j - (n - 1) / 2) * ied_mm
            elec[i, j] = [np.interp(target, s, P[:, d]) for d in range(3)]
    ied_z = np.linalg.norm(np.diff(elec, axis=0), axis=2)
    ied_t = np.linalg.norm(np.diff(elec, axis=1), axis=2)
    info = dict(fcu_ang_deg=mus_ang, muscle_ang_deg=mus_ang, zc_mm=zc, limb_centre_xy=cxy.tolist(),
                ied_along_mm=float(np.median(ied_z)) if m > 1 else float("nan"),
                ied_across_mm=float(np.median(ied_t)) if n > 1 else float("nan"),
                ied_along_range=[float(ied_z.min()), float(ied_z.max())] if m > 1 else [float("nan")] * 2,
                ied_across_range=[float(ied_t.min()), float(ied_t.max())] if n > 1 else [float("nan")] * 2,
                n_skin_triangles=int(len(tri)))
    return elec, info


# --------------------------------------------------------------------------- 8. lead fields
def phi_along_paths(fem, paths, cells=None):
    """φ of the current solve sampled along every path → (N, Nz)."""
    paths = np.asarray(paths, dtype=float)
    pts = np.ascontiguousarray(paths.reshape(-1, 3))
    return fem.evaluate_solution_at_points(pts, cells=cells).reshape(paths.shape[:2])


def solve_grid_leadfields(fem, elec, paths, *, source_sigma: float = 5.0, log=None) -> dict:
    """One reciprocity solve per electrode (Gaussian source of width ``source_sigma`` mm at
    the skin point), φ sampled along every fibre path. The fibre points' mesh cells are
    located once and reused for every electrode. Returns ``phi_grid (m, n, N, Nz)`` and
    the per-electrode solve / sampling times."""
    elec = np.asarray(elec, dtype=float)
    paths = np.asarray(paths, dtype=float)
    m, n = elec.shape[:2]
    N, Nz = paths.shape[:2]
    pts = np.ascontiguousarray(paths.reshape(-1, 3))
    t0 = time.time()
    cells = fem.locate_points(pts)
    t_locate = time.time() - t0
    phi = np.zeros((m, n, N, Nz))
    t_solve, t_samp = [], []
    for i in range(m):
        for j in range(n):
            t0 = time.time()
            fem.solve_for_point(elec[i, j], source_sigma=source_sigma)
            t_solve.append(time.time() - t0)
            t0 = time.time()
            phi[i, j] = fem.evaluate_solution_at_points(pts, cells=cells).reshape(N, Nz)
            t_samp.append(time.time() - t0)
        if log:
            log(f"  lead fields: row {i + 1}/{m} done (solve {np.mean(t_solve):.2f} s, "
                f"sample {np.mean(t_samp):.2f} s per electrode)")
    return dict(phi_grid=phi, locate_s=t_locate, solve_s=t_solve, sample_s=t_samp)


# --------------------------------------------------------------------------- 9/10. synthesis
def production_config(*, fs: float = 2048.0, v: float = 4.0, w: int = 256) -> SpatialConfig:
    """The direct line-source recipe (``synthesis/DIRECT_LINE_SOURCE.md``): 3-monopole fit of
    φ, short edge taper, 2× upsampling, second-derivative current source, one-sided tendon
    window, physical time from −10 ms (t = 0 at the NMJ discharge)."""
    return SpatialConfig(denoise="monopole", denoise_n_poles=3,
                         fiber_window="one_sided", tukey_alpha=0.25,
                         csd_derivative=2, upsample_factor=2,
                         fsamp=fs, w=w, edge_taper_left=5, edge_taper_right=10,
                         t_start_ms=-10.0, v=v)


def config_dict(cfg) -> dict:
    """A synthesis config as plain JSON-able values."""
    return {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
            for k, v in asdict(cfg).items()}


_CTX: dict = {}          # fork-shared context of the worker pools (read-only in the children)


def _pool(workers: int):
    return mp.get_context("fork").Pool(int(workers))


def _condition_task(e):
    phi, dz, n = _CTX["phi"], _CTX["dz"], _CTX["n_poles"]
    t0 = time.time()
    out = np.array([denoise_field_n(phi[e, i], float(dz[i]), n=n) for i in range(phi.shape[1])])
    return e, out, time.time() - t0


def condition_leadfields(phi, arc_dz, *, n_poles: int = 3, workers: int = 1, log=None):
    """The recipe's ``denoise="monopole"`` step — a free-position ``n_poles``-monopole fit of
    φ(z) — done once per (electrode, fibre) pair.

    Inside ``field_to_muap`` the fit runs per SFAP, so a fibre shared by k motor units is
    fitted k times per electrode; the fit depends on neither the unit's innervation nor
    its lengths, so factoring it out changes no number (:func:`verify_conditioning`).
    Returns ``(phi_conditioned, secs)`` with the same shape as ``phi`` (``(E, N, Nz)`` or
    ``(m, n, N, Nz)``); feed it to :func:`muap_tensor` with ``conditioned=True``.
    """
    phi = np.asarray(phi, dtype=float)
    shape = phi.shape
    P = phi.reshape(-1, *shape[-2:])
    E = P.shape[0]
    _CTX.update(phi=P, dz=np.asarray(arc_dz, dtype=float), n_poles=int(n_poles))
    out = np.zeros_like(P)
    secs = np.zeros(E)
    t0 = time.time()
    if workers > 1:
        with _pool(workers) as pool:
            for k, (e, Pe, dt) in enumerate(pool.imap_unordered(_condition_task, range(E))):
                out[e], secs[e] = Pe, dt
                if log and ((k + 1) % max(E // 5, 1) == 0 or k + 1 == E):
                    log(f"  conditioning: {k + 1}/{E} electrodes ({time.time() - t0:.0f} s wall)")
    else:
        for e in range(E):
            _, out[e], secs[e] = _condition_task(e)
            if log and ((e + 1) % max(E // 5, 1) == 0 or e + 1 == E):
                log(f"  conditioning: {e + 1}/{E} electrodes ({time.time() - t0:.0f} s wall)")
    return out.reshape(shape), secs


def _mu_task(k):
    bed, pool, arc_dz, L_fib, phi, cfg = (_CTX[n] for n in ("bed", "pool", "arc_dz", "L_fib", "phi", "cfg"))
    iz_frac, iz_jitter, v = _CTX["iz_frac"], _CTX["iz_jitter"], _CTX["v"]
    mu = pool[k]
    idx = mu.fiber_idxs
    sb = mu_synth_bed(bed, mu, arc_dz, L_fib, iz_frac=iz_frac, iz_jitter=iz_jitter, v=v)
    E = phi.shape[0]
    W = np.zeros((E, cfg.w))
    t_ms = None
    t0 = time.time()
    for e in range(E):
        res = field_to_muap(phi[e][idx], sb, cfg)
        W[e] = res.muap
        t_ms = res.t_ms
    return k, W, t_ms, time.time() - t0


def muap_tensor(phi, bed, pool, cfg: SpatialConfig, *, iz_frac: float = IZ_FRAC_FCU,
                iz_jitter: float = IZ_JITTER, v: float = 4.0, conditioned: bool = False,
                workers: int = 1, log=None):
    """The ``(n_mu, E, w)`` MUAP tensor: per unit, ``field_to_muap`` of its fibres' lead
    fields at every electrode with the direct recipe ``cfg`` (largest units first,
    parallel over units). With ``conditioned=True`` the input is
    :func:`condition_leadfields` output and the recipe runs with ``denoise="none"`` —
    the same numbers, without refitting shared fibres. Returns ``(W, t_ms, secs)``."""
    phi = np.asarray(phi, dtype=float)
    P = phi.reshape(-1, *phi.shape[-2:])
    if conditioned:
        if cfg.denoise != "monopole":
            raise ValueError("conditioned=True only replaces the recipe's monopole denoise")
        run_cfg = replace(cfg, denoise="none")
    else:
        run_cfg = cfg
    arc_dz, L_fib = bed_arc_geometry(bed)
    _CTX.update(bed=bed, pool=pool, arc_dz=arc_dz, L_fib=L_fib, phi=P, cfg=run_cfg,
                iz_frac=float(iz_frac), iz_jitter=float(iz_jitter), v=float(v))
    n_mu, E = len(pool), P.shape[0]
    W = np.zeros((n_mu, E, cfg.w))
    secs = np.zeros(n_mu)
    t_ms = None
    order = list(np.argsort([-p.size for p in pool]))       # largest first: best load balance
    t0 = time.time()
    n_fin = 0

    def _take(k, Wk, tk, dt):
        nonlocal t_ms, n_fin
        W[k], secs[k], t_ms = Wk, dt, tk
        n_fin += 1
        if log and (n_fin % max(n_mu // 5, 1) == 0 or n_fin == n_mu):
            log(f"  MUAPs: {n_fin}/{n_mu} units ({time.time() - t0:.0f} s wall)")

    if workers > 1:
        with _pool(workers) as pl:
            for k, Wk, tk, dt in pl.imap_unordered(_mu_task, order):
                _take(k, Wk, tk, dt)
    else:
        for k in order:
            _take(*_mu_task(k))
    return W, t_ms, secs


def verify_conditioning(phi_raw, phi_cond, bed, pool, cfg: SpatialConfig, k: int, *,
                        iz_frac: float = IZ_FRAC_FCU, iz_jitter: float = IZ_JITTER, v: float = 4.0) -> dict:
    """Full recipe on raw φ vs. the factored path on conditioned φ for unit ``k`` on every
    electrode — the check that :func:`condition_leadfields` changes nothing."""
    phi_raw = np.asarray(phi_raw, dtype=float).reshape(-1, *np.shape(phi_raw)[-2:])
    phi_cond = np.asarray(phi_cond, dtype=float).reshape(-1, *np.shape(phi_cond)[-2:])
    arc_dz, L_fib = bed_arc_geometry(bed)
    mu = pool[k]
    idx = mu.fiber_idxs
    sb = mu_synth_bed(bed, mu, arc_dz, L_fib, iz_frac=iz_frac, iz_jitter=iz_jitter, v=v)
    fact = replace(cfg, denoise="none")
    full = np.array([field_to_muap(phi_raw[e][idx], sb, cfg).muap for e in range(phi_raw.shape[0])])
    fast = np.array([field_to_muap(phi_cond[e][idx], sb, fact).muap for e in range(phi_cond.shape[0])])
    return dict(mu=int(k), size=int(mu.size),
                max_abs_diff_V=float(np.abs(full - fast).max()),
                p2p_max_V=float(np.ptp(full, axis=1).max()),
                identical=bool(np.array_equal(full, fast)))


__all__ = [
    "DEFAULT_SEG", "LABELS_JSON", "FCU", "IZ_FRAC_FCU", "IZ_JITTER", "muscle_name",
    "build_mesh", "build_fibre_config", "poisson_bed", "henneman_pool", "bed_arc_geometry",
    "mu_synth_bed", "build_volume_conductor", "exterior_triangles", "seg_slice_index",
    "limb_centre", "surface_contour", "grid_electrodes", "phi_along_paths",
    "solve_grid_leadfields", "production_config", "config_dict", "condition_leadfields",
    "muap_tensor", "verify_conditioning",
]
