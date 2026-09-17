"""Shared helpers for the F2 paper figures (6, 7, 8) and datasets (D2, D3).

Everything heavy is cached under ``_results/paper/cache/`` so each figure script is
re-runnable on its own:

* the WR forearm fibre model, the FCU Poisson (straight) and harmonic (single-NMJ
  streamline) fibre beds and the 100-MU Henneman pool — exactly the recipe of
  ``scripts/mri/sample_mu_pool.py`` / ``scripts/activation/build_grid_tensor.py``;
* a regular 5×5 HD-EMG electrode grid (10 mm IED) on the mesh skin surface over FCU,
  placed by ray-casting the exterior mesh triangles (the library helper
  ``get_skin_surface_point`` snaps to the outermost mesh vertex, which collapses nearby
  angles onto one vertex — the older cached grid had two duplicated electrodes);
* one FEM reciprocity solve per electrode with φ sampled along every bed fibre;
* the (100, 25, 256) MUAP tensor of the direct line-source synthesis (per-MU
  ``field_to_muap`` with the direct recipe's ``SpatialConfig``; parallel over MUs).

Nothing under ``src/`` is modified; only library calls are used.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
# The anatomy/electrode/synthesis helpers that used to live here are now the package's
# pipeline stages (emgforge.mri.pipeline; scripts/run_pipeline.py runs them end to end).
# The F2 wrappers below bind them to this file's constants so the figure scripts are unchanged.
from emgforge.mri import pipeline as _P
from emgforge.mri.pipeline import (  # noqa: F401  (re-exported for the figure scripts)
    bed_arc_geometry, exterior_triangles, limb_centre, phi_along_paths, production_config,
    seg_slice_index, surface_contour,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEG = ROOT / "src/emgforge/mri/data/forearm_WR_segmentation.nii.gz"
MESH = ROOT / "_results/sanity/fem_cache/forearm_WR.msh"
FIBER_CFG = ROOT / "_results/sanity/fem_cache/forearm_WR_fibers.json"
LABELS_JSON = ROOT / "src/emgforge/mri/data/pd_lab_labels.json"
CACHE = ROOT / "_results/paper/cache"
DATASETS = ROOT / "_results/paper/datasets"
KEY_JSON = HERE / "key_numbers_f2.json"

FCU = 8
N_MU = 100
DENSITY = 4.0          # fibres / mm² (Poisson bed)
IZ_FRAC = 0.305        # innervation-zone fraction along the fibre (FCU)
IZ_JITTER = 0.02       # per-fibre IZ scatter (fraction of fibre length)
SEED = 0
FS = 2048.0
CV = 4.0               # m/s
M = 5                  # grid is M×M
IED_MM = 10.0          # inter-electrode distance (along and across)
ZC_FRAC = 0.5          # grid centre at mid-mesh z
GRID_TAG = f"M{M}_ied{IED_MM:.0f}"

# the direct line-source synthesis recipe: the spatial engine's production config
# (emgforge.mri.pipeline.production_config == scripts/mri/sample_mu_pool.py::build_config)
SPCFG: SpatialConfig = production_config(fs=FS, v=CV, w=256)


def spcfg_dict() -> dict:
    """The SpatialConfig fields as plain JSON-able values (for the dataset manifest)."""
    from dataclasses import asdict
    return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in asdict(SPCFG).items()}


# --------------------------------------------------------------------------- anatomy
def load_fibre_model():
    from emgforge.mri.core.fiber_directions import MuscleFiberModel
    fm = MuscleFiberModel(str(SEG))
    fm.estimate_centerlines()
    fm.estimate_cross_sections()
    return fm


def poisson_bed(fm):
    """The straight (morphing-disk, Poisson-disk sampled) FCU bed — the pool's bed."""
    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    return build_muscle_beds(fm, density=DENSITY, method="poisson", labels=[FCU],
                             min_fibers=30)[FCU]


HARM_GRID_MM = 0.5     # streamline seed spacing for the density-matched harmonic bed (library default 2.0)


def harmonic_bed(fm, grid_mm: float = HARM_GRID_MM, force: bool = False):
    """The single-NMJ harmonic-streamline FCU bed (masked Laplace + one streamline per seed
    of a ``grid_mm`` lattice on the mid cross-section), cached per seed spacing."""
    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"fcu_harmonic_bed_g{grid_mm:g}.pkl"
    if f.exists() and not force:
        with open(f, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    bed = build_muscle_beds(fm, density=DENSITY, method="harmonic", labels=[FCU],
                            series=False, min_fibers=30, grid_mm=grid_mm)[FCU]
    bed.build_seconds = time.time() - t0
    with open(f, "wb") as fh:
        pickle.dump(bed, fh)
    return bed


def harmonic_frame(fm):
    """The FCU's PCA frame of ``HarmonicFibreField`` (no Laplace solve) — gives the
    longitudinal fraction of any point in the muscle (``frame._long_fraction``)."""
    from emgforge.mri.core.harmonic_fibers import HarmonicFibreField
    return HarmonicFibreField(fm.seg_data == FCU, fm.voxel_size, solve=False)


def harmonic_bed_at_iz(bed, frame, iz_fraction: float = IZ_FRAC):
    """The same single-NMJ harmonic bed with every NMJ re-placed at longitudinal fraction
    ``iz_fraction`` of the muscle — the placement rule of
    ``HarmonicFibreField.long_fibers(iz_fraction=…)`` (nearest sample of the arc-resampled
    path in the muscle's PCA frame; a streamline that does not reach the IZ gets its NMJ at
    the nearer end) applied to the cached streamlines, which do not depend on the IZ.
    ``build_muscle_beds`` / ``build_harmonic_fibers`` do not expose ``iz_fraction`` (they
    build at the library default 0.5), so the cached bed is re-innervated here instead of
    re-solving the Laplace field; ``make_fig_mri_fibres.py`` checks this against a direct
    ``long_fibers(iz_fraction=…)`` call.

    ``iz_fraction`` is in the library's frame: 0 at the end of the muscle where the PCA axis
    ``frame.p1`` starts, whose sign is arbitrary (for the FCU it points distally, so 0.305
    would be the far end). To innervate on a given z-plane use the frame fraction of points
    on it, e.g. ``frame._long_fraction(nmj_points).mean()``."""
    import copy
    out = copy.copy(bed)
    n = len(bed.paths)
    half1, half2, posz, izf = (np.zeros(n) for _ in range(4))
    for i, p in enumerate(bed.paths):
        p = np.asarray(p)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        frac = frame._long_fraction(p)
        j = int(np.argmin(np.abs(frac - iz_fraction)))
        nmj, total = float(arc[j]), float(arc[-1])
        half1[i], half2[i] = max(nmj, 1e-6), max(total - nmj, 1e-6)
        posz[i] = nmj - (len(p) // 2) * bed.dz_mm
        izf[i] = float(frac[j])
    out.half1_mm, out.half2_mm, out.posz_mm, out.iz_fractions = half1, half2, posz, izf
    out.half_mm = float(np.mean(half1 + half2) / 2.0)
    out.iz_target = float(iz_fraction)
    return out


def henneman_pool(bed):
    from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
    N = len(bed.r_norms)
    return sample_henneman_pool(bed, n_mu=N_MU, size_min=5, size_max=min(400, N), seed=SEED)


def mu_synth_bed(bed, mu, arc_dz, L_fib):
    """The synthesis FibreBed of one MU: IZ at IZ_FRAC (+jitter), posz = (Lp−Ld)/2."""
    return _P.mu_synth_bed(bed, mu, arc_dz, L_fib, iz_frac=IZ_FRAC, iz_jitter=IZ_JITTER, v=CV)


def fcu_outline(fm, z_mm: float, label: int = FCU):
    """Closed contour(s) (K, 2) in mm of one label on the axial segmentation slice."""
    from skimage.measure import find_contours
    kz = seg_slice_index(fm, z_mm)
    m = (fm.seg_data[:, :, kz] == label).astype(float)
    cs = find_contours(m, 0.5)
    vs = fm.voxel_size
    return [np.c_[c[:, 0] * vs[0], c[:, 1] * vs[1]] for c in sorted(cs, key=len, reverse=True)]


def mask_containment(paths, fm, label: int = FCU):
    """Fraction of fibre-path samples whose voxel is inside the label mask
    (the computation of tests/mri/test_harmonic_fibres.py::test_paths_stay_in_mask)."""
    mask = fm.seg_data == label
    vs, shape = fm.voxel_size, np.array(mask.shape)
    inside = total = 0
    per_fibre = []
    for p in paths:
        ci = np.round(np.asarray(p)[:, :3] / vs).astype(int)
        okb = np.all((ci >= 0) & (ci < shape), axis=1)
        vals = np.zeros(len(ci), bool)
        okc = ci[okb]
        vals[okb] = mask[okc[:, 0], okc[:, 1], okc[:, 2]]
        inside += int(vals.sum()); total += len(vals)
        per_fibre.append(vals.mean())
    return inside / max(total, 1), np.array(per_fibre)


# --------------------------------------------------------------------------- FEM
def build_fem():
    from emgforge.mri.core.fem_solver import MRIFEMModel
    t0 = time.time()
    fem = MRIFEMModel(str(MESH), fiber_config=str(FIBER_CFG), nifti_path=str(SEG),
                      skin_shell_mm=1.5, sigma_mode="centerline")
    fem.build_seconds = time.time() - t0
    return fem


def grid_electrodes(fem, fm, bed):
    """Regular M×M skin grid over FCU: rows along the arm (z, IED_MM apart) and columns
    around the arm at IED_MM *arc length* along the skin contour, centred on the ray from
    the limb axis (tissue centroid at the grid centre z) through the FCU centroid — which
    is also the skin point nearest the muscle (``emgforge.mri.pipeline.grid_electrodes``
    with this file's constants). Returns (elec_xyz (M, M, 3), info)."""
    return _P.grid_electrodes(fem, fm, bed, m=M, n=M, ied_mm=IED_MM, zc_frac=ZC_FRAC)


def ensure_grid_leadfields(force: bool = False):
    """One reciprocity solve per grid electrode, φ along every Poisson-bed fibre. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"phigrid_f2_{GRID_TAG}.npz"
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=True))
    fm = load_fibre_model()
    bed = poisson_bed(fm)
    fem = build_fem()
    elec, info = grid_electrodes(fem, fm, bed)
    N, Nz = bed.paths.shape[:2]
    phi = np.zeros((M, M, N, Nz))
    t_solve, t_samp = [], []
    for i in range(M):
        for j in range(M):
            t0 = time.time(); fem.solve_for_point(elec[i, j], source_sigma=5.0); t_solve.append(time.time() - t0)
            t0 = time.time(); phi[i, j] = phi_along_paths(fem, bed.paths); t_samp.append(time.time() - t0)
        print(f"  lead fields: row {i+1}/{M} done (solve {np.mean(t_solve):.2f}s, sample {np.mean(t_samp):.1f}s each)", flush=True)
    out = dict(phi_grid=phi, elec_xyz=elec, fcu_ang=info["fcu_ang_deg"], zc_mm=info["zc_mm"],
               ied_along_mm=info["ied_along_mm"], ied_across_mm=info["ied_across_mm"],
               ied_along_range=np.array(info["ied_along_range"]), ied_across_range=np.array(info["ied_across_range"]),
               limb_centre_xy=np.array(info["limb_centre_xy"]),
               fem_build_s=fem.build_seconds, solve_s=float(np.mean(t_solve)), sample_s=float(np.mean(t_samp)),
               bed_xy_mid=bed.xy_mid, bed_paths=bed.paths.astype(np.float32))
    np.savez_compressed(f, **out)
    return out


# --------------------------------------------------------------------------- MUAP tensor
_CTX: dict = {}


def _mu_task(k):
    bed, pool, arc_dz, L_fib, phi = (_CTX[n] for n in ("bed", "pool", "arc_dz", "L_fib", "phi"))
    mu = pool[k]; idx = mu.fiber_idxs
    sb = mu_synth_bed(bed, mu, arc_dz, L_fib)
    E = phi.shape[0]
    W = np.zeros((E, SPCFG.w)); t_ms = None
    t0 = time.time()
    for e in range(E):
        res = field_to_muap(phi[e][idx], sb, SPCFG)
        W[e] = res.muap; t_ms = res.t_ms
    return k, W, t_ms, time.time() - t0


def ensure_muap_tensor(n_workers: int = 10, force: bool = False):
    """(N_MU, M*M, w) direct-method MUAP tensor on the regular grid, parallel over MUs. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"muap_tensor_f2_{GRID_TAG}.npz"
    if f.exists() and not force:
        d = np.load(f)
        if int(d["done"]) >= N_MU:
            return dict(d)
    lf = ensure_grid_leadfields()
    fm = load_fibre_model(); bed = poisson_bed(fm); pool = henneman_pool(bed)
    arc_dz, L_fib = bed_arc_geometry(bed)
    phi = lf["phi_grid"].reshape(M * M, *lf["phi_grid"].shape[2:])
    _CTX.update(bed=bed, pool=pool, arc_dz=arc_dz, L_fib=L_fib, phi=phi)
    W = np.zeros((N_MU, M * M, SPCFG.w)); t_ms = None; done = np.zeros(N_MU, bool); secs = np.zeros(N_MU)
    if f.exists():
        d = np.load(f)
        if d["W"].shape == W.shape:
            W, done, secs = d["W"], d["done_mask"], d["secs"]; t_ms = d["t_ms"]
    todo = [k for k in np.argsort([-p.size for p in pool]) if not done[k]]   # largest first
    t0 = time.time(); n_fin = int(done.sum())
    print(f"MUAP tensor: {len(todo)} MUs × {M*M} electrodes on {n_workers} workers ...", flush=True)
    with mp.get_context("fork").Pool(n_workers) as P:
        for k, Wk, tk, dt in P.imap_unordered(_mu_task, todo):
            W[k], t_ms, done[k], secs[k] = Wk, tk, True, dt; n_fin += 1
            if n_fin % 10 == 0 or n_fin == N_MU:
                np.savez_compressed(f, W=W, t_ms=t_ms, M=M, sizes=np.array([p.size for p in pool]),
                                    done=int(done.sum()), done_mask=done, secs=secs, wall_s=time.time() - t0)
                print(f"  {n_fin}/{N_MU} MUs ({time.time()-t0:.0f}s wall)", flush=True)
    wall = time.time() - t0
    np.savez_compressed(f, W=W, t_ms=t_ms, M=M, sizes=np.array([p.size for p in pool]),
                        done=int(done.sum()), done_mask=done, secs=secs, wall_s=wall)
    print(f"MUAP tensor done: {wall:.0f}s wall, {secs.sum():.0f}s CPU", flush=True)
    return dict(np.load(f))


# --------------------------------------------------------------------------- MUAP source (new / legacy)
LEGACY_TENSOR = ROOT / "_results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz"          # 42 MUs, old grid
LEGACY_PHIGRID = ROOT / "_results/mu_pool/electrode_grid/_phigrid_M5_dt15_z0.30-0.70_N637.npz"
LEGACY_POOL = ROOT / "_results/mu_pool/spatial/mu_pool.npz"                              # 100 MUs, old centre electrode


def tensor_ready() -> bool:
    """True when the regular-grid MUAP tensor cache holds all N_MU units (never triggers a build)."""
    f = CACHE / f"muap_tensor_f2_{GRID_TAG}.npz"
    if not f.exists():
        return False
    try:
        return int(np.load(f)["done"]) >= N_MU
    except Exception:
        return False


def load_muaps(source: str | None = None) -> dict:
    """The MUAP set used by figs 7/8 and D2/D3.

    ``source`` (or env ``F2_TENSOR``): ``"new"`` — the regular 10 mm grid tensor (all 100 MUs;
    built by :func:`ensure_muap_tensor`); ``"legacy42"`` — the pre-existing
    ``scripts/activation/build_grid_tensor.py`` tensor (42 smallest MUs on the old
    vertex-snapped grid) with the 100 single-channel MUAPs of ``mu_pool.npz`` (its electrode
    is that grid's centre electrode — bit-identical waveforms); ``"auto"`` — new if complete,
    else legacy (never starts a build). Waveforms in V, t in ms.
    """
    source = source or os.environ.get("F2_TENSOR", "auto")
    if source == "auto":
        source = "new" if tensor_ready() else "legacy42"
    if source == "new":
        lf = ensure_grid_leadfields(); ten = ensure_muap_tensor()
        W = ten["W"]
        return dict(source="new", W=W, t_ms=ten["t_ms"], sizes=ten["sizes"], muap_single=W[:, E0],
                    elec_xyz=lf["elec_xyz"], n_grid_mus=int(W.shape[0]),
                    ied_along=float(lf["ied_along_mm"]), ied_across=float(lf["ied_across_mm"]),
                    ied_along_range=lf["ied_along_range"].tolist(), ied_across_range=lf["ied_across_range"].tolist(),
                    zc_mm=float(lf["zc_mm"]), fcu_ang=float(lf["fcu_ang"]),
                    grid_note=f"regular {M}×{M} grid, {IED_MM:.0f} mm IED along (z) and across (arc length), ray-cast onto the "
                              "mesh skin surface over FCU (f2_common.grid_electrodes)",
                    fem_build_s=float(lf["fem_build_s"]), solve_s=float(lf["solve_s"]), sample_s=float(lf["sample_s"]),
                    tensor_wall_s=float(ten["wall_s"]), tensor_cpu_s=float(ten["secs"].sum()),
                    tensor_s_per_mu=[float(ten["secs"].min()), float(ten["secs"].max())])
    if source == "legacy42":
        ten = np.load(LEGACY_TENSOR); pn = np.load(LEGACY_POOL); eg = np.load(LEGACY_PHIGRID)
        n = int(ten["done"]); W = ten["W"][:n]; elec = eg["elec_xyz"]
        if not np.allclose(pn["muap_wave"][:n], W[:, E0]):
            raise RuntimeError("legacy tensor centre electrode ≠ mu_pool.npz MUAPs")
        dz = np.linalg.norm(np.diff(elec, axis=0), axis=2); dt = np.linalg.norm(np.diff(elec, axis=1), axis=2)
        return dict(source="legacy42", W=W, t_ms=ten["t_ms"], sizes=pn["sizes"], muap_single=pn["muap_wave"],
                    elec_xyz=elec, n_grid_mus=n,
                    ied_along=float(np.median(dz)), ied_across=float(np.median(dt)),
                    ied_along_range=[float(dz.min()), float(dz.max())], ied_across_range=[float(dt.min()), float(dt.max())],
                    zc_mm=float(elec[M // 2, M // 2, 2]), fcu_ang=float(eg["fcu_ang"]),
                    grid_note="legacy grid of scripts/mri/mu_electrode_grid.py: 5×5, dθ = 15°, z 0.30–0.70 of the mesh, each "
                              "electrode snapped to the outermost mesh vertex (IED irregular: along 20–31 mm, across 0–20 mm; "
                              f"two duplicated electrodes in row 2); MUAP tensor covers the {n} smallest of the 100 MUs",
                    fem_build_s=None, solve_s=None, sample_s=None, tensor_wall_s=None, tensor_cpu_s=None, tensor_s_per_mu=None)
    raise ValueError(f"unknown MUAP source {source!r}")


# --------------------------------------------------------------------------- activation trials
TRAP = dict(rise_s=0.5, hold_s=2.0, fall_s=0.5, lead_s=0.2)       # the D3 trapezoid
LEVELS = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]                           # D3 drive levels
COMMON_DRIVE = dict(sigma=0.015, cutoff_hz=2.0)                    # shared low-pass drive noise
DYN = dict(duration_s=6.0, cycles=2, drive_sigma=0.012, drive_seed=1, amp_gain=0.8, warp_gain=0.22)
E0 = (M // 2) * M + M // 2                                          # centre electrode index in (M*M,)


def activation_models():
    """The motoneuron pool (recruitment + onion-skin rate coding) and its twitch/force layer."""
    from emgforge.activation import MotoneuronPool, TwitchPool
    mn = MotoneuronPool(n_mu=N_MU, fs=FS)
    return mn, TwitchPool(mn, fs=FS)


def plateau_slice(n: int):
    """Sample slice of the trapezoid plateau (0.3 s inside each end)."""
    a = TRAP["lead_s"] + TRAP["rise_s"] + 0.3
    b = TRAP["lead_s"] + TRAP["rise_s"] + TRAP["hold_s"] - 0.3
    return slice(int(a * FS), min(int(b * FS), n))


def trapezoid_trial(level: float, mn, tw, W1, Wg=None, seed: int = SEED):
    """One trapezoid contraction: drive (+ common drive) → spikes → EMG (single / grid) → force."""
    from emgforge.activation import drive, compound_emg, compound_emg_multi
    E = drive.trapezoid(level, TRAP["rise_s"], TRAP["hold_s"], TRAP["fall_s"], fs=FS, lead_s=TRAP["lead_s"])
    E = drive.add_common_drive(E, sigma=COMMON_DRIVE["sigma"], cutoff_hz=COMMON_DRIVE["cutoff_hz"], fs=FS, seed=seed)
    sp = mn.spike_trains(E, seed=seed)
    out = dict(level=level, drive=E, spikes=sp,
               emg_single=compound_emg(sp, W1, n_samples=len(E)),
               force=tw.force(sp, n_samples=len(E)))
    if Wg is not None:                                   # grid EMG from the units the tensor covers
        out["emg_grid"] = compound_emg_multi(sp[:Wg.shape[0]], Wg, n_samples=len(E))
        out["n_grid_mus"] = int(Wg.shape[0])
    return out


def dynamic_trial(mn, W1, seed: int = SEED):
    """Angle-modulated (non-stationary) trial: angle → drive, MUAP amplitude + time-warp."""
    from emgforge.activation import drive
    from emgforge.activation.dynamic import (angle_track, drive_from_angle, amp_from_angle,
                                             warp_from_angle, dynamic_compound_emg)
    angle = angle_track(DYN["duration_s"], fs=FS, cycles=DYN["cycles"])
    E = drive.add_common_drive(drive_from_angle(angle), sigma=DYN["drive_sigma"], fs=FS, seed=DYN["drive_seed"])
    amp = amp_from_angle(angle, gain=DYN["amp_gain"]); warp = warp_from_angle(angle, gain=DYN["warp_gain"])
    sp = mn.spike_trains(E, seed=seed)
    emg = dynamic_compound_emg(sp, W1, amp, warp, n_samples=len(E))
    return dict(angle=angle, drive=E, spikes=sp, emg=emg, amp=amp, warp=warp)


def pad_spikes(trains, pad: int = -1):
    """List of int arrays → (n_mu, max_spikes) int32 padded with ``pad`` + (n_mu,) counts."""
    n = max((len(s) for s in trains), default=0)
    out = np.full((len(trains), max(n, 1)), pad, dtype=np.int32)
    for i, s in enumerate(trains):
        out[i, :len(s)] = s
    return out, np.array([len(s) for s in trains], dtype=np.int32)


# --------------------------------------------------------------------------- key numbers
def update_key_numbers(section: str, values: dict):
    """Merge one section into paper/figures/key_numbers_f2.json (JSON-safe)."""
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in np.asarray(v).tolist()] if isinstance(v, np.ndarray) else [clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    data = {}
    if KEY_JSON.exists():
        with open(KEY_JSON) as fh:
            data = json.load(fh)
    data[section] = clean(values)
    with open(KEY_JSON, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"key numbers [{section}] → {KEY_JSON.name}")


__all__ = [n for n in dir() if not n.startswith("_")]
