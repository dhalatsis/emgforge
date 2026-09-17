#!/usr/bin/env python
"""One command from a labelled MRI segmentation to interference EMG on an HD-EMG grid.

    python scripts/run_pipeline.py                                     # FCU, 5×5 @ 10 mm, 20 MUs
    python scripts/run_pipeline.py --n-mu 100 --workers 2 --out _results/pipeline/fcu_100

Cold by construction: the mesh, the fibre config, the lead fields, the MUAPs and the EMG
are all rebuilt from the segmentation into ``--out``. ``--cache`` reuses the mesh and the
lead-field bank already in ``--out`` (never anything another script left under
``_results``).

Stages, each timed (``<out>/pipeline_timings.json``; ``--key-json`` merges the record into
a paper key-numbers file, ``--table`` renders the booktabs stage table):

  mesh → fibre directions → muscle geometry → fibre bed → motor-unit pool → volume
  conductor → electrode grid → lead fields → lead-field conditioning → MUAPs → activation

Output ``<out>/pipeline_output.npz``: electrodes, bed, pool, lead-field bank, MUAP tensor
and the drive / EMG / force / spikes of the trapezoid trials. ``Simulator.from_pipeline``
reloads it. Every stage is a call into ``emgforge.mri.pipeline`` (the package), nothing
here computes physics.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- bookkeeping
def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


class Logger:
    def __init__(self, path: Path):
        self.t0 = time.time()
        self.fh = open(path, "a")

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:7.1f} s] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n"); self.fh.flush()


class Stage:
    def __init__(self, key, name, what):
        self.key, self.name, self.what = key, name, what
        self.io = ""
        self.wall_s = None
        self.sizes: dict = {}
        self.notes: dict = {}
        self.cached = False

    def record(self) -> dict:
        return dict(key=self.key, name=self.name, what=self.what, io=self.io,
                    wall_s=round(float(self.wall_s), 3), cached=self.cached,
                    sizes=jsonable(self.sizes), notes=jsonable(self.notes))


class Timeline:
    def __init__(self, log):
        self.log = log
        self.stages: list[Stage] = []

    @contextmanager
    def stage(self, key, name, what):
        s = Stage(key, name, what)
        self.log(f"[{len(self.stages) + 1:2d}] {name} ...")
        t0 = time.time()
        yield s
        s.wall_s = time.time() - t0
        self.stages.append(s)
        self.log(f"     {name}: {s.io}  [{s.wall_s:.1f} s{' (cached)' if s.cached else ''}]")

    def total(self) -> float:
        return float(sum(s.wall_s for s in self.stages))


def environment(workers: int) -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 1e6, 1); break
    except OSError:
        pass
    vers = {}
    for mod in ("numpy", "scipy", "dolfinx", "petsc4py", "pytetwild", "nibabel", "trimesh", "skimage"):
        try:
            vers[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:                                   # noqa: BLE001
            vers[mod] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=False).stdout.strip()
    except OSError:
        commit = None
    import os
    return dict(host=platform.node(), cpu=cpu, n_cpu=os.cpu_count(), mem_gb=mem_gb,
                python=platform.python_version(), workers=int(workers), versions=vers, git_commit=commit)


# --------------------------------------------------------------------------- sanity read-offs
def xcorr_lag_ms(a: np.ndarray, b: np.ndarray, dt_ms: float) -> float:
    """Lag of ``b`` relative to ``a`` (positive = ``b`` later) from the cross-correlation peak
    with parabolic sub-sample interpolation."""
    c = np.correlate(b, a, mode="full")
    L = int(np.argmax(c))
    delta = 0.0
    if 0 < L < len(c) - 1:
        y0, y1, y2 = c[L - 1], c[L], c[L + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float((L + delta - (len(a) - 1)) * dt_ms)


def cv_readoff(W: np.ndarray, t_ms: np.ndarray, M: int, N: int, ied_along_mm: float, v_set: float,
               bed_paths: np.ndarray, elec: np.ndarray, min_p2p_V: float = 5e-6) -> dict:
    """Conduction velocity read off the grid column through the muscle direction: the
    cross-correlation lag between adjacent rows (IED apart along the arm) of each unit's
    MUAP → v = IED / lag, on the monopolar signals and on the single- and
    double-differential montages along the column (which cancel the non-propagating
    end-of-fibre component that biases the monopolar lag toward zero).

    The rows are spaced along the arm; on a fibre at angle α to that axis the detection
    points are IED·cos α apart, so the apparent velocity is v/cos α. ``cos_alpha`` is the
    mean z-component of the fibre tangents under the grid; ``*_corrected`` multiplies by it.
    """
    j0 = (N - 1) // 2
    dt = float(t_ms[1] - t_ms[0])
    if M < 3:
        return dict(column=j0, note="needs >= 3 rows")
    zlo, zhi = float(elec[:, :, 2].min()), float(elec[:, :, 2].max())
    paths = np.asarray(bed_paths, dtype=float)
    tan = np.gradient(paths, axis=1)
    tan /= np.maximum(np.linalg.norm(tan, axis=2, keepdims=True), 1e-12)
    under = (paths[:, :, 2] >= zlo) & (paths[:, :, 2] <= zhi)
    cos_a = float(np.abs(tan[:, :, 2][under]).mean()) if under.any() else 1.0
    montages = {"monopolar": lambda c: c,
                "single_differential": lambda c: -np.diff(c, axis=0),
                "double_differential": lambda c: np.diff(c, n=2, axis=0)}
    out = dict(column=j0, ied_along_mm=float(ied_along_mm), v_set_m_s=float(v_set),
               expected_row_lag_ms=float(ied_along_mm / v_set), sample_ms=dt,
               cos_alpha=cos_a, fibre_angle_under_grid_deg=float(np.degrees(np.arccos(min(cos_a, 1.0)))),
               montages={})
    for name, fn in montages.items():
        v_mu, used = [], []
        for k in range(W.shape[0]):
            col = fn(W[k].reshape(M, N, -1)[:, j0])
            if np.ptp(col, axis=1).max() < min_p2p_V:
                continue
            lag = float(np.median([xcorr_lag_ms(col[i], col[i + 1], dt) for i in range(len(col) - 1)]))
            if lag > 0:
                v_mu.append(ied_along_mm / lag)        # mm/ms == m/s
                used.append(int(k))
        v_mu = np.array(v_mu)
        summed = fn(W.sum(0).reshape(M, N, -1)[:, j0])
        lag_sum = float(np.median([xcorr_lag_ms(summed[i], summed[i + 1], dt) for i in range(len(summed) - 1)]))
        med = float(np.median(v_mu)) if len(v_mu) else None
        out["montages"][name] = dict(
            n_mu_used=int(len(v_mu)), mu_used=used, v_median_m_s=med,
            v_median_corrected_m_s=(med * cos_a) if med is not None else None,
            v_iqr_m_s=[float(np.percentile(v_mu, 25)), float(np.percentile(v_mu, 75))] if len(v_mu) else None,
            v_min_max_m_s=[float(v_mu.min()), float(v_mu.max())] if len(v_mu) else None,
            v_from_summed_muap_m_s=float(ied_along_mm / lag_sum) if lag_sum > 0 else None)
    sd = out["montages"]["single_differential"]
    out.update(v_median_m_s=sd["v_median_m_s"], v_median_corrected_m_s=sd["v_median_corrected_m_s"],
               v_iqr_m_s=sd["v_iqr_m_s"], n_mu_used=sd["n_mu_used"], headline_montage="single_differential")
    return out


# --------------------------------------------------------------------------- LaTeX table
_TEX_REPL = [("→", r"$\rightarrow$"), ("×", r"$\times$"), ("µ", r"$\mu$"), ("σ", r"$\sigma$"),
             ("φ", r"$\varphi$"), ("±", r"$\pm$"), ("≈", r"$\approx$"), ("−", "$-$"), ("—", "---"),
             ("–", "--"), ("²", r"$^2$"), ("·", r"$\cdot$"), ("∗", "$*$"), ("%", r"\%"), ("_", r"\_"),
             ("&", r"\&"), ("#", r"\#")]


def tex(s: str) -> str:
    for a, b in _TEX_REPL:
        s = s.replace(a, b)
    return re.sub(r"(\d),(\d{3})", r"\1\\,\2", re.sub(r"(\d),(\d{3})", r"\1\\,\2", s))


def write_table(path: Path, run: dict):
    env = run["environment"]
    lines = [
        "% Auto-generated by scripts/run_pipeline.py -- do not edit by hand.",
        f"% command: {run['command']}",
        f"% {env['host']} ({env['cpu']}, {env['workers']} worker processes); sum of stages "
        f"{run['totals']['stages_wall_s']:.0f} s, whole command {run['totals']['command_wall_s']:.0f} s.",
        "% Suggested caption: One cold run of the pipeline from the committed forearm segmentation to",
        "% interference EMG on the grid: what each stage does, its input and output sizes and its wall time.",
        r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{5.6cm} >{\raggedright\arraybackslash}p{3.7cm} r@{}}",
        r"\toprule",
        r"Stage & What it does & Input $\rightarrow$ output & Wall (s) \\",
        r"\midrule",
    ]
    for s in run["stages"]:
        lines.append(f"{tex(s['name'])} & {tex(s['what'])} & {tex(s['io'])} & {s['wall_s']:.1f} \\\\")
    lines += [r"\midrule",
              f"Total & sum of the stages above & & {run['totals']['stages_wall_s']:.0f} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def merge_key_json(path: Path, section: str, run: dict):
    data = {}
    if path.exists():
        with open(path) as fh:
            data = json.load(fh)
    data[section] = run
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    from emgforge.mri import pipeline as P

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("what to simulate")
    g.add_argument("--seg", default=str(P.DEFAULT_SEG),
                   help="labelled NIfTI segmentation (default: the committed WR forearm)")
    g.add_argument("--muscle", type=int, default=P.FCU,
                   help="segmentation label of the muscle under the grid (default 8 = flexor carpi ulnaris)")
    g.add_argument("--grid", default="5x5", help="electrode grid MxN: M rows along the arm, N columns around it")
    g.add_argument("--ied", type=float, default=10.0, help="inter-electrode distance (mm)")
    g.add_argument("--n-mu", type=int, default=20, help="motor units in the pool")
    g.add_argument("--fs", type=float, default=2048.0, help="sampling rate (Hz)")
    g.add_argument("--workers", type=int, default=2, help="worker processes for the conditioning and MUAP stages")
    g.add_argument("--out", default=None,
                   help="output directory (default _results/pipeline/L<muscle>_<grid>_ied<ied>_mu<n>)")
    g.add_argument("--seed", type=int, default=0, help="bed / pool / drive / spike seed")
    g.add_argument("--cache", action="store_true",
                   help="reuse the mesh and lead-field bank found in --out (default: rebuild everything)")
    g.add_argument("--no-verify", action="store_true",
                   help="skip the full-recipe vs factored-recipe check on one unit")
    g.add_argument("--no-leadfields", action="store_true", help="do not store the raw lead-field bank in the .npz")
    g.add_argument("--note", default="", help="free-text note stored with the timings (machine load, etc.)")
    a = ap.add_argument_group("anatomy / mesh / volume conductor")
    a.add_argument("--target-z", type=float, default=1.5, help="z voxel spacing after resampling (mm)")
    a.add_argument("--edge-length", type=float, default=0.03,
                   help="fTetWild edge-length factor (0.03 = 47k tets on the WR forearm, 0.02 = 100k)")
    a.add_argument("--surface-faces", type=int, default=50000, help="surface triangles after decimation")
    a.add_argument("--smooth-iters", type=int, default=10, help="Laplacian smoothing iterations")
    a.add_argument("--skin-shell", type=float, default=1.5, help="skin layer re-tagged inside the fat (mm)")
    a.add_argument("--sigma-mode", default="centerline", choices=["constant", "global", "centerline", "morphing"],
                   help="orientation of the muscle conductivity tensor")
    a.add_argument("--source-sigma", type=float, default=5.0, help="Gaussian electrode source width (mm)")
    a.add_argument("--zc-frac", type=float, default=0.5, help="grid centre along the mesh (fraction of its z extent)")
    f = ap.add_argument_group("fibres / units / synthesis")
    f.add_argument("--density", type=float, default=4.0, help="fibre-bed density (fibres / mm^2)")
    f.add_argument("--iz-frac", type=float, default=P.IZ_FRAC_FCU,
                   help="innervation zone as a fraction of fibre length (default: the FCU value of the released pool)")
    f.add_argument("--iz-jitter", type=float, default=P.IZ_JITTER, help="per-fibre IZ scatter (fraction of fibre length)")
    f.add_argument("--cv", type=float, default=4.0, help="conduction velocity (m/s)")
    f.add_argument("--w", type=int, default=256, help="MUAP window (samples)")
    d = ap.add_argument_group("contraction")
    d.add_argument("--levels", default="0.1,0.2,0.35,0.5,0.7,1.0",
                   help="trapezoid plateau drive levels (fraction of maximal drive)")
    d.add_argument("--rise", type=float, default=0.5, help="ramp-up (s)")
    d.add_argument("--hold", type=float, default=2.0, help="plateau (s)")
    d.add_argument("--fall", type=float, default=0.5, help="ramp-down (s)")
    d.add_argument("--lead", type=float, default=0.2, help="silent lead-in (s)")
    d.add_argument("--common-drive", type=float, default=0.015, help="sigma of the shared low-pass drive noise")
    d.add_argument("--drive-cutoff", type=float, default=2.0, help="common-drive low-pass cut-off (Hz)")
    p = ap.add_argument_group("paper outputs")
    p.add_argument("--key-json", default=None, help="merge the run record into this JSON under --key-section")
    p.add_argument("--key-section", default=None, help="section name in --key-json (default: the run tag)")
    p.add_argument("--table", default=None, help="write the booktabs stage table (LaTeX fragment) to this path")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_cmd = time.time()
    from emgforge.activation import drive
    from emgforge.mri import pipeline as P
    from emgforge.simulator import Simulator

    M, N = (int(x) for x in args.grid.lower().replace("×", "x").split("x"))
    E = M * N
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    tag = f"L{args.muscle}_{M}x{N}_ied{args.ied:g}_mu{args.n_mu}"
    out = Path(args.out) if args.out else ROOT / "_results/pipeline" / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "pipeline.log")
    T = Timeline(log)
    seg = Path(args.seg)
    mname = P.muscle_name(args.muscle)
    mesh_path, cfg_path, lf_path = out / "mesh.msh", out / "fibres.json", out / "leadfields.npz"
    argv_disp = sys.argv[1:] if argv is None else list(argv)
    command = "python scripts/run_pipeline.py " + " ".join(argv_disp)
    log(f"{command}")
    log(f"segmentation {seg.name}, muscle {args.muscle} = {mname}, grid {M}x{N} @ {args.ied:g} mm, "
        f"{args.n_mu} MUs, fs {args.fs:g} Hz, {args.workers} workers → {out}")

    import nibabel as nib
    img = nib.load(str(seg))
    seg_shape = [int(x) for x in img.shape]
    seg_vox = [float(x) for x in img.header.get_zooms()[:3]]
    del img

    # ---- 1. mesh ----------------------------------------------------------------
    with T.stage("mesh", "Mesh",
                 "resample the labels, marching cubes on the tissue mask, smooth and decimate the surface, "
                 "fTetWild tetrahedra, tissue tag per cell from the label under its centroid") as s:
        meta_path = mesh_path.with_name(mesh_path.stem + "_metadata.json")
        if args.cache and mesh_path.exists() and meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            mesh_info = dict(n_vertices=meta["n_vertices"], n_tets=meta["n_tetrahedra"],
                             tissue_tets=meta["tissue_type_counts"], build_params=meta.get("build_params", {}),
                             seg_shape_resampled=meta["segmentation_shape"], mesh=str(mesh_path))
            s.cached = True
        else:
            mesh_info = P.build_mesh(seg, mesh_path, target_z=args.target_z, edge_length=args.edge_length,
                                     surface_faces=args.surface_faces, smooth_iters=args.smooth_iters)
        s.sizes = dict(seg_shape=seg_shape, voxel_mm=seg_vox, **{k: v for k, v in mesh_info.items() if k != "mesh"})
        s.io = (f"{seg_shape[0]}×{seg_shape[1]}×{seg_shape[2]} voxels → "
                f"{mesh_info['n_tets']:,} tetrahedra, {mesh_info['n_vertices']:,} nodes")

    # ---- 2. fibre directions --------------------------------------------------------
    with T.stage("fibre_config", "Fibre directions",
                 "PCA fibre direction and smoothed per-slice centreline of every muscle label → "
                 "the fibre-aligned (5:1) conductivity tensor of each muscle") as s:
        fm, fc_info = P.build_fibre_config(seg, cfg_path)
        s.sizes = fc_info
        s.io = (f"{fc_info['n_labels']} labels → {fc_info['n_muscles']} muscles, "
                f"{fc_info['n_centerlines']} centrelines")

    # ---- 3. muscle geometry ---------------------------------------------------------
    with T.stage("geometry", "Muscle geometry",
                 "ray-cast boundary radius of each muscle cross-section per slice → the morphing-disk "
                 "frame that carries a fibre through the muscle") as s:
        fm.estimate_cross_sections()
        mus = fm.muscles[args.muscle]
        n_cs = sum(m.cross_section is not None for m in fm.muscles.values())
        s.sizes = dict(n_cross_sections=n_cs, muscle=mname, label=args.muscle,
                       z_span_mm=float(mus.z_span_mm), fibre_angle_from_z_deg=float(mus.fiber_angle_from_z_deg),
                       n_theta=int(mus.cross_section.n_theta) if hasattr(mus.cross_section, "n_theta") else None)
        s.io = f"{fc_info['n_centerlines']} centrelines → {n_cs} cross-section models"

    # ---- 4. fibre bed ---------------------------------------------------------------
    with T.stage("bed", "Fibre bed",
                 "Poisson-disk sampling of the muscle cross-section at the set density; one curved "
                 "morphing-disk path per fibre from tendon to tendon") as s:
        bed = P.poisson_bed(fm, args.muscle, density=args.density, seed=args.seed)
        arc_dz, L_fib = P.bed_arc_geometry(bed)
        n_fib, n_z = bed.paths.shape[:2]
        s.sizes = dict(n_fibres=int(n_fib), n_points_per_fibre=int(n_z), density_per_mm2=args.density,
                       area_mm2=float(bed.cross_section_area_mm2), fibre_length_mm_mean=float(L_fib.mean()),
                       fibre_length_mm_range=[float(L_fib.min()), float(L_fib.max())],
                       arc_dz_mm_mean=float(arc_dz.mean()), centroid_xy=bed.centroid_xy.tolist())
        s.io = (f"{mname}, {bed.cross_section_area_mm2:.0f} mm² → {n_fib} fibres × {n_z} points "
                f"({L_fib.mean():.0f} mm long)")

    # ---- 5. motor-unit pool ---------------------------------------------------------
    with T.stage("pool", "Motor-unit pool",
                 "exponential Henneman size distribution; each unit's territory grown around a random "
                 "anchor fibre until it holds its fibres; index = size rank = recruitment order") as s:
        pool = P.henneman_pool(bed, args.n_mu, seed=args.seed)
        sizes = np.array([m.size for m in pool])
        s.sizes = dict(n_mu=len(pool), sizes_min_median_max=[int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
                       fibres_total=int(sizes.sum()), n_fibres_used=int(len(np.unique(np.concatenate([m.fiber_idxs for m in pool])))),
                       territory_radius_mm_range=[float(min(m.territory_radius_mm for m in pool)),
                                                  float(max(m.territory_radius_mm for m in pool))])
        s.io = f"{n_fib} fibres → {len(pool)} units of {sizes.min()}–{sizes.max()} fibres"

    # ---- 6. volume conductor --------------------------------------------------------
    with T.stage("fem_build", "Volume conductor",
                 "conductivity tensor per cell from its tissue tag (muscle tensors rotated onto the "
                 "centreline tangent), a skin shell inside the fat, FEniCSx spaces and search trees") as s:
        fem = P.build_volume_conductor(mesh_path, cfg_path, seg, skin_shell_mm=args.skin_shell,
                                       sigma_mode=args.sigma_mode)
        n_skin = int(fem._is_skin_cell.sum()) if fem._is_skin_cell is not None else 0
        s.sizes = dict(n_cells=int(fem.n_cells), n_vertices=int(fem.n_vertices), n_dofs=int(fem.n_dofs),
                       n_skin_cells=n_skin, skin_shell_mm=args.skin_shell, sigma_mode=args.sigma_mode)
        s.io = f"{fem.n_cells:,} cells → {fem.n_cells:,} σ tensors, {fem.n_dofs:,} dofs"

    # ---- 7. electrode grid ----------------------------------------------------------
    with T.stage("electrodes", "Electrode grid",
                 "M×N grid ray-cast onto the skin surface over the muscle: rows IED apart along the "
                 "arm, columns IED apart along the skin arc, centred on the limb-axis → muscle ray") as s:
        elec, ginfo = P.grid_electrodes(fem, fm, bed, m=M, n=N, ied_mm=args.ied, zc_frac=args.zc_frac)
        s.sizes = dict(M=M, N=N, n_electrodes=E, ied_mm=args.ied, **ginfo)
        s.io = (f"{ginfo['n_skin_triangles']:,} skin triangles → {E} electrodes "
                f"({M}×{N}, {ginfo['ied_along_mm']:.1f}×{ginfo['ied_across_mm']:.1f} mm)")

    # ---- 8. lead fields -------------------------------------------------------------
    cfg = P.production_config(fs=args.fs, v=args.cv, w=args.w)
    cfg_n_poles = int(cfg.denoise_n_poles)
    with T.stage("leadfields", "Lead fields",
                 "one reciprocity solve per electrode (zero-mean Gaussian source at the skin point, "
                 "GMRES/ILU); φ sampled along every fibre of the bed") as s:
        lf, phi_c_cached = None, None
        if args.cache and lf_path.exists():
            d = np.load(lf_path)
            if d["phi_grid"].shape == (M, N, n_fib, n_z) and np.allclose(d["elec_xyz"], elec):
                lf = dict(phi_grid=d["phi_grid"].astype(float), locate_s=float(d["locate_s"]),
                          solve_s=d["solve_s"].tolist(), sample_s=d["sample_s"].tolist())
                s.cached = True
                if "phi_cond" in d.files and int(d["n_poles"]) == cfg_n_poles:
                    phi_c_cached = (d["phi_cond"].astype(float), d["cond_secs"].astype(float))
        if lf is None:
            lf = P.solve_grid_leadfields(fem, elec, bed.paths, source_sigma=args.source_sigma, log=log)
            np.savez_compressed(lf_path, phi_grid=lf["phi_grid"], elec_xyz=elec, locate_s=lf["locate_s"],
                                solve_s=np.array(lf["solve_s"]), sample_s=np.array(lf["sample_s"]))
        phi = lf["phi_grid"].reshape(E, n_fib, n_z)
        s.sizes = dict(n_solves=E, phi_shape=[E, n_fib, n_z], locate_s=lf["locate_s"],
                       solve_s_per_electrode=float(np.mean(lf["solve_s"])),
                       sample_s_per_electrode=float(np.mean(lf["sample_s"])), source_sigma_mm=args.source_sigma,
                       phi_peak_mV_median=float(np.median(np.abs(phi).max(axis=2)) * 1e3))
        s.io = f"{E} solves → φ bank {E}×{n_fib}×{n_z}"
    del fem
    gc.collect()

    # ---- 9. lead-field conditioning ---------------------------------------------------
    with T.stage("condition", "Lead-field conditioning",
                 "the recipe's denoising: a free-position 3-monopole fit of each φ(z) that removes "
                 "mesh ripple before the second derivative — once per electrode–fibre pair") as s:
        if phi_c_cached is not None:
            phi_c, cond_secs = phi_c_cached
            s.cached = True
        else:
            phi_c, cond_secs = P.condition_leadfields(phi, arc_dz, n_poles=cfg_n_poles,
                                                      workers=args.workers, log=log)
            d = dict(np.load(lf_path))
            d.update(phi_cond=phi_c.reshape(M, N, n_fib, n_z), cond_secs=cond_secs, n_poles=cfg_n_poles)
            np.savez_compressed(lf_path, **d)
        s.sizes = dict(n_fits=int(E * n_fib), n_poles=cfg.denoise_n_poles, cpu_s=float(cond_secs.sum()),
                       ms_per_fit=float(cond_secs.sum() / (E * n_fib) * 1e3))
        s.io = f"{E}×{n_fib} φ(z) → {E * n_fib:,} monopole fits"

    # ---- 10. MUAPs --------------------------------------------------------------------
    with T.stage("muaps", "MUAPs",
                 "per unit and electrode: line-source integral of the current-source density of a "
                 "Rosenfalck action potential (both directions from the NMJ, one-sided tendon window) "
                 "against φ over the unit's fibres; physical time, t = 0 at the NMJ") as s:
        W, t_ms, mu_secs = P.muap_tensor(phi_c, bed, pool, cfg, iz_frac=args.iz_frac, iz_jitter=args.iz_jitter,
                                         v=args.cv, conditioned=True, workers=args.workers, log=log)
        n_sfap = int(sizes.sum() * E)
        s.sizes = dict(tensor_shape=list(W.shape), n_sfaps=n_sfap, cpu_s=float(mu_secs.sum()),
                       ms_per_sfap=float(mu_secs.sum() / n_sfap * 1e3), s_per_mu_min_max=[float(mu_secs.min()), float(mu_secs.max())],
                       t_ms_range=[float(t_ms[0]), float(t_ms[-1])], config=P.config_dict(cfg),
                       iz_frac=args.iz_frac, iz_jitter=args.iz_jitter, cv_m_s=args.cv)
        if not args.no_verify:
            k = int(np.argmin(sizes))
            chk = P.verify_conditioning(phi, phi_c, bed, pool, cfg, k, iz_frac=args.iz_frac,
                                        iz_jitter=args.iz_jitter, v=args.cv)
            s.notes["recipe_check"] = chk
            log(f"  recipe check on MU {k} ({chk['size']} fibres): full recipe vs factored path "
                f"max|Δ| = {chk['max_abs_diff_V']:.3g} V on {chk['p2p_max_V'] * 1e6:.1f} µV — "
                f"{'identical' if chk['identical'] else 'DIFFERENT'}")
        s.io = f"φ + {len(pool)} units → MUAP tensor {W.shape[0]}×{E}×{args.w} ({n_sfap:,} SFAPs)"

    # ---- 11. activation ------------------------------------------------------------
    with T.stage("activation", "Activation",
                 "motoneuron pool (recruitment thresholds, onion-skin rate coding, renewal ISIs) → "
                 "spikes; twitch model → force; spikes convolved with the MUAPs → EMG on the grid, "
                 "one trapezoid per drive level") as s:
        sim = Simulator(W, fs=args.fs, grid=(M, N))
        recs = []
        for lv in levels:
            Ed = drive.trapezoid(lv, args.rise, args.hold, args.fall, fs=args.fs, lead_s=args.lead)
            Ed = drive.add_common_drive(Ed, sigma=args.common_drive, cutoff_hz=args.drive_cutoff, fs=args.fs, seed=args.seed)
            recs.append(sim.run(Ed, seed=args.seed))
        Tn = len(recs[0].drive)
        s.sizes = dict(n_levels=len(levels), levels=levels, n_samples=Tn, duration_s=Tn / args.fs, fs=args.fs,
                       trapezoid=dict(lead_s=args.lead, rise_s=args.rise, hold_s=args.hold, fall_s=args.fall),
                       common_drive=dict(sigma=args.common_drive, cutoff_hz=args.drive_cutoff),
                       n_active_per_level=[r.n_active for r in recs],
                       n_spikes_per_level=[int(sum(len(sp) for sp in r.spikes)) for r in recs])
        s.io = (f"{len(levels)} trapezoids × {Tn / args.fs:.1f} s → EMG {E}×{Tn} per level, force, "
                f"{sum(s.sizes['n_spikes_per_level']):,} spikes")

    # ---- sanity read-offs ----------------------------------------------------------
    p2p = np.ptp(W, axis=2) * 1e6                                # (n_mu, E) µV
    e0 = (M // 2) * N + N // 2
    per_mu_max = p2p.max(axis=1)
    iz_idx = int(round(args.iz_frac * (n_z - 1)))
    iz_z = float(bed.paths[:, iz_idx, 2].mean())
    a = args.lead + args.rise + 0.3
    b = args.lead + args.rise + args.hold - 0.3
    pl = slice(int(a * args.fs), min(int(b * args.fs), Tn))
    rms_grid = [float(np.sqrt((r.emg[:, pl] ** 2).mean()) * 1e6) for r in recs]
    rms_e0 = [float(np.sqrt((r.emg[e0, pl] ** 2).mean()) * 1e6) for r in recs]
    force_pl = [float(r.force[pl].mean() * 100) for r in recs]
    cv = cv_readoff(W, t_ms, M, N, ginfo["ied_along_mm"], args.cv, bed.paths, elec)

    def _dur_ms(w):                                            # span above 10 % of |peak|
        m = np.abs(w).max()
        return float(np.ptp(t_ms[np.abs(w) > 0.1 * m])) if m > 0 else 0.0

    sanity = dict(
        muap_p2p_uV=dict(per_mu_max_over_grid=dict(min=float(per_mu_max.min()), median=float(np.median(per_mu_max)),
                                                   max=float(per_mu_max.max())),
                         all_pairs_median=float(np.median(p2p)), centre_electrode_median=float(np.median(p2p[:, e0])),
                         centre_electrode_index=int(e0)),
        muap_duration_ms_median=float(np.median([_dur_ms(W[k, e0]) for k in range(W.shape[0])])),
        iz_plane_z_mm=iz_z, grid_z_range_mm=[float(elec[:, :, 2].min()), float(elec[:, :, 2].max())],
        cv_readoff=cv,
        emg_vs_drive=dict(levels=levels, plateau_window_s=[pl.start / args.fs, pl.stop / args.fs],
                          rms_grid_mean_uV=rms_grid, rms_centre_uV=rms_e0, force_pct_mvc=force_pl,
                          n_active=[r.n_active for r in recs],
                          rms_monotonic=bool(np.all(np.diff(rms_grid) > 0)),
                          force_monotonic=bool(np.all(np.diff(force_pl) > 0))),
    )
    log(f"sanity: MUAP p2p on the grid — per-unit max {per_mu_max.min():.1f}–{per_mu_max.max():.1f} µV "
        f"(median {np.median(per_mu_max):.1f}); centre electrode median {np.median(p2p[:, e0]):.1f} µV")
    mont = cv.get("montages", {})
    log(f"sanity: IZ plane z = {iz_z:.0f} mm, grid z {elec[:, :, 2].min():.0f}–{elec[:, :, 2].max():.0f} mm; "
        f"CV read-off on column {cv.get('column')} vs set {args.cv} m/s: "
        + ", ".join(f"{k} {v['v_median_m_s']:.2f} (IQR {v['v_iqr_m_s'][0]:.2f}–{v['v_iqr_m_s'][1]:.2f}, n={v['n_mu_used']})"
                    for k, v in mont.items() if v.get("v_median_m_s") is not None)
        + f"; fibre obliquity under the grid cos α = {cv.get('cos_alpha', 1.0):.3f} "
          f"(single-differential corrected {cv.get('v_median_corrected_m_s')})")
    log(f"sanity: plateau EMG RMS (grid mean, µV) {np.round(rms_grid, 2).tolist()} at drive {levels}; "
        f"force %MVC {np.round(force_pl, 1).tolist()}; active MUs {[r.n_active for r in recs]}")

    # ---- save ----------------------------------------------------------------------
    f32 = lambda x: np.asarray(x, dtype=np.float32)          # noqa: E731
    fibre_idx = np.full((len(pool), int(sizes.max())), -1, dtype=np.int32)
    for k, m in enumerate(pool):
        fibre_idx[k, :m.size] = m.fiber_idxs
    max_sp = max(max((len(sp) for sp in r.spikes), default=0) for r in recs)
    spikes = np.full((len(levels), len(pool), max(max_sp, 1)), -1, dtype=np.int32)
    counts = np.zeros((len(levels), len(pool)), dtype=np.int32)
    for i, r in enumerate(recs):
        for k, sp in enumerate(r.spikes):
            spikes[i, k, :len(sp)] = sp; counts[i, k] = len(sp)
    arrays = dict(
        grid_shape=np.array([M, N], dtype=np.int32), ied_mm=np.float32(args.ied), fs=np.float32(args.fs),
        muscle_label=np.int32(args.muscle), elec_xyz=f32(elec), elec_xyz_flat=f32(elec.reshape(-1, 3)),
        bed_paths=f32(bed.paths), bed_xy_mid=f32(bed.xy_mid), bed_arc_dz_mm=f32(arc_dz), bed_length_mm=f32(L_fib),
        mu_sizes=sizes.astype(np.int32), mu_fibre_idx=fibre_idx, mu_centre_xy=f32([m.centre_xy for m in pool]),
        mu_territory_radius_mm=f32([m.territory_radius_mm for m in pool]),
        mu_recruitment_threshold_excitation=f32(sim.pool.rte),
        muap_grid=f32(W * 1e6), t_ms=f32(t_ms),
        levels=f32(levels), t_s=f32(np.arange(Tn) / args.fs), drive=f32([r.drive for r in recs]),
        emg_grid=f32([r.emg * 1e6 for r in recs]), force=f32([r.force * 100 for r in recs]),
        spikes=spikes, spike_counts=counts,
    )
    if not args.no_leadfields:
        arrays["phi_grid"] = f32(phi)
    npz_path = out / "pipeline_output.npz"
    np.savez_compressed(npz_path, **arrays)
    units = dict(elec_xyz="mm (segmentation voxel frame)", bed_paths="mm", muap_grid="µV, (n_mu, E, w), e = row*N + col",
                 t_ms="ms, t = 0 at the NMJ discharge", phi_grid="V per unit source, (E, n_fibres, n_points)",
                 drive="fraction of maximal drive", emg_grid="µV, (levels, E, T)", force="%MVC", spikes="sample index, -1 = padding")

    run = dict(
        tag=tag, command=command, note=args.note, muscle=dict(label=args.muscle, name=mname),
        args=jsonable(vars(args)), environment=environment(args.workers),
        sizes=dict(n_voxels=int(np.prod(seg_shape)), seg_shape=seg_shape, n_cells=mesh_info["n_tets"], n_vertices=mesh_info["n_vertices"],
                   n_electrodes=E, grid=[M, N], n_fibres=int(n_fib), n_points_per_fibre=int(n_z), n_mu=len(pool),
                   fibres_in_pool=int(sizes.sum()), n_samples=Tn, n_levels=len(levels), w=args.w, fs=args.fs),
        stages=[s.record() for s in T.stages],
        totals=dict(stages_wall_s=round(T.total(), 1), command_wall_s=round(time.time() - t_cmd, 1),
                    cpu_s_conditioning=float(cond_secs.sum()), cpu_s_muaps=float(mu_secs.sum())),
        sanity=jsonable(sanity),
        outputs=dict(npz=str(npz_path.relative_to(ROOT)) if npz_path.is_relative_to(ROOT) else str(npz_path),
                     mesh=str(mesh_path), fibre_config=str(cfg_path), leadfields=str(lf_path),
                     npz_arrays={k: list(np.shape(v)) for k, v in arrays.items()}, units=units,
                     size_MB=round(npz_path.stat().st_size / 1e6, 2)),
    )
    with open(out / "pipeline_timings.json", "w") as fh:
        json.dump(run, fh, indent=2)
    if args.key_json:
        merge_key_json(Path(args.key_json), args.key_section or tag, run)
        log(f"key numbers → {args.key_json} [{args.key_section or tag}]")
    if args.table:
        write_table(Path(args.table), run)
        log(f"stage table → {args.table}")
    log(f"done: stages {T.total():.0f} s, command {time.time() - t_cmd:.0f} s → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
