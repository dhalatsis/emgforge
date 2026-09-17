#!/usr/bin/env python
"""Mesh-convergence study: what the FEM lead field, the direct-recipe SFAP/MUAP and their
derived read-offs do as the mesh is refined — on the validation cylinder (analytical oracle)
and on the MRI forearm (no oracle: successive-level convergence only).

    P=/home/dc23/miniconda3/envs/fenicsx-env/bin/python
    $P scripts/validation/mesh_convergence.py                 # everything (~2 h)
    $P scripts/validation/mesh_convergence.py --part cyl      # part 1 only
    $P scripts/validation/mesh_convergence.py --part forearm  # part 2 only
    $P scripts/validation/mesh_convergence.py --part analyse  # metrics + figure from the cache

Every mesh level runs in its own subprocess (one finite-element build/solve at a time;
peak RSS is the child's ``ru_maxrss``) and writes ONE file under
``_results/validation/mesh_convergence/`` — a crash resumes from the last complete level.
The analysis reads those files and writes ``key_numbers.json`` plus
``docs/validation/figures/mesh_convergence.{png,pdf}``.

Part 1 — validation cylinder (radii 10/35/38/40, Farina-2004 conductivities, the geometry of
``scripts/validation/cyl_fem.py``). Levels are gmsh ``Mesh.CharacteristicLengthFactor``
values; the validation cache was built at 0.3 (``cyl_fem._model``: ``char_length=0.3``,
NOT the builder's 0.2 default), so 0.3 is "current". One local variant refines the
skin/fat shell only (``size_min_factor`` 0.1 → 0.075 at the current global factor).
Per mesh and electrode-source width σ_s ∈ {5, 1} mm: one reciprocal solve, φ sampled along
fibre lines at 7/10/13/16/20/25 mm below the skin, a transverse profile at z = z_e.

Part 2 — WR forearm, FCU, 5×5 grid @ 10 mm, the production recipe
(``emgforge.mri.pipeline.production_config``). Levels are fTetWild ``edge_length`` values
(0.03 = current). The fibre bed and the pool are built from the segmentation with the
same seed (mesh-independent; the driver asserts the bed hash matches across levels); the
electrode rays come from the current level and are re-projected onto each level's skin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("EMGFORGE_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "_results/validation/mesh_convergence"
FIG_DIR = ROOT / "docs/validation/figures"
PY = sys.executable

# --------------------------------------------------------------------------- design
CYL_LEVELS = [0.6, 0.42, 0.3, 0.24, 0.2]          # global factor; 0.3 = current
CYL_CURRENT = 0.3
CYL_VARIANT = ("skin075", dict(size_min_factor=0.075))   # the ONE local-refinement variant
CYL_MAX_CELLS = 4.0e6
SIGMAS = (5.0, 1.0)                               # electrode-source widths (mm)
RADII = np.array([33.0, 30.0, 27.0, 25.0, 24.0, 20.0, 15.0])   # fibre radial positions
THETAS = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 180.0])   # fibre angles from the electrode
TRANS_THETA = np.arange(-180.0, 180.01, 1.0)                    # transverse profile at z = z_e (full circle)
RADIAL_R = np.arange(10.5, 39.51, 0.25)                         # radial profile on the electrode meridian
DEPTHS = [7.0, 10.0, 13.0, 16.0, 20.0, 25.0]                    # the study's depth ladder
B5_DEPTHS = [7.0, 10.0, 13.0, 15.0, 20.0, 25.0]                 # tier-B5 depth-law ladder

FA_LEVELS = [0.05, 0.03, 0.02, 0.015, 0.012]      # fTetWild edge_length; 0.03 = current
FA_CURRENT = 0.03
FA_MAX_CELLS = 1.0e6
FA_GRID, FA_IED, FA_NMU, FA_SEED, FA_DENSITY = (5, 5), 10.0, 20, 0, 4.0
FA_WORKERS = 4
RSS_LIMIT_GB = 35.0


def rss_gb() -> float:
    """Peak RSS of this process so far (GB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def free_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    return float("nan")


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
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# =========================================================================== part 1 worker
def cyl_tag(factor: float, variant: str | None) -> str:
    return f"cyl_f{factor:g}" + (f"_{variant}" if variant else "")


def cyl_file(factor: float, variant: str | None) -> Path:
    return OUT / f"{cyl_tag(factor, variant)}.npz"


def mesh_stats(model, radii):
    """Per-tissue cell counts and element sizes (edge of the regular tet with the cell's volume),
    plus the muscle element size at each fibre radius (cells whose centroid is within 1.5 mm)."""
    mesh = model.mesh
    x = mesh.geometry.x
    dm = mesh.geometry.dofmap
    dm = np.asarray(dm.array).reshape(-1, 4) if hasattr(dm, "array") else np.asarray(dm)
    v = x[dm]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v[:, 3] - v[:, 0])) / 6.0
    h = (6.0 * np.sqrt(2.0) * vol) ** (1.0 / 3.0)
    tag = np.zeros(len(vol), dtype=int)
    tag[np.asarray(model.cell_markers.indices)] = np.asarray(model.cell_markers.values)
    names = {1: "cancellous", 2: "cortical", 3: "muscle", 4: "fat", 5: "skin"}
    rc = np.hypot(v[:, :, 0].mean(1), v[:, :, 1].mean(1))
    out = dict(counts={n: int((tag == t).sum()) for t, n in names.items()},
               h_median={n: float(np.median(h[tag == t])) if (tag == t).any() else np.nan for t, n in names.items()},
               h_at_radius={f"{r:g}": float(np.median(h[(tag == 3) & (np.abs(rc - r) < 1.5)])) for r in radii})
    return out


def cyl_worker(factor: float, variant: str | None):
    """Build one cylinder mesh, solve for σ_s = 5 and 1 mm, sample the lines; one .npz."""
    import gmsh
    import cyl_fem
    from emgforge.fem import FEMModel
    from emgforge.fem.conductivity import TissueTable
    from emgforge.meshing import build_one_mesh

    tag = cyl_tag(factor, variant)
    mdir = OUT / "cyl_meshes"
    mdir.mkdir(parents=True, exist_ok=True)
    msh, meta_json = mdir / f"{tag}.msh", mdir / f"{tag}.json"
    geo = cyl_fem._geo()
    kw = dict(mesh_char_length_factor=float(factor), refine_on="skin")
    if variant:
        kw.update(dict(CYL_VARIANT[1]) if variant == CYL_VARIANT[0] else {})
    print(f"[{tag}] building mesh: {kw}", flush=True)
    t0 = time.time()
    gmsh.initialize()
    try:
        meta = build_one_mesh(geo.mesh_params(), out_msh=msh, out_json=meta_json, **kw)
    finally:
        gmsh.finalize()
    t_mesh = time.time() - t0
    rss_mesh = rss_gb()
    cells, nodes = int(meta["num_elems_3d"]), int(meta["num_nodes"])
    print(f"[{tag}] {cells:,} cells, {nodes:,} nodes in {t_mesh:.0f} s (RSS {rss_mesh:.1f} GB)", flush=True)

    t0 = time.time()
    model = FEMModel(str(msh), gdim=3, build_conductivity_map=True, point_source=False,
                     source_sigma=SIGMAS[0], conductivity=TissueTable.analytical())
    t_model = time.time() - t0
    n_dofs = int(model.V_scalar.dofmap.index_map.size_local)
    print(f"[{tag}] model built in {t_model:.0f} s ({n_dofs:,} dofs, RSS {rss_gb():.1f} GB)", flush=True)

    stats = mesh_stats(model, RADII)
    print(f"[{tag}] element sizes (mm): {stats['h_median']}; counts {stats['counts']}", flush=True)
    store = dict(factor=float(factor), variant=str(variant or ""), cells=cells, nodes=nodes, n_dofs=n_dofs,
                 mesh_stats=json.dumps(stats), radial_r=RADIAL_R,
                 t_mesh=t_mesh, t_mesh_generate=float(meta["generated_seconds"]), t_model=t_model,
                 rss_after_mesh_gb=rss_mesh, z_abs=cyl_fem.z_line(), radii=RADII, thetas=THETAS,
                 trans_theta=TRANS_THETA, ze=cyl_fem.ZE, mesh_settings=json.dumps(meta["mesh_settings"]))
    el = geo.electrode_on_skin(0.0, cyl_fem.ZE)
    for s in SIGMAS:
        model.options["source_sigma"] = float(s)
        t0 = time.time()
        uh = model.solve_for_point(el)
        t_solve = time.time() - t0
        t0 = time.time()
        lines = cyl_fem.sample_lines(model, uh, geo, RADII, THETAS)
        trans = np.zeros((len(RADII), len(TRANS_THETA)))
        for i, r in enumerate(RADII):
            xy = np.array([geo.fibre_xy_radial(r, th) for th in TRANS_THETA])
            pts = np.column_stack([xy, np.full(len(TRANS_THETA), cyl_fem.ZE)])
            trans[i] = model.evaluate_solution_at_points(pts, uh=uh)
        pts = np.column_stack([RADIAL_R, np.zeros(len(RADIAL_R)), np.full(len(RADIAL_R), cyl_fem.ZE)])
        radial = model.evaluate_solution_at_points(pts, uh=uh)
        t_sample = time.time() - t0
        k = f"s{s:g}"
        store[f"phi_{k}"] = lines
        store[f"trans_{k}"] = trans
        store[f"radial_{k}"] = radial
        store[f"t_solve_{k}"] = t_solve
        store[f"t_sample_{k}"] = t_sample
        print(f"[{tag}] σ_s={s:g}: solve {t_solve:.0f} s, sample {t_sample:.0f} s, "
              f"φ(r=30, z_e) = {lines[1, 0, np.argmin(np.abs(cyl_fem.z_line() - cyl_fem.ZE))]:.4e}", flush=True)
    store["rss_peak_gb"] = rss_gb()
    np.savez_compressed(cyl_file(factor, variant), **store)
    print(f"[{tag}] done, peak RSS {store['rss_peak_gb']:.1f} GB → {cyl_file(factor, variant)}", flush=True)


# =========================================================================== part 2 worker
def fa_tag(edge: float) -> str:
    return f"forearm_e{edge:g}"


def fa_file(edge: float) -> Path:
    return OUT / f"{fa_tag(edge)}.npz"


FA_COMMON = OUT / "forearm_common.npz"


def _bed_hash(bed) -> str:
    return hashlib.sha1(np.ascontiguousarray(bed.paths, dtype=np.float64).tobytes()).hexdigest()[:16]


def _reproject_electrodes(fem, elec_ref, cxy):
    """Each reference electrode re-hit on THIS mesh's skin along its own ray (limb axis at the
    row's z → electrode direction)."""
    from emgforge.mri import pipeline as P
    tri = P.exterior_triangles(fem)
    out = np.zeros_like(elec_ref)
    for i in range(elec_ref.shape[0]):
        for j in range(elec_ref.shape[1]):
            x, y, z = elec_ref[i, j]
            th = np.arctan2(y - cxy[1], x - cxy[0])
            d = np.array([np.cos(th), np.sin(th), 0.0])
            o = np.array([cxy[0], cxy[1], z])
            t = P._ray_outermost_hit(tri, o, d)
            out[i, j] = o + t * d
    return out


def fa_worker(edge: float):
    """Mesh at ``edge`` → volume conductor → 25 lead fields on the fixed FCU bed → conditioned
    φ → MUAP tensor of the fixed 20-unit pool; one .npz."""
    from emgforge.mri import pipeline as P

    tag = fa_tag(edge)
    ldir = OUT / "forearm_meshes" / f"e{edge:g}"
    ldir.mkdir(parents=True, exist_ok=True)
    seg = P.DEFAULT_SEG
    msh, cfg_json = ldir / "mesh.msh", ldir / "fibres.json"
    M, N = FA_GRID

    print(f"[{tag}] building mesh (target_z 1.5, surface_faces 50000, smooth 10)", flush=True)
    t0 = time.time()
    minfo = P.build_mesh(seg, msh, target_z=1.5, edge_length=float(edge), surface_faces=50000, smooth_iters=10)
    t_mesh = time.time() - t0
    rss_mesh = rss_gb()
    print(f"[{tag}] {minfo['n_tets']:,} tets, {minfo['n_vertices']:,} nodes in {t_mesh:.0f} s "
          f"(RSS {rss_mesh:.1f} GB); tissue tets {minfo['tissue_tets']}", flush=True)

    # mesh-independent: fibre config, bed, pool (same seed as the released run)
    fm, _ = P.build_fibre_config(seg, cfg_json)
    fm.estimate_cross_sections()
    bed = P.poisson_bed(fm, P.FCU, density=FA_DENSITY, seed=FA_SEED)
    pool = P.henneman_pool(bed, FA_NMU, seed=FA_SEED)
    bh = _bed_hash(bed)
    arc_dz, L_fib = P.bed_arc_geometry(bed)
    n_fib, n_z = bed.paths.shape[:2]
    sizes = np.array([m.size for m in pool])
    print(f"[{tag}] bed {n_fib} fibres × {n_z} pts (hash {bh}); pool sizes {sizes.min()}–{sizes.max()}", flush=True)

    t0 = time.time()
    fem = P.build_volume_conductor(msh, cfg_json, seg, skin_shell_mm=1.5, sigma_mode="centerline")
    t_fem = time.time() - t0
    n_skin = int(fem._is_skin_cell.sum()) if fem._is_skin_cell is not None else 0
    print(f"[{tag}] volume conductor in {t_fem:.0f} s: {fem.n_cells:,} cells, {fem.n_dofs:,} dofs, "
          f"{n_skin} skin cells (RSS {rss_gb():.1f} GB)", flush=True)

    if FA_COMMON.exists():
        c = np.load(FA_COMMON)
        if str(c["bed_hash"]) != bh:
            raise RuntimeError(f"bed hash {bh} != reference {c['bed_hash']}: the bed is not mesh-independent")
        elec_ref, cxy = c["elec_ref"], c["limb_centre_xy"]
        ginfo = json.loads(str(c["grid_info"]))
        elec = _reproject_electrodes(fem, elec_ref, cxy)
    else:
        if abs(edge - FA_CURRENT) > 1e-12:
            raise RuntimeError("the reference level must run first (it defines the electrode rays)")
        elec, ginfo = P.grid_electrodes(fem, fm, bed, m=M, n=N, ied_mm=FA_IED, zc_frac=0.5)
        cxy = np.array(ginfo["limb_centre_xy"])
        e0 = (M // 2) * N + N // 2
        # unit depth: mean over its fibres of the closest approach of the fibre path to the centre electrode
        depth = np.array([np.mean([np.linalg.norm(bed.paths[f] - elec.reshape(-1, 3)[e0], axis=1).min()
                                   for f in m.fiber_idxs]) for m in pool])
        np.savez(FA_COMMON, elec_ref=elec, limb_centre_xy=cxy, grid_info=json.dumps(jsonable(ginfo)),
                 bed_hash=bh, mu_sizes=sizes, mu_depth_mm=depth, arc_dz=arc_dz, L_fib=L_fib,
                 mu_fibre_idx=np.array([np.pad(m.fiber_idxs, (0, sizes.max() - m.size), constant_values=-1) for m in pool]),
                 bed_paths=bed.paths.astype(np.float32))
        elec_ref = elec
    disp = np.linalg.norm(elec - elec_ref, axis=2)
    print(f"[{tag}] electrodes re-projected: displacement vs reference median {np.median(disp):.2f}, "
          f"max {disp.max():.2f} mm", flush=True)

    lf = P.solve_grid_leadfields(fem, elec, bed.paths, source_sigma=5.0, log=print)
    rss_fem = rss_gb()
    phi = lf["phi_grid"].reshape(M * N, n_fib, n_z)
    del fem
    import gc; gc.collect()

    cfg = P.production_config()
    t0 = time.time()
    phi_c, cond_secs = P.condition_leadfields(phi, arc_dz, n_poles=int(cfg.denoise_n_poles), workers=FA_WORKERS, log=print)
    t_cond = time.time() - t0
    t0 = time.time()
    W, t_ms, mu_secs = P.muap_tensor(phi_c, bed, pool, cfg, conditioned=True, workers=FA_WORKERS, log=print)
    t_muap = time.time() - t0
    chk = P.verify_conditioning(phi, phi_c, bed, pool, cfg, int(np.argmin(sizes)))
    np.savez_compressed(fa_file(edge), edge=float(edge), cells=int(minfo["n_tets"]), nodes=int(minfo["n_vertices"]),
                        n_dofs=int(minfo["n_vertices"]), n_skin_cells=n_skin, tissue_tets=json.dumps(minfo["tissue_tets"]),
                        t_mesh=t_mesh, t_fem=t_fem, t_locate=lf["locate_s"], t_solve=np.array(lf["solve_s"]),
                        t_sample=np.array(lf["sample_s"]), t_cond=t_cond, t_muap=t_muap,
                        rss_after_mesh_gb=rss_mesh, rss_after_fem_gb=rss_fem, rss_peak_gb=rss_gb(),
                        elec=elec, elec_disp_mm=disp, bed_hash=bh, phi=phi, phi_cond=phi_c, W=W, t_ms=t_ms,
                        mu_sizes=sizes, recipe_check_identical=bool(chk["identical"]),
                        grid_ied_along=float(ginfo["ied_along_mm"]))
    print(f"[{tag}] done: solve {np.sum(lf['solve_s']):.0f} s (25), locate {lf['locate_s']:.0f} s, "
          f"condition {t_cond:.0f} s, MUAPs {t_muap:.0f} s; peak RSS {rss_gb():.1f} GB → {fa_file(edge)}", flush=True)


# =========================================================================== driver
def _run_worker(args: list[str], log_path: Path) -> int:
    with open(log_path, "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        fh.flush()
        p = subprocess.run([PY, str(Path(__file__).resolve()), *args], stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "EMGFORGE_ROOT": str(ROOT), "OMP_NUM_THREADS": "1"})
    return p.returncode


def run_cyl(levels, with_variant=True, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "cyl_levels.log"
    done = {}
    plan = [(f, None) for f in sorted(levels, reverse=True)]
    if with_variant:
        plan.append((CYL_CURRENT, CYL_VARIANT[0]))
    skipped = []
    for factor, variant in plan:
        f = cyl_file(factor, variant)
        if f.exists() and not force:
            d = np.load(f)
            done[(factor, variant)] = (int(d["cells"]), float(d["rss_peak_gb"]))
            print(f"[cyl] {cyl_tag(factor, variant)}: cached ({int(d['cells']):,} cells)", flush=True)
            continue
        # guards: predicted cells from the closest finished global level; RSS of the previous level
        pred = None
        glob = {k[0]: v for k, v in done.items() if k[1] is None}
        if glob:
            f_ref = min(glob, key=lambda x: abs(np.log(x / factor)))
            pred = glob[f_ref][0] * (f_ref / factor) ** 3
            if variant:
                pred *= 0.65 * (0.1 / CYL_VARIANT[1]["size_min_factor"]) ** 3 + 0.35
        last_rss = max((v[1] for v in done.values()), default=0.0)
        why = None
        if pred is not None and pred > CYL_MAX_CELLS:
            why = f"predicted {pred / 1e6:.1f} M cells > {CYL_MAX_CELLS / 1e6:.0f} M"
        elif last_rss > RSS_LIMIT_GB:
            why = f"previous level peaked at {last_rss:.0f} GB > {RSS_LIMIT_GB:.0f} GB"
        elif free_gb() < 12:
            why = f"only {free_gb():.0f} GB available"
        if why:
            print(f"[cyl] SKIP {cyl_tag(factor, variant)}: {why}", flush=True)
            skipped.append(dict(level=cyl_tag(factor, variant), why=why))
            continue
        print(f"[cyl] {cyl_tag(factor, variant)}: running (free {free_gb():.0f} GB"
              + (f", predicted {pred / 1e6:.2f} M cells" if pred else "") + ")", flush=True)
        args = ["--worker", "cyl", "--factor", f"{factor:g}"] + (["--variant", variant] if variant else [])
        rc = _run_worker(args, log)
        if rc != 0 or not f.exists():
            print(f"[cyl] FAILED {cyl_tag(factor, variant)} (rc {rc}); see {log}", flush=True)
            skipped.append(dict(level=cyl_tag(factor, variant), why=f"worker failed rc={rc}"))
            continue
        d = np.load(f)
        done[(factor, variant)] = (int(d["cells"]), float(d["rss_peak_gb"]))
        print(f"[cyl] {cyl_tag(factor, variant)}: {int(d['cells']):,} cells, mesh {float(d['t_mesh']):.0f} s, "
              f"model {float(d['t_model']):.0f} s, solve {float(d['t_solve_s5']):.0f}/{float(d['t_solve_s1']):.0f} s, "
              f"peak RSS {float(d['rss_peak_gb']):.1f} GB", flush=True)
    (OUT / "cyl_skipped.json").write_text(json.dumps(skipped, indent=1))


def run_forearm(levels, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "forearm_levels.log"
    order = [FA_CURRENT] + [e for e in sorted(levels, reverse=True) if abs(e - FA_CURRENT) > 1e-12]
    done, skipped = {}, []
    for edge in order:
        f = fa_file(edge)
        if f.exists() and FA_COMMON.exists() and not force:
            d = np.load(f)
            done[edge] = (int(d["cells"]), float(d["rss_peak_gb"]))
            print(f"[forearm] {fa_tag(edge)}: cached ({int(d['cells']):,} cells)", flush=True)
            continue
        pred = None
        if done:
            e_ref = min(done, key=lambda x: abs(np.log(x / edge)))
            pred = done[e_ref][0] * (e_ref / edge) ** 3
        last_rss = max((v[1] for v in done.values()), default=0.0)
        why = None
        if pred is not None and pred > FA_MAX_CELLS:
            why = f"predicted {pred / 1e6:.2f} M cells > {FA_MAX_CELLS / 1e6:.0f} M"
        elif last_rss > RSS_LIMIT_GB:
            why = f"previous level peaked at {last_rss:.0f} GB"
        elif free_gb() < 12:
            why = f"only {free_gb():.0f} GB available"
        if why:
            print(f"[forearm] SKIP {fa_tag(edge)}: {why}", flush=True)
            skipped.append(dict(level=fa_tag(edge), why=why))
            continue
        print(f"[forearm] {fa_tag(edge)}: running (free {free_gb():.0f} GB"
              + (f", predicted {pred / 1e3:.0f} k cells" if pred else "") + ")", flush=True)
        rc = _run_worker(["--worker", "forearm", "--edge", f"{edge:g}"], log)
        if rc != 0 or not f.exists():
            print(f"[forearm] FAILED {fa_tag(edge)} (rc {rc}); see {log}", flush=True)
            skipped.append(dict(level=fa_tag(edge), why=f"worker failed rc={rc}"))
            continue
        d = np.load(f)
        done[edge] = (int(d["cells"]), float(d["rss_peak_gb"]))
        print(f"[forearm] {fa_tag(edge)}: {int(d['cells']):,} cells, mesh {float(d['t_mesh']):.0f} s, fem {float(d['t_fem']):.0f} s, "
              f"solve {float(np.sum(d['t_solve'])):.0f} s, peak RSS {float(d['rss_peak_gb']):.1f} GB", flush=True)
    (OUT / "forearm_skipped.json").write_text(json.dumps(skipped, indent=1))


# =========================================================================== analysis
def _load_style():
    """The paper's matplotlib style (paper/arxiv:paper/figures/style.py) or an inline copy."""
    src = None
    try:
        src = subprocess.run(["git", "show", "paper/arxiv:paper/figures/style.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        pass
    ns: dict = {}
    if src:
        try:
            exec(compile(src, "style.py", "exec"), ns)
        except Exception:                                    # noqa: BLE001
            ns = {}
    if "use" not in ns:
        import matplotlib; matplotlib.use("Agg")             # noqa: E702
        import matplotlib.pyplot as plt

        def use():
            plt.rcParams.update({
                "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "sans-serif",
                "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False,
                "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
                "pdf.fonttype": 42, "ps.fonttype": 42})

        def letter(ax, s, dx=-0.12, dy=1.04):
            ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
        ns = dict(use=use, letter=letter, W2=7.0,
                  COL={"analytical": "#1f5fbf", "fem": "#d95f02", "direct": "#1b9e77", "raw": "#8c8c8c", "first": "#111111"})
    return ns


class Ana:
    """Memoised analytical cylinder φ(z) (Farina 2004) at (r, θ, electrode radius)."""

    def __init__(self):
        self.path = OUT / "analytical_cache.npz"
        self.d = dict(np.load(self.path)) if self.path.exists() else {}
        self.dirty = False

    def phi(self, r, th=0.0, dim1=5.0, w=None):
        from harness import W, ana_phi
        w = int(w or W)
        k = f"r{r:g}_th{th:g}_d{dim1:g}" + (f"_w{w}" if w != W else "")
        if k not in self.d:
            self.d[k] = ana_phi(float(r), distfib=float(th), dim1=float(dim1), w=w)[0]
            self.dirty = True
        return self.d[k]

    def save(self):
        if self.dirty:
            np.savez_compressed(self.path, **self.d)


def analyse_cyl(store: dict):
    import cyl_fem
    import harness as H
    from emgforge.synthesis.metrics import jaggedness
    from emgforge.synthesis.preprocessing import denoise_field_n
    from scipy.signal import butter, filtfilt

    DZ = H.V * 1000.0 / H.FS
    Wn = H.W
    z = (np.arange(Wn) - Wn // 2) * DZ
    core = np.abs(z) <= 60.0
    win100 = np.abs(z) <= 100.0
    ana = Ana()
    b_lp, a_lp = butter(4, 500.0 / (H.FS / 2))

    def dc_free(p, n_edge=12):
        return p - np.mean(np.r_[p[:n_edge], p[-n_edge:]])

    def dd(p):
        return np.gradient(np.gradient(p, DZ), DZ)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def _antipode(theta, prof):
        return prof - prof[int(np.argmin(np.abs(np.abs(theta) - 180.0)))]

    files = sorted(OUT.glob("cyl_f*.npz"))
    levels = []
    for f in files:
        d = dict(np.load(f))
        d["tag"] = f.stem
        levels.append(d)
    if not levels:
        print("no cylinder levels found"); return
    glob = sorted([d for d in levels if str(d["variant"]) == ""], key=lambda d: -float(d["factor"]))   # coarse → fine
    variants = [d for d in levels if str(d["variant"]) != ""]
    order = glob + variants
    z_abs, ze = glob[0]["z_abs"], float(glob[0]["ze"])
    radii = glob[0]["radii"]
    i_th0 = int(np.argmin(np.abs(glob[0]["thetas"] - 0.0)))

    def fem_phi(d, s, r, th=0.0):
        i, j = int(np.argmin(np.abs(radii - r))), int(np.argmin(np.abs(d["thetas"] - th)))
        return cyl_fem.window(d[f"phi_s{s:g}"][i, j], z_abs, ze, Wn, DZ)

    # ---- analytical references (level-independent) --------------------------------
    ana_ref = {}
    for dep in sorted(set(DEPTHS + B5_DEPTHS)):
        r = 40.0 - dep
        pa = ana.phi(r)
        ana_ref[dep] = dict(phi=pa, sfap=H.sfap(pa, DZ, 60, 60, -20.0)[1],
                            sfap_b5=H.sfap(pa, DZ, 100, 100, -20.0)[1],
                            dd={dim: dd(ana.phi(r, 0.0, dim)) for dim in (2.5, 5.0, 7.5, 10.0)})
    n_ana, r2_ana = H.power_law(B5_DEPTHS, [H.p2p(ana_ref[dp]["sfap_b5"]) for dp in B5_DEPTHS])
    # analytical transverse profile at z=0 over the full half-circle → angular FWHM (degrees);
    # arc length at the skin (what a surface array measures) = 40 mm · FWHM_rad
    th_a = np.arange(0.0, 180.01, 5.0)
    fw_ana = {}
    for dep in DEPTHS:
        r = 40.0 - dep
        prof = np.array([ana.phi(r, th)[Wn // 2] for th in th_a])
        prof = prof - prof[-1]                       # referenced to the antipode (θ = 180°)
        fw_ana[dep] = H.fwhm(np.r_[-th_a[::-1][:-1], th_a], np.r_[prof[::-1][:-1], prof])
    # oracle side-check: the analytical φ is periodic with period w·dz (250 mm at w = 256); at
    # w = 512 (500 mm) the deep fibres' tails no longer wrap into the window edges
    pk512 = [abs(dc_free(ana.phi(40.0 - dp, w=512))[256]) for dp in DEPTHS]
    pk256 = [abs(dc_free(ana.phi(40.0 - dp))[Wn // 2]) for dp in DEPTHS]
    amp512 = [H.p2p(H.sfap(ana.phi(40.0 - dp, w=512), DZ, 100, 100, -20.0)[1]) for dp in B5_DEPTHS]
    ana_w512 = dict(phi_peak_n_w256=H.power_law(DEPTHS, pk256)[0], phi_peak_n_w512=H.power_law(DEPTHS, pk512)[0],
                    sfap_n_w512=H.power_law(B5_DEPTHS, amp512)[0],
                    edge_over_peak_w256={f"{dp:g}": float(np.mean(np.r_[ana.phi(40.0 - dp)[:12], ana.phi(40.0 - dp)[-12:]]) / ana.phi(40.0 - dp)[Wn // 2]) for dp in DEPTHS},
                    edge_over_peak_w512={f"{dp:g}": float(np.mean(np.r_[ana.phi(40.0 - dp, w=512)[:12], ana.phi(40.0 - dp, w=512)[-12:]]) / ana.phi(40.0 - dp, w=512)[256]) for dp in DEPTHS})
    ana.save()
    # analytical EOF onsets (A2.3 recipe on analytical φ at r=30)
    pa30 = ana_ref[10.0]["phi"]

    def eof_pair(p):
        out = {}
        for L_near, posz, l2 in ((40.0, -20.0, 80.0), (60.0, 0.0, 60.0)):
            t1, s1 = H.sfap(p, DZ, L_near, l2, posz, H.golden_cfg(fiber_window="boxcar"))
            t2, s2 = H.sfap(p, DZ, L_near + 60.0, l2, posz, H.golden_cfg(fiber_window="boxcar"))
            dlt = np.abs(s1 - s2); k = np.where(dlt > 0.03 * dlt.max())[0]
            out[f"L{L_near:g}"] = dict(expect_ms=L_near / H.V, onset_ms=float(t1[k[0]]) if len(k) else np.nan)
        return out
    eof_ana = eof_pair(pa30)

    # ---- per-level metrics -------------------------------------------------------------
    res = {}
    for d in order:
        tag = d["tag"]
        rec = dict(factor=float(d["factor"]), variant=str(d["variant"]), cells=int(d["cells"]), nodes=int(d["nodes"]),
                   n_dofs=int(d["n_dofs"]), t_mesh_s=float(d["t_mesh"]), t_model_s=float(d["t_model"]),
                   t_solve_s={f"s{s:g}": float(d[f"t_solve_s{s:g}"]) for s in SIGMAS},
                   t_sample_s={f"s{s:g}": float(d[f"t_sample_s{s:g}"]) for s in SIGMAS},
                   rss_peak_gb=float(d["rss_peak_gb"]), rss_after_mesh_gb=float(d["rss_after_mesh_gb"]),
                   mesh_settings=json.loads(str(d["mesh_settings"])),
                   mesh_stats=json.loads(str(d["mesh_stats"])) if "mesh_stats" in d else {}, per_sigma={})
        rec["t_build_plus_solve_s"] = rec["t_mesh_s"] + rec["t_model_s"] + rec["t_solve_s"]["s5"]
        for s in SIGMAS:
            m = dict(depth={}, spectrum={})
            for dep in DEPTHS:
                r = 40.0 - dep
                pf = fem_phi(d, s, r)
                pa = ana_ref[dep]["phi"]
                a, f_ = dc_free(pa), dc_free(pf)
                pf_m = denoise_field_n(pf, DZ, n=3)
                ddf_raw, ddf_m = dd(pf), dd(pf_m)
                r_dd = {f"{dim:g}": corr(ana_ref[dep]["dd"][dim][core], ddf_m[core]) for dim in (2.5, 5.0, 7.5, 10.0)}
                t_f, s_f = H.sfap(pf, DZ, 60, 60, -20.0)
                _, s_n = H.sfap(pf, DZ, 60, 60, -20.0, H.golden_cfg(denoise="none"))
                s_a = ana_ref[dep]["sfap"]
                # FEM and analytical φ differ by a constant unit-convention scale (~10×; tier A only
                # compares shapes) → least-squares scale the analytical φ to the FEM one on the core
                scale = float(np.dot(a[core], f_[core]) / np.dot(a[core], a[core]))
                e = (f_ - scale * a)[win100]
                hw = np.hanning(len(e))
                E = np.abs(np.fft.rfft(e * hw)); fq = np.fft.rfftfreq(len(e), DZ)
                lam, Pw = 1.0 / fq[1:], E[1:] ** 2
                bands = {"gt40": lam > 40, "15to40": (lam > 15) & (lam <= 40), "6to15": (lam > 6) & (lam <= 15), "lt6": lam <= 6}
                Ptot = Pw.sum()
                # FEM-only ripple: residual of φ from its own monopole fit (no oracle involved)
                rr = (pf - pf_m)[win100]
                R = np.abs(np.fft.rfft(rr * hw)); Pr = R[1:] ** 2
                sub = lam <= 40
                m["depth"][f"{dep:g}"] = dict(
                    r_phi=corr(a[core], f_[core]),
                    r_dd_raw=corr(ana_ref[dep]["dd"][5.0][core], ddf_raw[core]),
                    r_dd_mono_dim5=r_dd["5"], r_dd_mono_best=max(r_dd.values()),
                    r_dd_mono_by_dim=r_dd,
                    peak_ratio=float(f_[Wn // 2] / a[Wn // 2]),
                    fwhm_z_fem=H.fwhm(z, f_), fwhm_z_ana=H.fwhm(z, a),
                    sfap_p2p_ratio=H.p2p(s_f) / H.p2p(s_a),
                    sfap_p2p_ratio_nodenoise=H.p2p(s_n) / H.p2p(s_a),
                    sfap_p2p_ratio_lp500=H.p2p(filtfilt(b_lp, a_lp, s_f)) / H.p2p(filtfilt(b_lp, a_lp, s_a)),
                    sfap_r=corr(s_a, s_f), sfap_r_nodenoise=corr(s_a, s_n),
                    jag_raw=jaggedness(s_n), jag_mono=jaggedness(s_f),
                    scale_fem_over_ana=scale,
                    err_rel_energy=float(np.sum(e ** 2) / np.sum((f_[win100]) ** 2)),
                    err_rms_rel_peak=float(np.sqrt(np.mean(e ** 2)) / np.abs(f_).max()),
                    err_band_frac={k: float(Pw[mask].sum() / Ptot) for k, mask in bands.items()},
                    err_peak_lambda_sub40_mm=float(lam[sub][np.argmax(Pw[sub])]),
                    ripple_rms_rel=float(np.sqrt(np.mean(rr ** 2)) / np.abs(f_).max()),
                    ripple_peak_lambda_sub40_mm=float(lam[sub][np.argmax(Pr[sub])]),
                    ripple_band_frac={k: float(Pr[mask].sum() / Pr.sum()) for k, mask in bands.items()},
                    ripple_short_rms_rel=float(np.sqrt(Pr[lam <= 15].sum() / Pr.sum()) * np.sqrt(np.mean(rr ** 2)) / np.abs(f_).max()),
                    ripple_short_peak_lambda_mm=float(lam[lam <= 15][np.argmax(Pr[lam <= 15])]),
                    fwhm_t_fem_deg=float(H.fwhm(d["trans_theta"], _antipode(d["trans_theta"], d[f"trans_s{s:g}"][int(np.argmin(np.abs(radii - r)))]))),
                    fwhm_t_ana_deg=float(fw_ana[dep]),
                )
                v_ = m["depth"][f"{dep:g}"]
                v_["fwhm_t_fem_skin_mm"] = 40.0 * np.radians(v_["fwhm_t_fem_deg"])
                v_["fwhm_t_ana_skin_mm"] = 40.0 * np.radians(v_["fwhm_t_ana_deg"])
                if dep == 10.0:
                    pk = float(np.abs(f_).max())
                    m["spectrum"] = dict(lambda_mm=lam.tolist(), err_amp=(E[1:] / pk).tolist(), ripple_amp=(R[1:] / pk).tolist())
            # depth law (tier-B5 definition)
            amp = [H.p2p(H.sfap(fem_phi(d, s, 40.0 - dp), DZ, 100, 100, -20.0)[1]) for dp in B5_DEPTHS]
            n_fem, r2_fem = H.power_law(B5_DEPTHS, amp)
            ratios = np.array([m["depth"][f"{dp:g}"]["sfap_p2p_ratio"] for dp in DEPTHS])
            coef = np.polyfit(np.log(DEPTHS), np.log(ratios), 1)
            resid = np.log(ratios) - np.polyval(coef, np.log(DEPTHS))
            pk_fem = [abs(dc_free(fem_phi(d, s, 40.0 - dp))[Wn // 2]) for dp in DEPTHS]
            pk_ana = [abs(dc_free(ana_ref[dp]["phi"])[Wn // 2]) for dp in DEPTHS]
            n_pk_fem, _ = H.power_law(DEPTHS, pk_fem); n_pk_ana, _ = H.power_law(DEPTHS, pk_ana)
            n_fem_10, _ = H.power_law(B5_DEPTHS[1:], amp[1:])
            n_ana_10, _ = H.power_law(B5_DEPTHS[1:], [H.p2p(ana_ref[dp]["sfap_b5"]) for dp in B5_DEPTHS[1:]])
            m["depth_law"] = dict(n_fem=n_fem, r2_fem=r2_fem, n_ana=n_ana, r2_ana=r2_ana, gap=n_fem - n_ana,
                                  n_fem_10_25=n_fem_10, n_ana_10_25=n_ana_10,
                                  amp_b5=amp, phi_peak_n_fem=n_pk_fem, phi_peak_n_ana=n_pk_ana,
                                  phi_peak_gap=n_pk_fem - n_pk_ana)
            m["ratio_scatter"] = dict(spread_max_over_min=float(ratios.max() / ratios.min()),
                                      trend_exponent=float(coef[0]),
                                      resid_rms_pct=float(100.0 * (np.exp(np.sqrt(np.mean(resid ** 2))) - 1.0)),
                                      resid_max_pct=float(100.0 * (np.exp(np.abs(resid).max()) - 1.0)),
                                      ratios_raw=ratios.tolist(),
                                      ratios_rel7=(ratios / ratios[0]).tolist(),            # tier-A1.2 convention
                                      ratios_norm=(ratios / np.exp(np.mean(np.log(ratios)))).tolist())
            m["eof"] = eof_pair(fem_phi(d, s, 30.0))
            m["lateral_r_phi"] = {f"{th:g}": corr(dc_free(ana.phi(30.0, th))[core], dc_free(fem_phi(d, s, 30.0, th))[core])
                                  for th in (10.0, 20.0, 30.0, 45.0)}
            rec["per_sigma"][f"s{s:g}"] = m
        res[tag] = rec
        print(f"  [analyse] {tag}: {rec['cells']:,} cells; σ5: n={rec['per_sigma']['s5']['depth_law']['n_fem']:.2f}, "
              f"scatter {rec['per_sigma']['s5']['ratio_scatter']['resid_rms_pct']:.0f} %, "
              f"r(φ'') @10mm {rec['per_sigma']['s5']['depth']['10']['r_dd_mono_dim5']:.3f}", flush=True)
    ana.save()

    # ---- successive-level convergence (FEM vs FEM, no oracle) ----------------------------
    succ = []
    seq = glob
    for k in range(len(seq) - 1):
        d0, d1 = seq[k], seq[k + 1]
        rec = dict(coarse=d0["tag"], fine=d1["tag"], cells=[int(d0["cells"]), int(d1["cells"])], per_sigma={})
        for s in SIGMAS:
            per = {}
            for dep in DEPTHS:
                r = 40.0 - dep
                p0, p1 = fem_phi(d0, s, r), fem_phi(d1, s, r)
                a0, a1 = dc_free(p0), dc_free(p1)
                m0, m1 = denoise_field_n(p0, DZ, n=3), denoise_field_n(p1, DZ, n=3)
                s0, s1 = H.sfap(p0, DZ, 60, 60, -20.0)[1], H.sfap(p1, DZ, 60, 60, -20.0)[1]
                cfg_n = H.golden_cfg(denoise="none")
                n0, n1 = H.sfap(p0, DZ, 60, 60, -20.0, cfg_n)[1], H.sfap(p1, DZ, 60, 60, -20.0, cfg_n)[1]
                per[f"{dep:g}"] = dict(r_phi=corr(a0[core], a1[core]), r_dd_raw=corr(dd(p0)[core], dd(p1)[core]),
                                       r_dd_mono=corr(dd(m0)[core], dd(m1)[core]),
                                       peak_ratio=float(a0[Wn // 2] / a1[Wn // 2]),
                                       sfap_p2p_ratio=H.p2p(s0) / H.p2p(s1), sfap_r=corr(s0, s1),
                                       sfap_p2p_ratio_nodenoise=H.p2p(n0) / H.p2p(n1), sfap_r_nodenoise=corr(n0, n1))
            n0 = res[d0["tag"]]["per_sigma"][f"s{s:g}"]["depth_law"]["n_fem"]
            n1 = res[d1["tag"]]["per_sigma"][f"s{s:g}"]["depth_law"]["n_fem"]
            fw0 = res[d0["tag"]]["per_sigma"][f"s{s:g}"]["depth"]["10"]["fwhm_t_fem_deg"]
            fw1 = res[d1["tag"]]["per_sigma"][f"s{s:g}"]["depth"]["10"]["fwhm_t_fem_deg"]
            ratios = np.array([per[f"{dp:g}"]["sfap_p2p_ratio"] for dp in DEPTHS])
            ratios_n = np.array([per[f"{dp:g}"]["sfap_p2p_ratio_nodenoise"] for dp in DEPTHS])
            rec["per_sigma"][f"s{s:g}"] = dict(
                depth=per, exponent_change=n1 - n0, fwhm_t10_change_deg=fw1 - fw0,
                sfap_p2p_max_abs_change_pct=float(100 * np.abs(ratios - 1).max()),
                sfap_p2p_median_abs_change_pct=float(100 * np.median(np.abs(ratios - 1))),
                sfap_p2p_nodenoise_max_abs_change_pct=float(100 * np.abs(ratios_n - 1).max()),
                sfap_p2p_nodenoise_median_abs_change_pct=float(100 * np.median(np.abs(ratios_n - 1))),
                min_sfap_r_nodenoise=float(min(v["sfap_r_nodenoise"] for v in per.values())),
                min_r_phi=float(min(v["r_phi"] for v in per.values())),
                min_r_dd_raw=float(min(v["r_dd_raw"] for v in per.values())),
                min_r_dd_mono=float(min(v["r_dd_mono"] for v in per.values())),
                min_sfap_r=float(min(v["sfap_r"] for v in per.values())))
        succ.append(rec)
    # vs the finest global level (for the amplitude plot)
    fin = seq[-1]
    vs_finest = {}
    for d in order:
        vs_finest[d["tag"]] = {}
        for s in SIGMAS:
            rat = [H.p2p(H.sfap(fem_phi(d, s, 40.0 - dp), DZ, 60, 60, -20.0)[1]) /
                   H.p2p(H.sfap(fem_phi(fin, s, 40.0 - dp), DZ, 60, 60, -20.0)[1]) for dp in DEPTHS]
            cfg_n = H.golden_cfg(denoise="none")
            rat_n = [H.p2p(H.sfap(fem_phi(d, s, 40.0 - dp), DZ, 60, 60, -20.0, cfg_n)[1]) /
                     H.p2p(H.sfap(fem_phi(fin, s, 40.0 - dp), DZ, 60, 60, -20.0, cfg_n)[1]) for dp in DEPTHS]
            pk = [abs(dc_free(fem_phi(d, s, 40.0 - dp))[Wn // 2]) / abs(dc_free(fem_phi(fin, s, 40.0 - dp))[Wn // 2]) for dp in DEPTHS]
            vs_finest[d["tag"]][f"s{s:g}"] = dict(sfap_p2p_ratio=rat, sfap_p2p_ratio_nodenoise=rat_n, phi_peak_ratio=pk)
    # the local variant vs the current global level (same factor)
    var_vs_cur = {}
    cur = [d for d in glob if abs(float(d["factor"]) - CYL_CURRENT) < 1e-9]
    for dv in variants:
        if not cur:
            break
        d0 = cur[0]
        var_vs_cur[dv["tag"]] = {}
        for s in SIGMAS:
            per = {}
            for dep in DEPTHS:
                r = 40.0 - dep
                p0, p1 = fem_phi(d0, s, r), fem_phi(dv, s, r)
                s0, s1 = H.sfap(p0, DZ, 60, 60, -20.0)[1], H.sfap(p1, DZ, 60, 60, -20.0)[1]
                per[f"{dep:g}"] = dict(r_phi=corr(dc_free(p0)[core], dc_free(p1)[core]),
                                       r_dd_raw=corr(dd(p0)[core], dd(p1)[core]),
                                       sfap_p2p_ratio=H.p2p(s1) / H.p2p(s0))
            var_vs_cur[dv["tag"]][f"s{s:g}"] = per
    store["cylinder"] = dict(design=dict(levels=[float(d["factor"]) for d in glob], current=CYL_CURRENT,
                                         variant=dict(name=CYL_VARIANT[0], **CYL_VARIANT[1]), sigmas=list(SIGMAS),
                                         depths_mm=DEPTHS, b5_depths_mm=B5_DEPTHS, dz_mm=DZ, core_mm=60.0, spectrum_window_mm=100.0),
                             analytical=dict(n_ana=n_ana, r2_ana=r2_ana, fwhm_t_ana={f"{k:g}": v for k, v in fw_ana.items()},
                                             eof=eof_ana, window_check=ana_w512),
                             levels=res, successive=succ, vs_finest=vs_finest, variant_vs_current=var_vs_cur,
                             skipped=json.loads((OUT / "cyl_skipped.json").read_text()) if (OUT / "cyl_skipped.json").exists() else [])
    return dict(order=order, glob=glob, res=res, succ=succ, vs_finest=vs_finest, ana_n=n_ana, fw_ana=fw_ana)


def analyse_forearm(store: dict):
    import importlib.util
    import harness as H
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts/run_pipeline.py")
    rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

    files = sorted(OUT.glob("forearm_e*.npz"))
    if not files or not FA_COMMON.exists():
        print("no forearm levels found"); return
    common = np.load(FA_COMMON)
    levels = sorted([dict(np.load(f)) | {"tag": f.stem} for f in files], key=lambda d: -float(d["edge"]))   # coarse → fine
    M, N = FA_GRID
    e0 = (M // 2) * N + N // 2
    sizes, depth = common["mu_sizes"], common["mu_depth_mm"]
    ied_along = float(json.loads(str(common["grid_info"]))["ied_along_mm"])
    bed_paths = common["bed_paths"].astype(float)

    # fixed units: size terciles × (shallowest, median, deepest) + the largest unit → ~10
    order_sz = np.argsort(sizes)
    terc = np.array_split(order_sz, 3)
    units = []
    for t in terc:
        by_depth = t[np.argsort(depth[t])]
        units += [int(by_depth[0]), int(by_depth[len(by_depth) // 2]), int(by_depth[-1])]
    if int(order_sz[-1]) not in units:
        units.append(int(order_sz[-1]))
    units = sorted(set(units))
    # reference level → the largest-amplitude column
    ref = [d for d in levels if abs(float(d["edge"]) - FA_CURRENT) < 1e-12][0]
    Wr = ref["W"]
    col_amp = [np.ptp(Wr[units].reshape(len(units), M, N, -1)[:, :, j], axis=2).max(axis=1).sum() for j in range(N)]
    jcol = int(np.argmax(col_amp))
    col_idx = [i * N + jcol for i in range(M)]
    t_ms = ref["t_ms"]

    dt = float(t_ms[1] - t_ms[0])

    def dur(w):
        return max(H.duration_ms(t_ms, w, frac=0.10), dt)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    res = {}
    for d in levels:
        W = d["W"]; phi, phic = d["phi"], d["phi_cond"]
        p2p = np.ptp(W, axis=2)
        resid = np.sqrt(np.mean((phi - phic) ** 2, axis=2)) / np.maximum(np.ptp(phi, axis=2), 1e-30)
        cv = rp.cv_readoff(W, t_ms, M, N, ied_along, 4.0, bed_paths, d["elec"])
        # amplitude scatter across depth at the centre electrode (p2p per fibre vs depth → power law → residual)
        y, x = p2p[:, e0] / sizes, depth
        coef = np.polyfit(np.log(x), np.log(y), 1); rs = np.log(y) - np.polyval(coef, np.log(x))
        rec = dict(edge=float(d["edge"]), cells=int(d["cells"]), nodes=int(d["nodes"]), n_skin_cells=int(d["n_skin_cells"]),
                   tissue_tets=json.loads(str(d["tissue_tets"])),
                   t_mesh_s=float(d["t_mesh"]), t_fem_build_s=float(d["t_fem"]), t_locate_s=float(d["t_locate"]),
                   t_solve_total_s=float(np.sum(d["t_solve"])), t_solve_per_electrode_s=float(np.mean(d["t_solve"])),
                   t_condition_s=float(d["t_cond"]), t_muap_s=float(d["t_muap"]),
                   rss_after_mesh_gb=float(d["rss_after_mesh_gb"]), rss_after_fem_gb=float(d["rss_after_fem_gb"]),
                   rss_peak_gb=float(d["rss_peak_gb"]),
                   elec_disp_mm=dict(median=float(np.median(d["elec_disp_mm"])), max=float(d["elec_disp_mm"].max())),
                   mono_resid_rel=dict(median=float(np.median(resid)), p90=float(np.percentile(resid, 90)), max=float(resid.max())),
                   phi_peak_mV_median=float(np.median(np.abs(phi).max(axis=2)) * 1e3),
                   cv_sd_median=cv["v_median_m_s"], cv_mono_median=cv["montages"]["monopolar"]["v_median_m_s"],
                   cv_sd_corrected=cv["v_median_corrected_m_s"],
                   muap_p2p_uV_centre_median=float(np.median(p2p[:, e0]) * 1e6),
                   muap_p2p_uV_units=(p2p[units, e0] * 1e6).tolist(),
                   muap_p2p_uV_column=(p2p[np.ix_(units, col_idx)] * 1e6).tolist(),
                   duration_ms_units=[dur(W[k, e0]) for k in units],
                   amp_depth=dict(exponent=float(-coef[0]), resid_rms_pct=float(100 * (np.exp(np.sqrt(np.mean(rs ** 2))) - 1))),
                   recipe_check_identical=bool(d["recipe_check_identical"]))
        rec["t_build_plus_solve_s"] = rec["t_mesh_s"] + rec["t_fem_build_s"] + rec["t_locate_s"] + rec["t_solve_total_s"]
        res[d["tag"]] = rec
        print(f"  [analyse] {d['tag']}: {rec['cells']:,} cells, skin cells {rec['n_skin_cells']}, centre p2p median "
              f"{rec['muap_p2p_uV_centre_median']:.1f} µV, CV(SD) {rec['cv_sd_median']}", flush=True)

    def compare(d0, d1):
        """d0 vs d1 (reference d1): φ along the bed, MUAPs of the fixed units."""
        phi0, phi1 = d0["phi"], d1["phi"]
        c0, c1 = d0["phi_cond"], d1["phi_cond"]
        E, n_fib = phi0.shape[:2]
        r_phi = np.array([[corr(phi0[e, f], phi1[e, f]) for f in range(0, n_fib, 3)] for e in range(E)])
        dd0, dd1 = np.diff(phi0, 2, axis=2), np.diff(phi1, 2, axis=2)
        r_dd = np.array([[corr(dd0[e, f], dd1[e, f]) for f in range(0, n_fib, 3)] for e in range(E)])
        cd0, cd1 = np.diff(c0, 2, axis=2), np.diff(c1, 2, axis=2)
        r_ddc = np.array([[corr(cd0[e, f], cd1[e, f]) for f in range(0, n_fib, 3)] for e in range(E)])
        W0, W1 = d0["W"], d1["W"]
        out = dict(r_phi=dict(median=float(np.median(r_phi)), min=float(r_phi.min()), centre_median=float(np.median(r_phi[e0]))),
                   r_dd_raw=dict(median=float(np.median(r_dd)), min=float(r_dd.min()), centre_median=float(np.median(r_dd[e0]))),
                   r_dd_cond=dict(median=float(np.median(r_ddc)), min=float(r_ddc.min())))
        for name, idx in (("centre", [e0]), ("column", col_idx)):
            rat = np.array([[np.ptp(W0[k, e]) / np.ptp(W1[k, e]) for e in idx] for k in units])
            rr = np.array([[corr(W0[k, e], W1[k, e]) for e in idx] for k in units])
            du = np.array([[dur(W0[k, e]) / dur(W1[k, e]) for e in idx] for k in units])
            # signal-bearing pairs: the unit's p2p on that electrode (finer level) is ≥ 20 % of its
            # largest p2p on the column — a near-zero MUAP has an unstable ratio and r
            colmax = np.array([np.ptp(W1[k, col_idx], axis=1).max() for k in units])
            sig = np.array([[np.ptp(W1[k, e]) >= 0.2 * colmax[i] for e in idx] for i, k in enumerate(units)])
            out[name] = dict(p2p_ratio=rat.tolist(), r=rr.tolist(), duration_ratio=du.tolist(), signal_bearing=sig.tolist(),
                             p2p_median_abs_change_pct=float(100 * np.median(np.abs(rat - 1))),
                             p2p_max_abs_change_pct=float(100 * np.abs(rat - 1).max()),
                             r_median=float(np.median(rr)), r_min=float(rr.min()),
                             p2p_median_abs_change_pct_sig=float(100 * np.median(np.abs(rat[sig] - 1))),
                             p2p_max_abs_change_pct_sig=float(100 * np.abs(rat[sig] - 1).max()),
                             r_median_sig=float(np.median(rr[sig])), r_min_sig=float(rr[sig].min()),
                             n_pairs_sig=int(sig.sum()), n_pairs=int(sig.size),
                             duration_median_abs_change_pct=float(100 * np.median(np.abs(du - 1))))
        out["cv_sd_change"] = (res[d0["tag"]]["cv_sd_median"] or np.nan) - (res[d1["tag"]]["cv_sd_median"] or np.nan)
        return out

    succ = []
    for k in range(len(levels) - 1):
        rec = dict(coarse=levels[k]["tag"], fine=levels[k + 1]["tag"], cells=[int(levels[k]["cells"]), int(levels[k + 1]["cells"])])
        rec.update(compare(levels[k], levels[k + 1]))
        succ.append(rec)
        print(f"  [analyse] {rec['coarse']} → {rec['fine']}: centre p2p change median {rec['centre']['p2p_median_abs_change_pct']:.1f} % "
              f"(max {rec['centre']['p2p_max_abs_change_pct']:.1f}), r median {rec['centre']['r_median']:.4f} (min {rec['centre']['r_min']:.4f})", flush=True)
    fin = levels[-1]
    vs_finest = {d["tag"]: compare(d, fin) for d in levels[:-1]}
    # verdict: first level whose MUAPs change < 5 % (median, centre + column) and r > 0.99 (median) vs the next finer
    verdict = None
    for rec in succ:
        ok = (max(rec["centre"]["p2p_median_abs_change_pct"], rec["column"]["p2p_median_abs_change_pct_sig"]) < 5.0
              and min(rec["centre"]["r_median"], rec["column"]["r_median_sig"]) > 0.99)
        if ok:
            verdict = dict(level=rec["coarse"], edge=res[rec["coarse"]]["edge"], cells=res[rec["coarse"]]["cells"],
                           vs=rec["fine"], cost_s=res[rec["coarse"]]["t_build_plus_solve_s"],
                           rss_gb=res[rec["coarse"]]["rss_peak_gb"]); break
    store["forearm"] = dict(design=dict(levels=[float(d["edge"]) for d in levels], current=FA_CURRENT, grid=list(FA_GRID),
                                        ied_mm=FA_IED, n_mu=FA_NMU, seed=FA_SEED, density=FA_DENSITY, workers=FA_WORKERS,
                                        fixed_units=units, unit_sizes=sizes[units].tolist(), unit_depth_mm=depth[units].tolist(),
                                        centre_electrode=e0, column=jcol, column_electrodes=col_idx, bed_hash=str(common["bed_hash"])),
                            levels=res, successive=succ, vs_finest=vs_finest, verdict=verdict,
                            skipped=json.loads((OUT / "forearm_skipped.json").read_text()) if (OUT / "forearm_skipped.json").exists() else [])
    return dict(levels=levels, res=res, succ=succ, vs_finest=vs_finest, units=units, verdict=verdict)


def figure(cyl, fa, store):
    import matplotlib.pyplot as plt
    st = _load_style(); st["use"]()
    COL, letter, W2 = st["COL"], st["letter"], st["W2"]
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fig, axes = plt.subplots(2, 4, figsize=(W2, 5.0))
    cmap = plt.get_cmap("viridis")

    def cells_axis(ax, subs=(1.0, 2.0, 5.0)):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=5.5)
    # ---- (a) cylinder: r(φ'') and p2p ratio vs depth -------------------------------------
    if cyl:
        glob, res = cyl["glob"], cyl["res"]
        n = len(glob)
        for k, d in enumerate(glob):
            c = cmap(0.1 + 0.8 * k / max(n - 1, 1)); m = res[d["tag"]]["per_sigma"]["s5"]
            lab = f"{float(d['factor']):g} ({int(d['cells']) / 1e6:.2f} M)"
            axes[0, 0].plot(DEPTHS, [m["depth"][f"{dp:g}"]["r_dd_mono_dim5"] for dp in DEPTHS], "o-", color=c, ms=2.5, label=lab)
            axes[0, 1].plot(DEPTHS, m["ratio_scatter"]["ratios_norm"], "o-", color=c, ms=2.5, label=lab)
            if abs(float(d["factor"]) - CYL_CURRENT) < 1e-9:
                m1 = res[d["tag"]]["per_sigma"]["s1"]
                axes[0, 1].plot(DEPTHS, m1["ratio_scatter"]["ratios_norm"], ":", color=c, lw=1.0, label=f"{float(d['factor']):g}, σ_s = 1 mm")
        for dv in cyl["order"]:
            if str(dv["variant"]):
                m = res[dv["tag"]]["per_sigma"]["s5"]
                axes[0, 0].plot(DEPTHS, [m["depth"][f"{dp:g}"]["r_dd_mono_dim5"] for dp in DEPTHS], "s--", color=COL["fem"], ms=2.5, label="skin-refined @0.3")
                axes[0, 1].plot(DEPTHS, m["ratio_scatter"]["ratios_norm"], "s--", color=COL["fem"], ms=2.5)
        axes[0, 0].set(xlabel="depth below skin (mm)", ylabel="r(φ″), FEM vs analytical", ylim=(0.8, 1.005))
        axes[0, 0].legend(title="factor (cells)", fontsize=5, title_fontsize=5.5, loc="lower right", handlelength=1.5)
        axes[0, 1].axhline(1.0, color="0.6", lw=0.6)
        axes[0, 1].set(xlabel="depth below skin (mm)", ylabel="SFAP p2p, FEM / analytical\n(÷ geometric mean over depth)")
        axes[0, 1].text(0.03, 0.97, "colours as in (a)\ndotted: 0.3, σ_s = 1 mm", transform=axes[0, 1].transAxes, fontsize=5.5, va="top")
        # ---- (b) error spectrum at depth 10 mm -------------------------------------------
        ax = axes[0, 2]
        for k, d in enumerate(glob):
            c = cmap(0.1 + 0.8 * k / max(n - 1, 1)); sp = res[d["tag"]]["per_sigma"]["s5"]["spectrum"]
            lam, E = np.array(sp["lambda_mm"]), np.array(sp["err_amp"])
            ax.plot(lam, E, "-", color=c, lw=0.9)
            ax.plot(lam, np.array(sp["ripple_amp"]), ":", color=c, lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel="wavelength along z (mm)", ylabel="|spectrum| / peak φ  (10 mm, σ_s = 5)")
        ax.axvline(8.0, color="0.7", lw=0.6, ls="--")
        ax.text(0.03, 0.03, "solid: φ_FEM − φ_ana\ndotted: φ_FEM − 3-monopole fit", transform=ax.transAxes, fontsize=5.5)
        ax.set_xlim(2.0, 150.0)
        # ---- (c) exponent and transverse FWHM vs cells ---------------------------------------
        cells = [int(d["cells"]) for d in glob]
        ax = axes[0, 3]
        for s, ls, mk in ((5.0, "-", "o"), (1.0, "--", "s")):
            ax.plot(cells, [res[d["tag"]]["per_sigma"][f"s{s:g}"]["depth_law"]["n_fem"] for d in glob], ls, marker=mk, ms=3,
                    color=COL["fem"], label=f"FEM σ_s={s:g} mm")
        ax.axhline(cyl["ana_n"], color=COL["analytical"], lw=1, label="analytical")
        for dv in cyl["order"]:
            if str(dv["variant"]):
                ax.plot([int(dv["cells"])], [res[dv["tag"]]["per_sigma"]["s5"]["depth_law"]["n_fem"]], "D", color=COL["direct"], ms=4, label="skin-refined")
        ax.set(xlabel="cells", ylabel="SFAP depth-law exponent n"); cells_axis(ax, subs=(1.0, 3.0)); ax.legend(fontsize=5, loc="best", handlelength=1.5)
        ax = axes[1, 0]
        for s, ls, mk in ((5.0, "-", "o"), (1.0, "--", "s")):
            ax.plot(cells, [res[d["tag"]]["per_sigma"][f"s{s:g}"]["depth"]["10"]["fwhm_t_fem_deg"] for d in glob], ls, marker=mk, ms=3,
                    color=COL["fem"], label=f"FEM σ_s={s:g} mm")
        ax.axhline(cyl["fw_ana"][10.0], color=COL["analytical"], lw=1, label="analytical (disc r=5)")
        ax.set(xlabel="cells", ylabel="transverse FWHM at 10 mm (deg)"); cells_axis(ax, subs=(1.0, 3.0)); ax.legend(fontsize=5, loc="best", handlelength=1.5)
    # ---- (d) forearm: MUAP p2p ratio and r vs cells for the fixed units -------------------------
    if fa:
        lv, res_f, units, vsf = fa["levels"], fa["res"], fa["units"], fa["vs_finest"]
        cells = [int(d["cells"]) for d in lv]
        ax1, ax2 = axes[1, 1], axes[1, 2]
        rat = np.array([[np.mean(vsf[d["tag"]]["centre"]["p2p_ratio"][i]) for d in lv[:-1]] + [1.0] for i in range(len(units))])
        rr = np.array([[np.mean(vsf[d["tag"]]["centre"]["r"][i]) for d in lv[:-1]] + [1.0] for i in range(len(units))])
        for i in range(len(units)):
            ax1.plot(cells, rat[i], "-", color=COL["direct"], alpha=0.35, lw=0.7)
            ax2.plot(cells, rr[i], "-", color=COL["direct"], alpha=0.35, lw=0.7)
        ax1.plot(cells, np.median(rat, axis=0), "o-", color=COL["direct"], ms=3, label="median of 10 units")
        ax2.plot(cells, np.median(rr, axis=0), "o-", color=COL["direct"], ms=3)
        ax1.axhspan(0.95, 1.05, color="0.9", zorder=0); ax1.axhline(1, color="0.6", lw=0.6)
        ax2.axhline(0.99, color="0.6", lw=0.6, ls="--")
        ax1.set(xlabel="cells", ylabel="forearm MUAP p2p / finest\n(10 units, centre electrode)"); cells_axis(ax1); ax1.legend(fontsize=5, loc="lower right")
        ax2.set(xlabel="cells", ylabel="forearm MUAP r vs finest\n(10 units, centre electrode)"); cells_axis(ax2)
        ax2.set_ylim(min(0.9, float(rr.min()) - 0.01), 1.002)
        cur = [d for d in lv if abs(float(d["edge"]) - FA_CURRENT) < 1e-12]
        if cur:
            for ax_ in (ax1, ax2):
                ax_.axvline(int(cur[0]["cells"]), color="0.75", lw=0.6, ls=":")
            ax1.text(int(cur[0]["cells"]), float(rat.min()), f" edge {FA_CURRENT:g}\n (current)", fontsize=5, va="bottom", color="0.4")
    # ---- (e) cost ------------------------------------------------------------------------------
    ax = axes[1, 3]; ax2 = ax.twinx()
    if cyl:
        glob, res = cyl["glob"], cyl["res"]
        cells = [int(d["cells"]) for d in glob]
        ax.plot(cells, [res[d["tag"]]["t_build_plus_solve_s"] for d in glob], "o-", color=COL["fem"], ms=3, label="cylinder time")
        ax2.plot(cells, [res[d["tag"]]["rss_peak_gb"] for d in glob], "o--", color=COL["fem"], ms=3, alpha=0.6, label="cylinder RSS")
    if fa:
        lv, res_f = fa["levels"], fa["res"]
        cells = [int(d["cells"]) for d in lv]
        ax.plot(cells, [res_f[d["tag"]]["t_build_plus_solve_s"] for d in lv], "s-", color=COL["direct"], ms=3, label="forearm time (25 solves)")
        ax2.plot(cells, [res_f[d["tag"]]["rss_peak_gb"] for d in lv], "s--", color=COL["direct"], ms=3, alpha=0.6, label="forearm RSS")
    ax.set(yscale="log", xlabel="cells", ylabel="mesh + build + solve (s)"); cells_axis(ax, subs=(1.0,))
    ax2.set(ylabel="peak RSS (GB)", yscale="log"); ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5, loc="upper left", handlelength=1.5)
    for ax_, s in zip(axes.ravel(), "abcdefgh"):
        letter(ax_, s)
    fig.tight_layout(w_pad=0.6, h_pad=1.2)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"mesh_convergence.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", FIG_DIR / "mesh_convergence.png", flush=True)


def analyse():
    store = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), machine=dict(cpus=os.cpu_count()))
    cyl = analyse_cyl(store)
    fa = analyse_forearm(store)
    (OUT / "key_numbers.json").write_text(json.dumps(jsonable(store), indent=1))
    print("wrote", OUT / "key_numbers.json", flush=True)
    figure(cyl, fa, store)


# =========================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=["all", "cyl", "forearm", "analyse"])
    ap.add_argument("--cyl-levels", default=None, help="comma-separated CharacteristicLengthFactor values")
    ap.add_argument("--forearm-levels", default=None, help="comma-separated fTetWild edge_length values")
    ap.add_argument("--no-variant", action="store_true", help="skip the skin-refined cylinder variant")
    ap.add_argument("--force", action="store_true", help="recompute cached levels")
    ap.add_argument("--worker", default=None, choices=["cyl", "forearm"], help=argparse.SUPPRESS)
    ap.add_argument("--factor", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--edge", type=float, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.worker == "cyl":
        cyl_worker(a.factor, a.variant); return 0
    if a.worker == "forearm":
        fa_worker(a.edge); return 0
    cl = [float(x) for x in a.cyl_levels.split(",")] if a.cyl_levels else CYL_LEVELS
    fl = [float(x) for x in a.forearm_levels.split(",")] if a.forearm_levels else FA_LEVELS
    if a.part in ("all", "cyl"):
        run_cyl(cl, with_variant=not a.no_variant, force=a.force)
    if a.part in ("all", "forearm"):
        run_forearm(fl, force=a.force)
    if a.part in ("all", "analyse"):
        analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
