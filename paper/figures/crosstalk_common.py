"""Crosstalk study helper — motor units of the muscles around the flexor carpi ulnaris
(FCU) seen by the 5×5 HD-EMG grid placed over the FCU (paper Study section).

Everything is built with library calls and the F2 recipe of ``f2_common``:

* the same WR forearm fibre model, and for every study muscle the same straight
  Poisson-disk fibre bed (4 fibres/mm²) and the same 100-MU Henneman pool recipe
  (``build_muscle_beds`` / ``sample_henneman_pool`` with the FCU parameters — the size
  draws are seeded, so unit ``k`` has the same fibre count in every muscle);
* the cached FCU grid (``phigrid_f2_M5_ied10.npz``, ``elec_xyz``): the 25 reciprocity
  solves are repeated (one FEniCSx solve per electrode, bit-identical to the cache — the
  lead field is a field on the whole mesh) and φ is sampled along the fibres of the
  chosen neighbouring-muscle units; plus one extra skin electrode per neighbouring unit,
  the skin point at the grid-centre height nearest to that unit ("own" electrode);
* the direct line-source synthesis (``field_to_muap`` + ``f2_common.SPCFG``) per unit,
  innervation at the atlas IZ fraction (see :func:`iz_fraction`);
* the FCU units come from the cached 100-unit MUAP tensor (same recipe).

Caches under ``_results/paper/cache/`` (``crosstalk_phi_*.pkl``, ``crosstalk_muaps_*.pkl``).
Nothing under ``src/`` is modified. Heavy stages fork *spawned* workers (dolfinx is only
imported in the children), so the calling script must guard its top level with
``if __name__ == "__main__":``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f2_common as C                                   # noqa: E402
from emgforge.synthesis import FibreBed, field_to_muap  # noqa: E402

FCU = C.FCU
# label → (abbreviation, name); the WR label key of the package's atlas map
# (``forearm_muscle_atlas.json:wr_label_to_muscle``, used by ``harmonic_fibers``)
MUSCLES = {
    8: ("FCU", "flexor carpi ulnaris"),
    13: ("FDS", "flexor digitorum superficialis"),
    7: ("FDP", "flexor digitorum profundus"),
    6: ("ECU", "extensor carpi ulnaris"),
    16: ("FCR", "flexor carpi radialis"),
}
NEIGHBOURS = [13, 7, 6, 16]
ORDER = [FCU] + NEIGHBOURS                              # bed-building order (FCU first → identical to f2_common)
COLOURS = {8: "#1b9e77", 13: "#d95f02", 7: "#7570b3", 6: "#e7298a", 16: "#e6ab02"}
N_UNITS = 7                                             # units per neighbouring muscle (log-spaced sizes, largest included)
SOURCE_SIGMA = 5.0                                      # Gaussian source width of the reciprocity solves (f2_common)
E_GRID = C.M * C.M
ATLAS = C.ROOT / "src/emgforge/mri/data/forearm_muscle_atlas.json"
CACHE_LF = C.CACHE / f"crosstalk_phi_{C.GRID_TAG}.pkl"
CACHE_W = C.CACHE / f"crosstalk_muaps_{C.GRID_TAG}.pkl"
KEY_JSON = C.HERE / "key_numbers_study_crosstalk.json"
MONTAGES = ("mono", "sd", "dd")


# --------------------------------------------------------------------------- anatomy
def iz_fraction(label: int) -> tuple[float, str]:
    """Innervation-zone fraction along the straight bed fibre (0 = proximal end of the
    field of view). FCU: the production recipe's 0.305. Others: the mean of the atlas IZ
    bands (Lieber/Barbero atlas, proximal = 0) for the label's atlas muscle, or mid-length
    (0.5) if the atlas has no value."""
    if label == FCU:
        return C.IZ_FRAC, "production FCU recipe (f2_common.IZ_FRAC = 0.305)"
    with open(ATLAS) as fh:
        atlas = json.load(fh)
    key = atlas.get("wr_label_to_muscle", {}).get(str(label))
    bands = atlas["muscles"].get(key, {}).get("IZ_fraction") if key else None
    if not bands:
        return 0.5, "no atlas IZ value → mid-length"
    return float(np.mean(bands)), f"mean of the atlas IZ bands {bands} of '{key}' (proximal = 0)"


def load_anatomy():
    """Fibre model, straight Poisson beds and Henneman pools of the five study muscles
    (FCU bed/pool identical to ``f2_common.poisson_bed`` / ``henneman_pool``)."""
    from emgforge.mri.core.motor_unit_pool import sample_henneman_pool
    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    fm = C.load_fibre_model()
    beds = build_muscle_beds(fm, density=C.DENSITY, method="poisson", labels=ORDER, min_fibers=30)
    pools = {l: sample_henneman_pool(b, n_mu=C.N_MU, size_min=5, size_max=min(400, len(b.r_norms)), seed=C.SEED)
             for l, b in beds.items()}
    return fm, beds, pools


def select_units(pool, n: int = N_UNITS) -> list[int]:
    """Pool indices with sizes nearest to ``n`` log-spaced targets from the smallest to the
    largest unit (the largest is always included)."""
    sizes = np.array([m.size for m in pool])
    chosen: list[int] = []
    for t in np.geomspace(sizes.min(), sizes.max(), n):
        for k in np.argsort(np.abs(np.log(sizes) - np.log(t)), kind="stable"):
            if int(k) not in chosen:
                chosen.append(int(k)); break
    if int(np.argmax(sizes)) not in chosen:
        chosen[-1] = int(np.argmax(sizes))
    return sorted(chosen)


def unit_xy_at(bed, mu, z_mm: float):
    """Mean (x, y) of the unit's fibres on the bed z-plane nearest ``z_mm``."""
    jz = int(np.argmin(np.abs(bed.z_vals - z_mm)))
    return bed.paths[mu.fiber_idxs][:, jz, :2].mean(axis=0), float(bed.z_vals[jz])


def unit_synth_bed(bed, mu, arc_dz, L_fib, iz_frac: float):
    """The synthesis FibreBed of one unit — ``f2_common.mu_synth_bed`` with the muscle's IZ
    fraction: IZ at ``iz_frac`` (+ the same per-fibre jitter), posz = (Lp − Ld)/2."""
    idx = mu.fiber_idxs
    rng = np.random.default_rng(int(mu.idx))
    izf = np.clip(iz_frac + rng.normal(0.0, C.IZ_JITTER, mu.size), 0.1, 0.9)
    Lp, Ld = izf * L_fib[idx], (1.0 - izf) * L_fib[idx]
    return dict(dz=arc_dz[idx], Lp=Lp, Ld=Ld)


# --------------------------------------------------------------------------- lead fields
def _lf_worker(args):
    """One spawned worker: build the FEM once, then either return the skin contour at the
    grid-centre height or solve + sample a list of electrode jobs."""
    kind, payload = args
    fem = C.build_fem()
    if kind == "contour":
        tri = C.exterior_triangles(fem)
        return C.surface_contour(tri, np.asarray(payload["cxy"]), payload["zc"], np.arange(0.0, 360.0, 0.5))
    out = []
    for job in payload:
        t0 = time.time()
        fem.solve_for_point(np.asarray(job["xyz"]), source_sigma=SOURCE_SIGMA)
        phi = {l: C.phi_along_paths(fem, P) for l, P in job["paths"].items()}
        out.append(dict(name=job["name"], phi=phi, secs=time.time() - t0))
    return out


def ensure_leadfields(n_workers: int = 2, force: bool = False) -> dict:
    """Grid lead fields along the chosen units' fibres of every neighbouring muscle + one
    'own' skin electrode per unit. Cached."""
    if CACHE_LF.exists() and not force:
        with open(CACHE_LF, "rb") as fh:
            return pickle.load(fh)
    lf = C.ensure_grid_leadfields()
    elec, zc, cxy = lf["elec_xyz"], float(lf["zc_mm"]), np.asarray(lf["limb_centre_xy"])
    fm, beds, pools = load_anatomy()
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(1, mp_context=ctx) as P:
        contour = P.submit(_lf_worker, ("contour", dict(cxy=cxy, zc=zc))).result()
    print(f"skin contour at z = {zc:.1f} mm: {len(contour)} points ({time.time()-t0:.0f}s)", flush=True)

    muscles, own_jobs = {}, []
    for l in NEIGHBOURS:
        bed, pool = beds[l], pools[l]
        units = select_units(pool)
        fib_sel = np.unique(np.concatenate([pool[k].fiber_idxs for k in units]))
        info = []
        for k in units:
            xy, zk = unit_xy_at(bed, pool[k], zc)
            own = contour[int(np.argmin(np.linalg.norm(contour[:, :2] - xy, axis=1)))]
            info.append(dict(k=int(k), size=int(pool[k].size), xy_zc=xy, z_plane=zk, own_xyz=own))
            own_jobs.append(dict(name=("own", l, int(k)), xyz=own, paths={l: bed.paths[pool[k].fiber_idxs]}))
        muscles[l] = dict(units=info, fib_sel=fib_sel)
    grid_jobs = [dict(name=("grid", i, j), xyz=elec[i, j],
                      paths={l: beds[l].paths[muscles[l]["fib_sel"]] for l in NEIGHBOURS})
                 for i in range(C.M) for j in range(C.M)]
    jobs = grid_jobs + own_jobs
    cost = [sum(P.shape[0] * P.shape[1] for P in j["paths"].values()) for j in jobs]
    order = np.argsort(cost)[::-1]
    chunks = [[jobs[i] for i in order[w::n_workers]] for w in range(n_workers)]      # greedy balance
    print(f"lead fields: {len(grid_jobs)} grid + {len(own_jobs)} own electrodes, "
          f"{sum(cost)/1e6:.2f} M sample points on {n_workers} workers ...", flush=True)
    t0 = time.time(); results = []
    with ProcessPoolExecutor(n_workers, mp_context=ctx) as P:
        for fut in as_completed([P.submit(_lf_worker, ("solve", ch)) for ch in chunks]):
            results.extend(fut.result())
    wall = time.time() - t0
    for l in NEIGHBOURS:
        n_sel, Nz = len(muscles[l]["fib_sel"]), beds[l].paths.shape[1]
        muscles[l]["phi_grid"] = np.zeros((E_GRID, n_sel, Nz))
    secs = []
    for r in results:
        kind, a, b = r["name"]; secs.append(r["secs"])
        if kind == "grid":
            for l in NEIGHBOURS:
                muscles[l]["phi_grid"][a * C.M + b] = r["phi"][l]
        else:
            u = next(u for u in muscles[a]["units"] if u["k"] == b)
            u["phi_own"] = r["phi"][a]
    out = dict(elec_xyz=elec, zc_mm=zc, limb_centre_xy=cxy, skin_contour_zc=contour, muscles=muscles,
               wall_s=wall, solve_sample_s_mean=float(np.mean(secs)), n_workers=n_workers)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE_LF, "wb") as fh:
        pickle.dump(out, fh)
    print(f"lead fields done: {wall:.0f}s wall → {CACHE_LF.name}", flush=True)
    return out


# --------------------------------------------------------------------------- MUAPs
def _mu_worker(job):
    phi, g = job["phi"], job["bed"]
    sb = FibreBed.from_arrays(dz_mm=g["dz"], len1_mm=g["Lp"], len2_mm=g["Ld"], posz_mm=(g["Lp"] - g["Ld"]) / 2.0, v=C.CV)
    W = np.zeros((phi.shape[0], C.SPCFG.w)); t_ms = None; t0 = time.time()
    for e in range(phi.shape[0]):
        res = field_to_muap(phi[e], sb, C.SPCFG)
        W[e] = res.muap; t_ms = res.t_ms
    return job["label"], job["k"], W, t_ms, time.time() - t0


def ensure_muaps(n_workers: int = 2, force: bool = False) -> dict:
    """Direct-method MUAPs of every chosen neighbouring unit on the 25 grid electrodes + its
    own electrode: ``W[label][k]`` is (26, w) V (e = row*5 + col; e = 25 the own electrode).
    Parallel over units (largest first), saved after every unit. Cached."""
    done = {}
    if CACHE_W.exists() and not force:
        with open(CACHE_W, "rb") as fh:
            done = pickle.load(fh)
    LF = ensure_leadfields(n_workers=n_workers)
    fm, beds, pools = load_anatomy()
    jobs, iz = [], {}
    for l in NEIGHBOURS:
        bed, pool = beds[l], pools[l]
        arc_dz, L_fib = C.bed_arc_geometry(bed)
        iz[l] = iz_fraction(l)
        pos = {int(f): i for i, f in enumerate(LF["muscles"][l]["fib_sel"])}
        for u in LF["muscles"][l]["units"]:
            k = u["k"]
            if (l, k) in done.get("W", {}):
                continue
            idx = pool[k].fiber_idxs
            phi = np.concatenate([LF["muscles"][l]["phi_grid"][:, [pos[int(f)] for f in idx]], u["phi_own"][None]], axis=0)
            jobs.append(dict(label=l, k=k, size=pool[k].size, phi=phi,
                             bed=unit_synth_bed(bed, pool[k], arc_dz, L_fib, iz[l][0])))
    out = dict(W=done.get("W", {}), secs=done.get("secs", {}), t_ms=done.get("t_ms"), iz=iz, wall_s=done.get("wall_s", 0.0))
    if jobs:
        jobs.sort(key=lambda j: -j["size"])
        print(f"MUAPs: {len(jobs)} units ({sum(j['size'] for j in jobs)} fibres × {E_GRID + 1} electrodes) on {n_workers} workers ...", flush=True)
        t0 = time.time(); n = 0
        with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) as P:
            for fut in as_completed([P.submit(_mu_worker, j) for j in jobs]):
                l, k, W, t_ms, dt = fut.result(); n += 1
                out["W"][(l, k)] = W; out["secs"][(l, k)] = dt; out["t_ms"] = t_ms
                print(f"  {n}/{len(jobs)}: {MUSCLES[l][0]} unit {k} ({W.shape[0]} electrodes) {dt:.0f}s "
                      f"[{time.time()-t0:.0f}s wall]", flush=True)
                with open(CACHE_W, "wb") as fh:
                    pickle.dump(out, fh)
        out["wall_s"] += time.time() - t0
        with open(CACHE_W, "wb") as fh:
            pickle.dump(out, fh)
    return out


# --------------------------------------------------------------------------- montages / metrics
def montages(Wgrid: np.ndarray) -> dict:
    """(25, T) grid MUAP → {mono (5,5,T), sd (4,5,T), dd (3,5,T)}; differences along the arm
    (grid rows, the fibre direction) with the library's ``single_diff`` / ``double_diff``."""
    from emgforge.activation import double_diff, single_diff
    g = np.asarray(Wgrid).reshape(C.M, C.M, -1)
    return {"mono": g, "sd": single_diff(g, axis=0), "dd": double_diff(g, axis=0)}


def amplitude(Wgrid: np.ndarray) -> dict:
    """Per montage: p2p map, RMS map (over the synthesis window), grid-max p2p, centre-channel
    p2p (mono (2,2); sd row 2 = electrodes 2–3; dd row 1 = electrodes 1–2–3) and the summed
    energy Σ_channels Σ_t x²."""
    out = {}
    for name, g in montages(Wgrid).items():
        p2p = np.ptp(g, axis=-1); rms = np.sqrt((g ** 2).mean(axis=-1))
        out[name] = dict(p2p=p2p, rms=rms, max=float(p2p.max()), centre=float(p2p[p2p.shape[0] // 2, C.M // 2]),
                         energy=float((g ** 2).sum()))
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
