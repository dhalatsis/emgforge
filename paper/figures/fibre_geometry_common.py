"""Study: does fibre geometry matter? — straight (Poisson) vs harmonic-streamline fibre beds
of the WR FCU, same electrode grid, same motor units (helpers + cached compute stages).

Everything reuses ``f2_common`` (beds, pool, grid, the direct line-source recipe ``SPCFG``)
and the library synthesis (``field_to_muap`` / ``FibreBed``). The only new logic is the
*matched-unit* rule (the smallest helper that lets the same unit exist on both beds, which
the library has no API for): a released-pool unit (centre, size) is re-drawn on the harmonic
bed as the ``size`` streamlines nearest to the same centre in the muscle mid-z plane — the
exact rule ``sample_henneman_pool`` used on the straight bed (``size`` nearest fibres to the
anchor's ``xy_mid``).

Cached under ``_results/paper/cache/``:
  * ``phigrid_harm_f2_M5_ied10_g0.5.npz`` — φ of every grid electrode along every harmonic
    streamline (one FEM reciprocity solve per electrode, same solver/grid as the straight
    lead fields; NaN-padded because the streamlines have different lengths);
  * ``study_fibres_muaps.npz`` — the synthesized MUAPs of the chosen units on both beds.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_fibre_geometry.py
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f2_common as C
from emgforge.synthesis import FibreBed, field_to_muap

HARM_LF = C.CACHE / f"phigrid_harm_f2_{C.GRID_TAG}_g{C.HARM_GRID_MM:g}.npz"
MUAP_CACHE = C.CACHE / "study_fibres_muaps.npz"
KEY_JSON = C.HERE / "key_numbers_study_fibres.json"
N_UNITS = 24
N_PER_STRATUM = 8
FORCED = [6, 58, 95, 99]                  # Fig 5's S/M/L units + the largest unit
STRATA = dict(small=(0, 20), medium=(21, 150), large=(151, 10 ** 6))     # fibres per MU
Z_PLANE = None                            # set by mid_plane_z(): the muscle mid-z plane of the straight bed


# --------------------------------------------------------------------------- beds
def mid_plane_z(pbed) -> float:
    """z of the straight bed's ``xy_mid`` sample (index Nz//2 of its z grid)."""
    return float(pbed.z_vals[len(pbed.z_vals) // 2])


def xy_at_plane(paths, z_plane: float):
    """(N, 2) xy of each path at the sample nearest to ``z_plane`` (+ its |Δz|)."""
    xy = np.zeros((len(paths), 2)); dzs = np.zeros(len(paths))
    for i, p in enumerate(paths):
        p = np.asarray(p); j = int(np.argmin(np.abs(p[:, 2] - z_plane)))
        xy[i] = p[j, :2]; dzs[i] = abs(p[j, 2] - z_plane)
    return xy, dzs


def harmonic_bed_iz(fm, pbed):
    """The density-matched harmonic bed re-innervated on the straight bed's IZ plane
    (the convention of Fig 4 / key_numbers_f2 fig6_mri_fibres.harmonic_nmj)."""
    k_iz = int(round(C.IZ_FRAC * (pbed.paths.shape[1] - 1)))
    nmj_p = pbed.paths[:, k_iz]
    frame = C.harmonic_frame(fm)
    iz_frame = float(frame._long_fraction(nmj_p).mean())
    hbed05 = C.harmonic_bed(fm, grid_mm=C.HARM_GRID_MM)
    hbed = C.harmonic_bed_at_iz(hbed05, frame, iz_frame)
    lfr = [frame._long_fraction(np.asarray(p)) for p in hbed.paths]
    hbed.reaches_iz = np.array([f.min() <= iz_frame <= f.max() for f in lfr])
    hbed.iz_frame = iz_frame; hbed.z_iz_plane = float(nmj_p[:, 2].mean())
    return hbed, frame


def matched_unit(hbed_xy, centre_xy, size: int):
    """The harmonic fibres of a released-pool unit: the ``size`` streamlines nearest to the
    unit's centre in the mid-z plane (the pool's own territory rule). Returns (idx, radius)."""
    d = np.linalg.norm(hbed_xy - np.asarray(centre_xy), axis=1)
    order = np.argsort(d)[:int(size)]
    return order, float(d[order].max())


def harm_synth_bed(hbed, idx, mu_idx: int, sigma_mm: float):
    """The synthesis FibreBed of one matched unit on the harmonic bed: NMJ at the IZ plane
    (+ the same jitter as the straight recipe, in mm), per-fibre arc dz, posz in the
    engine's centred-array convention (HarmonicFiber.posz_mm)."""
    rng = np.random.default_rng(int(mu_idx))
    dz = np.zeros(len(idx)); l1 = np.zeros(len(idx)); l2 = np.zeros(len(idx)); pz = np.zeros(len(idx))
    for k, i in enumerate(idx):
        p = np.asarray(hbed.paths[i]); seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total = float(seg.sum()); dz[k] = float(seg.mean())
        nmj = float(np.clip(hbed.half1_mm[i] + rng.normal(0.0, sigma_mm), 0.5, total - 0.5))
        l1[k], l2[k] = nmj, total - nmj
        pz[k] = nmj - (len(p) // 2) * dz[k]
    return FibreBed.from_arrays(dz_mm=dz, len1_mm=l1, len2_mm=l2, posz_mm=pz, v=C.CV)


def muap_ragged(phis, bed):
    """``field_to_muap`` for a bed whose fibres have different lengths: the same per-fibre
    ``compute_sfap_spatial`` sum, fibre by fibre (the library's matrix entry needs equal Nz)."""
    out = None; t_ms = None
    for phi_i, fb in zip(phis, bed):
        res = field_to_muap(np.asarray(phi_i, float), FibreBed((fb,)), C.SPCFG)
        out = res.muap if out is None else out + res.muap; t_ms = res.t_ms
    return t_ms, out


# --------------------------------------------------------------------------- lead fields
def ensure_harmonic_leadfields(force: bool = False):
    """φ of every grid electrode along every harmonic streamline (cached, NaN-padded)."""
    if HARM_LF.exists() and not force:
        return dict(np.load(HARM_LF))
    lf = C.ensure_grid_leadfields()
    fm = C.load_fibre_model(); pbed = C.poisson_bed(fm)
    hbed, _ = harmonic_bed_iz(fm, pbed)
    fem = C.build_fem()
    elec = lf["elec_xyz"]
    paths = [np.asarray(p) for p in hbed.paths]
    n_pts = np.array([len(p) for p in paths]); flat = np.vstack(paths); off = np.r_[0, np.cumsum(n_pts)]
    N, Nmax = len(paths), int(n_pts.max())
    phi = np.full((C.M, C.M, N, Nmax), np.nan, np.float32)
    chk = None; t_solve, t_samp = [], []
    for i in range(C.M):
        for j in range(C.M):
            t0 = time.time(); fem.solve_for_point(elec[i, j], source_sigma=5.0); t_solve.append(time.time() - t0)
            t0 = time.time(); v = fem.evaluate_solution_at_points(flat); t_samp.append(time.time() - t0)
            for k in range(N):
                phi[i, j, k, :n_pts[k]] = v[off[k]:off[k + 1]]
            if (i, j) == (C.M // 2, C.M // 2):        # same solve → must reproduce the cached straight φ
                sp = C.phi_along_paths(fem, pbed.paths[:20])
                chk = float(np.max(np.abs(sp - lf["phi_grid"][i, j, :20])) / np.max(np.abs(sp)))
            print(f"  harmonic lead fields: electrode ({i},{j}) solve {t_solve[-1]:.2f}s sample {t_samp[-1]:.1f}s", flush=True)
    out = dict(phi_harm=phi, n_pts=n_pts, elec_xyz=elec, solve_s=float(np.mean(t_solve)), sample_s=float(np.mean(t_samp)),
               straight_phi_rel_maxdiff=chk, fem_build_s=fem.build_seconds)
    np.savez_compressed(HARM_LF, **out)
    return out


# --------------------------------------------------------------------------- unit choice
def choose_units(sizes, depth):
    """N_PER_STRATUM units per size stratum spanning depth (forced units first, then the
    unit nearest to each of the remaining evenly spaced depth targets)."""
    chosen = []
    for lo, hi in STRATA.values():
        cand = [k for k in range(len(sizes)) if lo <= sizes[k] <= hi]
        forced = [k for k in FORCED if k in cand]
        picked = list(forced)
        n_more = N_PER_STRATUM - len(picked)
        targets = np.linspace(depth[cand].min(), depth[cand].max(), N_PER_STRATUM)
        # drop the targets already covered by the forced units
        for k in forced:
            targets = np.delete(targets, int(np.argmin(np.abs(targets - depth[k]))))
        for tg in targets[:n_more]:
            rest = [k for k in cand if k not in picked]
            picked.append(int(rest[int(np.argmin(np.abs(depth[rest] - tg)))]))
        chosen += sorted(picked)
    return chosen


# --------------------------------------------------------------------------- synthesis
_CTX: dict = {}


def _task(args):
    k, which = args
    d = _CTX
    E = d["elecs"][k]                                     # list of flat electrode indices
    t0 = time.time()
    if which == "straight":
        mu = d["pool"][k]; idx = mu.fiber_idxs
        sb = C.mu_synth_bed(d["pbed"], mu, d["arc_dz"], d["L_fib"])
        W = np.zeros((len(E), C.SPCFG.w)); t_ms = None
        for n, e in enumerate(E):
            res = field_to_muap(d["phi_p"][e][idx], sb, C.SPCFG); W[n] = res.muap; t_ms = res.t_ms
    else:
        idx = d["hidx"][k]; sb = d["hbeds"][k]
        W = np.zeros((len(E), C.SPCFG.w)); t_ms = None
        for n, e in enumerate(E):
            phis = [d["phi_h"][e, i, :d["n_pts"][i]] for i in idx]
            t_ms, W[n] = muap_ragged(phis, sb)
    return k, which, W, t_ms, time.time() - t0


def ensure_study_muaps(n_workers: int = 2, force: bool = False):
    """The chosen units' MUAPs on both beds at their column + centre electrodes (cached)."""
    if MUAP_CACHE.exists() and not force:
        d = dict(np.load(MUAP_CACHE, allow_pickle=True))
        if d["done"].all():
            return d
    lf = C.ensure_grid_leadfields(); ten = C.ensure_muap_tensor()
    hl = ensure_harmonic_leadfields()
    fm = C.load_fibre_model(); pbed = C.poisson_bed(fm); pool = C.henneman_pool(bed=pbed)
    hbed, frame = harmonic_bed_iz(fm, pbed)
    Wt = ten["W"]; sizes = np.array([m.size for m in pool])
    e0 = lf["elec_xyz"][C.M // 2, C.M // 2]
    cxy = np.array([m.centre_xy for m in pool]); depth = np.linalg.norm(cxy - e0[:2], axis=1)
    units = choose_units(sizes, depth)
    # electrode set per unit: the column holding the largest |MUAP| in the released tensor + the centre
    Wg = Wt.reshape(C.N_MU, C.M, C.M, -1)
    col = np.argmax(np.abs(Wg).max(axis=(1, 3)), axis=1)
    elecs = {}
    for k in units:
        E = [i * C.M + int(col[k]) for i in range(C.M)]
        if C.E0 not in E:
            E.append(C.E0)
        elecs[k] = E
    # matched units on the harmonic bed
    zp = mid_plane_z(pbed)
    hxy, _ = xy_at_plane(hbed.paths, zp)
    sigma_mm = C.IZ_JITTER * float(C.bed_arc_geometry(pbed)[1].mean())
    hidx, hrad, hbeds = {}, {}, {}
    for k in units:
        hidx[k], hrad[k] = matched_unit(hxy, pool[k].centre_xy, pool[k].size)
        hbeds[k] = harm_synth_bed(hbed, hidx[k], k, sigma_mm)
    arc_dz, L_fib = C.bed_arc_geometry(pbed)
    _CTX.update(pool=pool, pbed=pbed, arc_dz=arc_dz, L_fib=L_fib, elecs=elecs, hidx=hidx, hbeds=hbeds,
                phi_p=lf["phi_grid"].reshape(C.M * C.M, *lf["phi_grid"].shape[2:]),
                phi_h=hl["phi_harm"].reshape(C.M * C.M, *hl["phi_harm"].shape[2:]), n_pts=hl["n_pts"])
    nE = max(len(E) for E in elecs.values())
    W = np.full((len(units), 2, nE, C.SPCFG.w), np.nan); done = np.zeros((len(units), 2), bool); secs = np.zeros((len(units), 2))
    t_ms = None
    if MUAP_CACHE.exists():
        d = dict(np.load(MUAP_CACHE, allow_pickle=True))
        if list(d["units"]) == units and d["W"].shape == W.shape:
            W, done, secs, t_ms = d["W"], d["done"], d["secs"], d["t_ms"]
    tasks = [(k, w) for k in sorted(units, key=lambda k: -sizes[k]) for w in ("straight", "harmonic")
             if not done[units.index(k), ("straight", "harmonic").index(w)]]
    print(f"study MUAPs: {len(tasks)} tasks ({len(units)} units × 2 beds, ≤{nE} electrodes) on {n_workers} workers", flush=True)

    def save():
        np.savez_compressed(MUAP_CACHE, W=W, t_ms=t_ms, done=done, secs=secs, units=np.array(units),
                            n_elec=np.array([len(elecs[k]) for k in units]),
                            elecs=np.array([elecs[k] + [-1] * (nE - len(elecs[k])) for k in units]),
                            column=np.array([int(col[k]) for k in units]),
                            hidx=np.array([np.asarray(hidx[k]) for k in units], dtype=object),
                            hrad=np.array([hrad[k] for k in units]),
                            h_len1=np.array([hbeds[k].len1_mm for k in units], dtype=object),
                            h_len2=np.array([hbeds[k].len2_mm for k in units], dtype=object),
                            h_posz=np.array([hbeds[k].posz_mm for k in units], dtype=object),
                            sigma_jitter_mm=sigma_mm, z_plane=zp)

    t0 = time.time(); n_fin = 0
    with mp.get_context("fork").Pool(n_workers) as P:
        for k, which, Wk, tk, dt in P.imap_unordered(_task, tasks):
            u, b = units.index(k), ("straight", "harmonic").index(which)
            W[u, b, :Wk.shape[0]] = Wk; t_ms = tk; done[u, b] = True; secs[u, b] = dt; n_fin += 1
            print(f"  {n_fin}/{len(tasks)}: MU {k} ({sizes[k]} fibres) {which} {dt:.0f}s  [{time.time()-t0:.0f}s wall]", flush=True)
            if n_fin % 4 == 0:
                save()
    save()
    print(f"study MUAPs done: {time.time()-t0:.0f}s wall, {secs.sum():.0f}s CPU", flush=True)
    return dict(np.load(MUAP_CACHE, allow_pickle=True))


FULLSPAN_CACHE = C.CACHE / "study_fibres_muaps_fullspan.npz"


def ensure_fullspan_muaps(n_workers: int = 2, force: bool = False):
    """Robustness variant: the same units re-drawn on the harmonic bed from the streamlines
    that reach the IZ plane only (what a full-length-only harmonic pool would give — the
    library's `min_span_frac` modelling call). Only the units whose fibre set changes are
    re-synthesized; the others copy the main harmonic result. Cached."""
    if FULLSPAN_CACHE.exists() and not force:
        return dict(np.load(FULLSPAN_CACHE, allow_pickle=True))
    S = ensure_study_muaps(n_workers=n_workers)
    lf = C.ensure_grid_leadfields(); hl = ensure_harmonic_leadfields()
    fm = C.load_fibre_model(); pbed = C.poisson_bed(fm); pool = C.henneman_pool(bed=pbed)
    hbed, _ = harmonic_bed_iz(fm, pbed)
    units = [int(u) for u in S["units"]]
    zp = float(S["z_plane"]); sigma_mm = float(S["sigma_jitter_mm"])
    hxy, _ = xy_at_plane(hbed.paths, zp)
    glob = np.where(hbed.reaches_iz)[0]
    elecs = {k: [int(e) for e in S["elecs"][u][:S["n_elec"][u]]] for u, k in enumerate(units)}
    hidx, hrad, hbeds, changed = {}, {}, {}, []
    for u, k in enumerate(units):
        order, r = matched_unit(hxy[glob], pool[k].centre_xy, pool[k].size)
        hidx[k], hrad[k] = glob[order], r
        hbeds[k] = harm_synth_bed(hbed, hidx[k], k, sigma_mm)
        if set(hidx[k].tolist()) != set(np.asarray(S["hidx"][u]).tolist()):
            changed.append(k)
    _CTX.update(pool=pool, elecs=elecs, hidx=hidx, hbeds=hbeds,
                phi_h=hl["phi_harm"].reshape(C.M * C.M, *hl["phi_harm"].shape[2:]), n_pts=hl["n_pts"])
    W = S["W"][:, 1].copy(); secs = np.zeros(len(units))
    tasks = [(k, "harmonic") for k in sorted(changed, key=lambda k: -pool[k].size)]
    print(f"full-span variant: {len(tasks)} units to re-synthesize ({changed})", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(n_workers) as P:
        for k, _, Wk, tk, dt in P.imap_unordered(_task, tasks):
            u = units.index(k); W[u, :Wk.shape[0]] = Wk; secs[u] = dt
            print(f"  MU {k} ({pool[k].size} fibres) {dt:.0f}s  [{time.time()-t0:.0f}s wall]", flush=True)
    np.savez_compressed(FULLSPAN_CACHE, W=W, units=np.array(units), changed=np.array(changed), secs=secs,
                        hidx=np.array([np.asarray(hidx[k]) for k in units], dtype=object),
                        hrad=np.array([hrad[k] for k in units]),
                        h_len1=np.array([hbeds[k].len1_mm for k in units], dtype=object),
                        h_len2=np.array([hbeds[k].len2_mm for k in units], dtype=object),
                        n_reaching=int(len(glob)))
    return dict(np.load(FULLSPAN_CACHE, allow_pickle=True))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("lf", "all"):
        hl = ensure_harmonic_leadfields()
        print("harmonic lead fields:", {k: (v if np.ndim(v) == 0 else np.shape(v)) for k, v in hl.items()})
    if stage in ("muaps", "all"):
        d = ensure_study_muaps(n_workers=2)
        print("units:", d["units"].tolist(), "secs per (unit, bed):", np.round(d["secs"], 0).tolist())
    if stage in ("fullspan", "all"):
        d = ensure_fullspan_muaps(n_workers=2)
        print("full-span variant: changed units", d["changed"].tolist(), "secs", np.round(d["secs"], 0).tolist())
