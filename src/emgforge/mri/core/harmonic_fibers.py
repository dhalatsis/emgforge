"""Harmonic-streamline muscle-fibre geometry (Noura's 3rd method).

Where :mod:`fiber_directions` gives each muscle a single PCA/endpoint direction
(the RED/GREY morphing-disk assumption of one full-length fibre with a mid-belly
NMJ), this module resolves the *internal architecture* of a muscle from its MRI
label mask by solving a **masked Laplace problem**:

    ∇²φ = 0 in the muscle,  φ = 0 on the proximal cap,  φ = 1 on the distal cap,
    ∂φ/∂n = 0 (insulated) on the muscle surface.

The streamlines of ∇φ are contained inside the muscle and never cross by
construction, so they are anatomically-faithful fibre trajectories. Each long
streamline is then **cut into SHORT in-series fibres** (~one fascicle length
``Lf`` each), and every fibre gets ONE neuromuscular junction (NMJ) pinned to an
atlas innervation-zone (IZ) fraction (with geometric fill bands so no gap exceeds
``Lf``). This reproduces the fascicle-in-series / placed-IZ picture that the
single-direction methods throw away.

This is a clean, packaged port of the working prototype in
``_results/mri_check/noura_fibre/`` (``harmonic.py`` + ``densefib.py``); the
numerics (Laplace stencil, mask-aware gradient, RK2 tracer, IZ-cut bands) are
reproduced faithfully. Atlas fascicle lengths and IZ fractions are read from
``mri/data/forearm_muscle_atlas.json`` via its ``wr_label_to_muscle`` map.

Usage
-----
    from emgforge.mri.core.fiber_directions import MuscleFiberModel
    from emgforge.mri.core.harmonic_fibers import build_harmonic_fibers

    fm = MuscleFiberModel("WR_Segmentation.nii.gz")
    fibres = build_harmonic_fibers(fm, label=8)   # FCU short fibres
    fibres[0].half1_mm, fibres[0].half2_mm, fibres[0].iz_fraction
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label as _cc_label
from scipy.sparse import csr_matrix, identity, lil_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Atlas access (fascicle length + IZ fractions per WR label)
# ---------------------------------------------------------------------------
_ATLAS_PATH = Path(__file__).resolve().parent.parent / "data" / "forearm_muscle_atlas.json"


@lru_cache(maxsize=1)
def _load_atlas() -> dict:
    with open(_ATLAS_PATH) as f:
        return json.load(f)


def atlas_fibre_params(
    label: int,
    atlas: Optional[dict] = None,
    default_Lf: float = 60.0,
    default_iz: Tuple[float, ...] = (0.3,),
) -> Tuple[float, List[float], Optional[str]]:
    """Return ``(fascicle_length_mm, sorted_IZ_fractions, muscle_key)`` for a
    WR segmentation label.

    Falls back to ``default_Lf`` / ``default_iz`` (and ``key=None``) if the
    label is not present in the atlas' ``wr_label_to_muscle`` map — so an
    unknown label degrades gracefully instead of raising.
    """
    atlas = atlas if atlas is not None else _load_atlas()
    key = atlas.get("wr_label_to_muscle", {}).get(str(int(label)))
    if key is None or key not in atlas.get("muscles", {}):
        return float(default_Lf), sorted(default_iz), None
    m = atlas["muscles"][key]
    Lf = float(m.get("fascicle_length_mm", default_Lf))
    iz = sorted(m.get("IZ_fraction", list(default_iz)))
    return Lf, iz, key


# ---------------------------------------------------------------------------
# Masked Laplace solve + streamline tracer (ported from harmonic.py)
# ---------------------------------------------------------------------------

def keep_largest(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    lab, n = _cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def pca_frame(mask: np.ndarray, vs: np.ndarray):
    """Principal axes of the muscle voxel cloud (physical mm).

    Returns ``(centroid, pc1_long, pc2, pc3_thin)`` — pc1 is the longitudinal
    (most-elongated) axis, pc3 the thinnest.
    """
    c = np.argwhere(mask) * vs
    cc = c - c.mean(0)
    w, V = np.linalg.eigh(np.cov(cc.T))
    o = np.argsort(w)[::-1]
    return c.mean(0), V[:, o[0]], V[:, o[1]], V[:, o[2]]


def solve_laplace(
    mask: np.ndarray, vs: np.ndarray, d0: np.ndarray, d1: np.ndarray
) -> np.ndarray:
    """Solve ∇²φ = 0 with φ=0 on ``d0`` voxels, φ=1 on ``d1``, insulated
    (zero-flux Neumann) on the muscle surface.

    The 7-point stencil skips out-of-mask neighbours, which encodes the
    insulated wall (no flux across the surface). Returns the full-grid φ
    (0 outside the mask).
    """
    idx = np.full(mask.shape, -1, int)
    free = mask & ~d0 & ~d1
    fvox = np.argwhere(free)
    idx[free] = np.arange(len(fvox))
    N = len(fvox)
    A = lil_matrix((N, N))
    b = np.zeros(N)
    h2 = np.array([1 / vs[0] ** 2, 1 / vs[1] ** 2, 1 / vs[2] ** 2])
    off = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    ax = [0, 0, 1, 1, 2, 2]
    for n, (i, j, k) in enumerate(fvox):
        diag = 0.0
        for (di, dj, dk), a in zip(off, ax):
            ii, jj, kk = i + di, j + dj, k + dk
            if not (0 <= ii < mask.shape[0] and 0 <= jj < mask.shape[1] and 0 <= kk < mask.shape[2]):
                continue
            if not mask[ii, jj, kk]:
                continue  # outside muscle -> insulated (skip => zero flux)
            diag -= h2[a]
            if free[ii, jj, kk]:
                A[n, idx[ii, jj, kk]] += h2[a]
            elif d1[ii, jj, kk]:
                b[n] -= h2[a] * 1.0
            # d0 contributes 0
        A[n, n] = diag if diag != 0 else 1.0  # guard fully-insulated stray voxel
    phi = np.zeros(mask.shape)
    phi[d1] = 1.0
    Acsr = csr_matrix(A) - 1e-9 * identity(N)  # tiny Tikhonov vs residual singularity
    sol = spsolve(Acsr, b)
    phi[free] = sol
    return phi


def grad_field(phi: np.ndarray, mask: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Mask-aware gradient of φ.

    At a wall voxel a one-sided difference is taken toward the in-mask
    neighbour (never against the φ=0 fill outside the muscle, which would
    corrupt the direction toward the surface normal); central differences in
    the interior. Returns ``(..., 3)`` gradient, zero outside the mask.
    """
    g = np.zeros(phi.shape + (3,))
    m = mask
    for a in range(3):
        h = vs[a]
        pp = np.roll(phi, -1, axis=a)
        pm = np.roll(phi, 1, axis=a)
        mp = np.roll(m, -1, axis=a).copy()
        mm = np.roll(m, 1, axis=a).copy()
        last = [slice(None)] * 3
        last[a] = -1
        mp[tuple(last)] = False  # kill roll wrap-around
        first = [slice(None)] * 3
        first[a] = 0
        mm[tuple(first)] = False
        central = mp & mm
        g[..., a] = np.where(
            central,
            (pp - pm) / (2 * h),
            np.where(mp, (pp - phi) / h, np.where(mm, (phi - pm) / h, 0.0)),
        )
    g[~m] = 0
    return g


def trace(
    seed: np.ndarray,
    g: np.ndarray,
    mask: np.ndarray,
    vs: np.ndarray,
    step: float = 1.0,
    maxlen: float = 400.0,
    sign: int = 1,
) -> np.ndarray:
    """RK2-integrate the normalised gradient from ``seed`` (mm).

    Returns a polyline ``(M, 3)`` in mm. Stops at the muscle boundary.
    """
    inv = 1 / vs
    sh = np.array(mask.shape)

    def samp(p):
        c = p * inv
        i0 = np.floor(c).astype(int)
        f = c - i0
        acc = np.zeros(3)
        tot = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = i0 + [dx, dy, dz]
                    if np.any(ii < 0) or np.any(ii >= sh):
                        continue
                    if not mask[ii[0], ii[1], ii[2]]:
                        continue
                    w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                    acc += w * g[ii[0], ii[1], ii[2]]
                    tot += w
        if tot < 1e-9:
            return None
        v = acc / tot
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 1e-9 else None

    pts = [seed.copy()]
    p = seed.copy()
    for _ in range(int(maxlen / step)):
        v = samp(p)
        if v is None:
            break
        v = v * sign
        pmid = p + 0.5 * step * v
        vm = samp(pmid)
        if vm is None:
            break
        pn = p + step * (vm * sign)
        ci = np.round(pn * inv).astype(int)
        if np.any(ci < 0) or np.any(ci >= sh) or not mask[ci[0], ci[1], ci[2]]:
            break
        pts.append(pn.copy())
        p = pn
    return np.array(pts)


def _caps_by_fraction(
    mask: np.ndarray, vs: np.ndarray,
    lo=(0.05, 0.10), hi=(0.90, 0.95),
) -> Tuple[np.ndarray, np.ndarray]:
    """Proximal (φ=0) and distal (φ=1) internal cap voxel sets, defined by
    z-fraction bands of the muscle's z-extent (matches the prototype's
    ``densefib.setup``: proximal 5-10%, distal 90-95%)."""
    vox = np.argwhere(mask)
    zmm = vox[:, 2] * vs[2]
    z0, z1 = zmm.min(), zmm.max()
    span = max(z1 - z0, 1e-9)
    zf = (zmm - z0) / span
    d0 = np.zeros(mask.shape, bool)
    d1 = np.zeros(mask.shape, bool)
    for v, f in zip(vox, zf):
        if lo[0] <= f <= lo[1]:
            d0[tuple(v)] = True
        elif hi[0] <= f <= hi[1]:
            d1[tuple(v)] = True
    return d0, d1


def _smooth_polyline(f: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Drop repeated points then Gaussian-smooth each coordinate (mode=nearest)."""
    if len(f) < 2:
        return f
    keep = np.concatenate([[True], np.linalg.norm(np.diff(f, axis=0), axis=1) > 1e-6])
    f = f[keep]
    if len(f) <= 7:
        return f
    return np.stack([gaussian_filter1d(f[:, d], sigma, mode="nearest") for d in range(3)], 1)


def _arc_length(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length (mm) along a polyline, starting at 0."""
    if len(path) < 2:
        return np.zeros(len(path))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def _resample_arc(path: np.ndarray, dz: float) -> np.ndarray:
    """Resample a polyline to uniform arc-length spacing ``dz`` (mm)."""
    arc = _arc_length(path)
    total = float(arc[-1])
    if total < dz or len(path) < 2:
        return path.copy()
    n = max(int(round(total / dz)) + 1, 2)
    s_new = np.linspace(0.0, total, n)
    return np.stack([np.interp(s_new, arc, path[:, d]) for d in range(3)], 1)


def _tangents(path: np.ndarray) -> np.ndarray:
    """Unit tangents along a polyline (central diff, canonical +z)."""
    if len(path) < 2:
        return np.tile([0.0, 0.0, 1.0], (len(path), 1))
    t = np.gradient(path, axis=0)
    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(nrm, 1e-12)
    flip = t[:, 2] < 0
    t[flip] = -t[flip]
    return t


# ---------------------------------------------------------------------------
# IZ-band covering (ported from densefib._cover_bands / _cut_by_iz)
# ---------------------------------------------------------------------------

def _cover_bands(lo: float, hi: float, atlas_iz, step: float):
    """Bands ``[(phi, is_atlas), ...]`` covering the longitudinal fraction
    range ``[lo, hi]``: atlas IZ fractions pinned in place, plus geometric
    fill bands so no gap exceeds ``step`` (= Lf/Lm) — keeps fibres ~Lf long."""
    xs = sorted(f for f in atlas_iz if lo + 0.01 < f < hi - 0.01)
    res = [(f, True) for f in xs]
    pts = [lo] + xs + [hi]
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.ceil((b - a) / step)) - 1
        for i in range(1, n + 1):
            res.append((a + i * (b - a) / (n + 1), False))
    return sorted(res)


# ---------------------------------------------------------------------------
# Public container + builder
# ---------------------------------------------------------------------------

@dataclass
class HarmonicFiber:
    """One short, in-series fibre with a single placed NMJ.

    ``path`` is resampled to uniform arc-length ``dz_mm`` spacing so the
    downstream spatial SFAP engine can consume it directly. ``half1_mm`` /
    ``half2_mm`` are the semi-fibre lengths NMJ→start / NMJ→end (their sum is
    the fibre length ≈ ``Lf``). ``nmj_arc_mm`` is the NMJ arc position measured
    from ``path[0]`` (== ``half1_mm``).
    """

    path: np.ndarray            # (Nz, 3) mm, uniform arc-length spacing
    tangents: np.ndarray        # (Nz, 3) unit tangents
    dz_mm: float                # arc-length step of ``path``
    nmj_xyz: np.ndarray         # (3,) NMJ point (mm)
    nmj_arc_mm: float           # arc length from path[0] to NMJ (== half1_mm)
    half1_mm: float             # NMJ -> proximal (start) semi-length
    half2_mm: float             # NMJ -> distal  (end)  semi-length
    iz_fraction: float          # longitudinal fraction the NMJ sits on
    is_atlas: bool              # True = atlas IZ, False = geometric fill band
    band: int                   # band index along the parent streamline

    @property
    def length_mm(self) -> float:
        return self.half1_mm + self.half2_mm

    @property
    def posz_mm(self) -> float:
        """NMJ position in the spatial engine's centred-array convention:
        ``z = (arange(Nz) - Nz//2) * dz``. path[0] sits at ``-(Nz//2)*dz``,
        so the NMJ (at arc ``half1``) is at ``half1 - (Nz//2)*dz``."""
        nz = len(self.path)
        return float(self.nmj_arc_mm - (nz // 2) * self.dz_mm)


class HarmonicFibreField:
    """Masked Laplace field for one muscle + its streamline / short-fibre tools.

    Parameters
    ----------
    mask : (X, Y, Z) bool
        Binary muscle label mask (largest component is kept automatically).
    voxel_size : (3,) mm
    solve : bool
        Solve the Laplace field on construction (default True). Set False to
        only use :meth:`mean_direction` fallbacks / geometry.
    """

    def __init__(self, mask: np.ndarray, voxel_size: np.ndarray, solve: bool = True):
        self.vs = np.asarray(voxel_size, dtype=float)
        self.mask = keep_largest(np.asarray(mask, dtype=bool))
        self.ctr, self.p1, self.p2, self.p3 = pca_frame(self.mask, self.vs)
        vox = np.argwhere(self.mask)
        zmm = vox[:, 2] * self.vs[2]
        self.z0, self.z1 = float(zmm.min()), float(zmm.max())
        self.zlo = self.z0 + 0.07 * (self.z1 - self.z0)
        self.zhi = self.z1 - 0.07 * (self.z1 - self.z0)
        # Longitudinal fraction reference (project all voxels on pc1)
        s1 = (vox * self.vs - self.ctr) @ self.p1
        self.s1min = float(s1.min())
        self.Lm = float(s1.max() - s1.min())
        self.phi = None
        self.g = None
        if solve:
            self.solve()

    # -- field ------------------------------------------------------------
    def solve(self):
        d0, d1 = _caps_by_fraction(self.mask, self.vs)
        self.phi = solve_laplace(self.mask, self.vs, d0, d1)
        self.g = grad_field(self.phi, self.mask, self.vs)
        return self

    def mean_direction(self) -> np.ndarray:
        """Volume-averaged unit fibre direction from the harmonic field
        (canonical +z). Falls back to pc1 if the field is unavailable."""
        if self.g is None:
            d = self.p1.copy()
        else:
            gm = self.g[self.mask]
            nrm = np.linalg.norm(gm, axis=1)
            good = nrm > 1e-9
            if good.sum() < 3:
                d = self.p1.copy()
            else:
                d = (gm[good] / nrm[good, None]).mean(0)
        nrm = np.linalg.norm(d)
        if nrm < 1e-9:
            d = np.array([0.0, 0.0, 1.0])
        else:
            d = d / nrm
        if d[2] < 0:
            d = -d
        return d

    # -- streamlines ------------------------------------------------------
    def _streamline(self, seed: np.ndarray, step: float = 1.0, maxlen: float = 320.0):
        up = trace(seed, self.g, self.mask, self.vs, sign=1, step=step, maxlen=maxlen)
        dn = trace(seed, self.g, self.mask, self.vs, sign=-1, step=step, maxlen=maxlen)
        f = _smooth_polyline(np.vstack([dn[::-1], up]))
        return f[(f[:, 2] >= self.zlo) & (f[:, 2] <= self.zhi)]

    def _long_fraction(self, path: np.ndarray) -> np.ndarray:
        return ((path - self.ctr) @ self.p1 - self.s1min) / self.Lm

    def short_fibers(
        self,
        Lf: float,
        iz_fractions,
        grid_mm: float = 2.0,
        dz_mm: float = 1.0,
        min_pts: int = 15,
        min_seg_pts: int = 4,
    ) -> List[HarmonicFiber]:
        """Tile the muscle cross-section with streamline seeds, cut each into
        short in-series fibres, and pin every fibre's NMJ to an atlas IZ
        fraction (+ geometric fill).

        Reproduces ``densefib.dense_fill_iz`` but returns fully-resolved
        :class:`HarmonicFiber` objects (uniform-spaced path, tangents,
        per-fibre semi-lengths, placed NMJ).
        """
        if self.g is None:
            raise RuntimeError("Laplace field not solved; call .solve() first")
        iz = sorted(iz_fractions)
        step = float(np.clip(Lf / self.Lm, 0.12, 0.6))
        zc = 0.5 * (self.z0 + self.z1)
        zi = int(round(zc / self.vs[2]))
        sm = np.argwhere(self.mask[:, :, zi])
        if len(sm) == 0:
            return []
        st_x = grid_mm / self.vs[0]
        st_y = grid_mm / self.vs[1]
        shape = np.array(self.mask.shape)

        fibres: List[HarmonicFiber] = []
        for gx in np.arange(sm[:, 0].min(), sm[:, 0].max(), st_x):
            for gy in np.arange(sm[:, 1].min(), sm[:, 1].max(), st_y):
                c = np.array([int(gx), int(gy), zi])
                if not (np.all(c >= 0) and np.all(c < shape) and self.mask[tuple(c)]):
                    continue
                f = self._streamline(np.array([gx * self.vs[0], gy * self.vs[1], zc]))
                if len(f) < min_pts:
                    continue
                fibres.extend(
                    self._cut_streamline(f, iz, step, dz_mm, min_seg_pts)
                )
        return fibres

    def _cut_streamline(
        self, f: np.ndarray, iz, step: float, dz_mm: float, min_seg_pts: int,
    ) -> List[HarmonicFiber]:
        """Cut one full streamline into in-series short fibres whose NMJs sit
        on the covering bands (ported from ``densefib._cut_by_iz``)."""
        frac = self._long_fraction(f)
        lo, hi = float(frac.min()), float(frac.max())
        bands = _cover_bands(lo, hi, iz, step)
        if not bands:
            bands = [(float(frac[len(frac) // 2]), False)]
        fr = [b[0] for b in bands]
        bnds = [lo] + [(fr[k] + fr[k + 1]) / 2 for k in range(len(fr) - 1)] + [hi]
        out: List[HarmonicFiber] = []
        for k, (phik, is_atl) in enumerate(bands):
            sel = (frac >= bnds[k]) & (frac <= bnds[k + 1])
            seg = f[sel]
            if len(seg) < min_seg_pts:
                continue
            # NMJ = streamline point whose longitudinal fraction == phik
            j_global = int(np.argmin(np.abs(frac - phik)))
            nmj_xyz = f[j_global].copy()
            # arc position of the NMJ within the segment
            seg_arc = _arc_length(seg)
            j_local = int(np.argmin(np.linalg.norm(seg - nmj_xyz, axis=1)))
            nmj_arc = float(seg_arc[j_local])
            total = float(seg_arc[-1])
            half1 = max(nmj_arc, 1e-6)
            half2 = max(total - nmj_arc, 1e-6)
            rs = _resample_arc(seg, dz_mm)
            out.append(HarmonicFiber(
                path=rs,
                tangents=_tangents(rs),
                dz_mm=float(dz_mm),
                nmj_xyz=nmj_xyz,
                nmj_arc_mm=half1,
                half1_mm=half1,
                half2_mm=half2,
                iz_fraction=float(phik),
                is_atlas=bool(is_atl),
                band=int(k),
            ))
        return out


def build_harmonic_fibers(
    fiber_model,
    label: int,
    atlas: Optional[dict] = None,
    grid_mm: float = 2.0,
    dz_mm: float = 1.0,
    Lf: Optional[float] = None,
    iz_fractions=None,
) -> List[HarmonicFiber]:
    """Build the short harmonic fibres for one muscle label of a
    :class:`~emgforge.mri.core.fiber_directions.MuscleFiberModel`.

    Reads the muscle mask from ``fiber_model.seg_data == label`` and the
    fascicle length / IZ fractions from the forearm atlas (overridable via
    ``Lf`` / ``iz_fractions``).
    """
    if not hasattr(fiber_model, "seg_data"):
        raise RuntimeError("fiber_model has no seg_data; load a segmentation first")
    mask = fiber_model.seg_data == int(label)
    vs = np.asarray(fiber_model.voxel_size, dtype=float)
    if Lf is None or iz_fractions is None:
        Lf_a, iz_a, _ = atlas_fibre_params(label, atlas)
        Lf = Lf_a if Lf is None else Lf
        iz_fractions = iz_a if iz_fractions is None else iz_fractions
    field = HarmonicFibreField(mask, vs, solve=True)
    return field.short_fibers(Lf, iz_fractions, grid_mm=grid_mm, dz_mm=dz_mm)


__all__ = [
    "HarmonicFiber",
    "HarmonicFibreField",
    "build_harmonic_fibers",
    "atlas_fibre_params",
    "keep_largest",
    "solve_laplace",
    "grad_field",
    "trace",
    "pca_frame",
]
