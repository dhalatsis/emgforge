"""Integration tests for the harmonic-streamline fibre method.

Exercises both models on the *packaged* WR segmentation:

* **single-NMJ (production default)** — one full-length fibre per curved streamline,
  innervated once at mid-belly. `build_muscle_beds(method="harmonic")`.
* **series-fibering (experimental)** — each streamline cut into short in-series fibres
  with atlas-placed IZ bands. `build_muscle_beds(method="harmonic", series=True)`; emits
  an experimental warning.

Plus the regression guard that the existing poisson/hex methods are untouched. Needs the
packaged segmentation + scipy/nibabel; skipped cleanly on a pip-only install without them.
"""
import os

import numpy as np
import pytest

FCU = 8  # flexor carpi ulnaris; atlas IZ = [0.17, 0.62], Lf = 51 mm

try:
    from importlib.resources import files

    from emgforge.mri.core.fiber_directions import MuscleFiberModel
    from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
    from emgforge.mri.core.realistic_muap import (
        compute_muap_from_bed, compute_per_fiber_muap,
    )
    from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi

    _SEG = str(files("emgforge.mri.data") / "forearm_WR_segmentation.nii.gz")
    _IMPORT_ERR = None
except Exception as exc:                                   # pragma: no cover
    _SEG, _IMPORT_ERR = "", exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None or not os.path.isfile(_SEG),
    reason=f"harmonic MRI stack/data unavailable ({_IMPORT_ERR})",
)


@pytest.fixture(scope="module")
def fm():
    """The WR model with harmonic fibre directions — built once for the module."""
    m = MuscleFiberModel(_SEG)
    m.estimate_centerlines()
    m.estimate_cross_sections()
    m.estimate_fibers(method="harmonic")
    return m


@pytest.fixture(scope="module")
def bed(fm):
    """Single-NMJ harmonic bed — the production default."""
    return build_muscle_beds(fm, method="harmonic", labels=[FCU], min_fibers=10)[FCU]


def _fake_phi_along(path, elec, sigma=5.0):
    """A smooth monopole lead field 1/(4πσr) — exercises synthesis without a FEM solve."""
    r = np.linalg.norm(np.asarray(path)[:, :3] - elec, axis=1)
    return 1.0 / (4.0 * np.pi * sigma * np.maximum(r, 1.0))


def test_estimate_fibers_harmonic(fm):
    mu = fm.muscles[FCU]
    assert mu.direction_source == "harmonic"
    assert abs(mu.fiber_direction[2]) > 0.8               # points along the limb
    assert mu.fiber_angle_from_z_deg < 35.0


def test_single_nmj_at_shared_iz(fm, bed):
    """Default = single-NMJ: full-length fibres, all innervated at the SHARED IZ (frac 0.5),
    so the NMJs are co-located along the muscle even when fibre lengths differ — a truncated
    streamline gets unequal half-lengths, not an off-centre NMJ, keeping SFAPs time-aligned.
    """
    lengths = bed.half1_mm + bed.half2_mm
    assert float(lengths.mean()) > 100.0                  # spans the muscle, not ~Lf
    assert not bool(bed.is_atlas.any())                   # single IZ, not the atlas series bands
    assert float(np.abs(bed.iz_fractions - 0.5).max()) < 0.05   # every NMJ on the shared IZ band

    poisson = build_muscle_beds(fm, method="poisson", labels=[FCU], min_fibers=10)[FCU]
    ratio = float(lengths.mean() / 2) / poisson.half_mm   # both are full-length models
    assert 0.5 < ratio < 2.0


def test_paths_stay_in_mask(fm, bed):
    mask = fm.seg_data == FCU
    vs, shape = fm.voxel_size, np.array(mask.shape)
    inside = total = 0
    for p in bed.paths:
        ci = np.round(np.asarray(p) / vs).astype(int)
        okb = np.all((ci >= 0) & (ci < shape), axis=1)
        vals = np.zeros(len(ci), bool)
        okc = ci[okb]
        vals[okb] = mask[okc[:, 0], okc[:, 1], okc[:, 2]]
        inside += int(vals.sum())
        total += len(vals)
    assert inside / max(total, 1) > 0.95


def test_single_nmj_muap_spans_the_fibre(bed):
    idx = np.arange(min(8, len(bed.half1_mm)))
    elec = np.array([bed.centroid_xy[0] + 25.0, bed.centroid_xy[1] + 25.0,
                     0.5 * (bed.z_vals[0] + bed.z_vals[1])])
    phi = [_fake_phi_along(bed.paths[i], elec) for i in idx]
    _t_ms, muap = compute_muap_from_bed(bed, phi, fiber_idxs=idx)
    assert np.all(np.isfinite(muap)) and np.abs(muap).max() > 0
    assert float((bed.half1_mm[idx] + bed.half2_mm[idx]).mean()) > 80.0   # full-length


def test_series_fibering_is_experimental_and_short(fm):
    """series=True: warns, and yields short in-series fibres with atlas-placed IZ bands."""
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        bs = build_muscle_beds(fm, method="harmonic", series=True,
                               labels=[FCU], min_fibers=10)[FCU]
    lengths = bs.half1_mm + bs.half2_mm
    assert 30.0 <= float(lengths.mean()) <= 60.0          # short, ~Lf — not the whole muscle
    atl = bs.iz_fractions[bs.is_atlas]
    assert bs.is_atlas.sum() > 0
    on_band = atl[(np.abs(atl - 0.17) < 0.03) | (np.abs(atl - 0.62) < 0.03)]
    assert len(on_band) == len(atl)                       # every atlas NMJ on a literature IZ


def test_existing_methods_unchanged(fm):
    pb = build_muscle_beds(fm, method="poisson", labels=[FCU], seed=0, min_fibers=10)[FCU]
    pb2 = build_muscle_beds(fm, method="poisson", labels=[FCU], seed=0, min_fibers=10)[FCU]
    assert pb.paths.shape == pb2.paths.shape and np.allclose(pb.paths, pb2.paths)
    assert pb.half_mm == pb2.half_mm and pb.half1_mm is None   # morphing-disk fields stay unset

    n = min(10, len(pb.paths))
    elecp = np.array([pb.centroid_xy[0] + 25.0, pb.centroid_xy[1] + 25.0,
                      0.5 * (pb.z_vals[0] + pb.z_vals[-1])])
    phi_p = np.array([_fake_phi_along(pb.paths[i], elecp) for i in range(n)])
    dz = float(pb.z_vals[1] - pb.z_vals[0])
    _tA, mA = compute_per_fiber_muap(phi_p, dz, pb.half_mm,
                                     vs_per_fiber=np.full(n, 4.0), posz_per_fiber=np.zeros(n))
    _tB, mB = compute_per_fiber_muap(phi_p, dz, pb.half_mm,
                                     vs_per_fiber=np.full(n, 4.0), posz_per_fiber=np.zeros(n))
    res = generate_muap_from_phi(phi_p, dz,
                                 MUAPConfig(len1_mm=pb.half_mm, len2_mm=pb.half_mm, w=256))
    assert np.array_equal(mA, mB)
    assert np.all(np.isfinite(mA)) and np.all(np.isfinite(res.muap))
