#!/usr/bin/env python3
"""Integration tests for the harmonic-streamline fibre method (T1-T5).

Run with the fibre venv:
    fibre_venv/bin/python _results/mri_check/noura_fibre/tests/test_harmonic_integration.py

Each test prints PASS/FAIL with the actual numbers.
"""
import sys
import numpy as np

SEG = "/home/noura/Documents/Projects/PhD/mri/data/Lab/WR/WR_Segmentation.nii.gz"
FCU = 8  # bipennate flexor carpi ulnaris; atlas IZ = [0.17, 0.62], Lf = 51mm

from emgforge.mri.core.fiber_directions import MuscleFiberModel
from emgforge.mri.core.muscle_fiber_bed import build_muscle_beds
from emgforge.mri.core.realistic_muap import (
    compute_muap_from_bed, compute_per_fiber_muap,
)
from emgforge.synthesis.api import MUAPConfig, generate_muap_from_phi

results = []


def check(name, cond, detail):
    tag = "PASS" if cond else "FAIL"
    results.append(cond)
    print(f"[{tag}] {name}: {detail}")


def fake_phi_along(path, elec, sigma=5.0):
    """A smooth monopole lead field 1/(4πσ r) sampled along a fibre path —
    enough to exercise the synthesis machinery without a real FEM solve."""
    r = np.linalg.norm(np.asarray(path)[:, :3] - elec, axis=1)
    return 1.0 / (4.0 * np.pi * sigma * np.maximum(r, 1.0))


# ---------------------------------------------------------------------------
print("Loading segmentation + estimating geometry ...")
fm = MuscleFiberModel(SEG)
fm.estimate_centerlines()
fm.estimate_cross_sections()

# === T1 ====================================================================
try:
    fm.estimate_fibers(method="harmonic")
    fd = fm.muscles[FCU].fiber_direction
    angle = fm.muscles[FCU].fiber_angle_from_z_deg
    src = fm.muscles[FCU].direction_source
    ok = (src == "harmonic") and abs(fd[2]) > 0.8 and angle < 35.0
    check("T1 estimate_fibers(harmonic)",
          ok, f"FCU dir={np.round(fd,3)} angle_from_z={angle:.1f}deg source={src}")
except Exception as e:
    check("T1 estimate_fibers(harmonic)", False, f"crashed: {e!r}")

# === T2 ====================================================================
try:
    beds_h = build_muscle_beds(fm, method="harmonic", labels=[FCU], min_fibers=10)
    bed = beds_h[FCU]
    lengths = bed.half1_mm + bed.half2_mm
    mean_len = float(lengths.mean())

    # atlas-pinned NMJs should cluster on {0.17, 0.62}
    atl = bed.iz_fractions[bed.is_atlas]
    near17 = atl[np.abs(atl - 0.17) < 0.03]
    near62 = atl[np.abs(atl - 0.62) < 0.03]
    clustered = (bed.is_atlas.sum() > 0
                 and len(near17) + len(near62) == len(atl))

    poisson = build_muscle_beds(fm, method="poisson", labels=[FCU], min_fibers=10)
    half_p = poisson[FCU].half_mm

    ok = (40.0 <= mean_len <= 55.0) and clustered and half_p > 80.0
    check("T2 short fibres + placed IZ vs full-length poisson", ok,
          f"harmonic mean fibre len={mean_len:.1f}mm "
          f"(min {lengths.min():.0f}/max {lengths.max():.0f}), "
          f"n_fibres={len(lengths)}, atlas-IZ unique={np.unique(np.round(atl,2))}; "
          f"poisson half_mm={half_p:.1f} (full len ~{2*half_p:.0f}mm)")
except Exception as e:
    import traceback; traceback.print_exc()
    check("T2 short fibres + placed IZ", False, f"crashed: {e!r}")
    bed = None

# === T3 ====================================================================
try:
    mask = fm.seg_data == FCU
    vs = fm.voxel_size
    shape = np.array(mask.shape)
    inside = 0
    total = 0
    for p in bed.paths:
        ci = np.round(np.asarray(p) / vs).astype(int)
        ok_bounds = np.all((ci >= 0) & (ci < shape), axis=1)
        vals = np.zeros(len(ci), bool)
        okc = ci[ok_bounds]
        vals[ok_bounds] = mask[okc[:, 0], okc[:, 1], okc[:, 2]]
        inside += int(vals.sum())
        total += len(vals)
    frac = inside / max(total, 1)
    check("T3 containment (paths inside mask)", frac > 0.95,
          f"{frac*100:.1f}% of {total} sampled points in-mask")
except Exception as e:
    check("T3 containment", False, f"crashed: {e!r}")

# === T4 ====================================================================
try:
    # a handful of fibres = a small harmonic MU
    idx = np.arange(min(8, len(bed.half1_mm)))
    z_span = float(bed.z_vals[-1] - bed.z_vals[0])  # muscle z-extent
    elec = np.array([bed.centroid_xy[0] + 25.0, bed.centroid_xy[1] + 25.0,
                     0.5 * (bed.z_vals[0] + bed.z_vals[1])])
    phi_list = [fake_phi_along(bed.paths[i], elec) for i in idx]
    t_ms, muap = compute_muap_from_bed(bed, phi_list, fiber_idxs=idx)
    fibre_len = (bed.half1_mm[idx] + bed.half2_mm[idx])
    finite = np.all(np.isfinite(muap)) and np.abs(muap).max() > 0
    extent_ok = fibre_len.mean() < 0.4 * z_span and fibre_len.mean() < 60.0
    check("T4 harmonic MUAP via spatial engine", finite and extent_ok,
          f"muap finite={finite} peak={np.abs(muap).max():.2e}, "
          f"mean propagation extent={fibre_len.mean():.1f}mm "
          f"vs muscle z-extent={z_span:.0f}mm")
except Exception as e:
    import traceback; traceback.print_exc()
    check("T4 harmonic MUAP", False, f"crashed: {e!r}")

# === T5 ====================================================================
try:
    # Existing poisson/hex path must still run and be deterministic/unchanged.
    pb = build_muscle_beds(fm, method="poisson", labels=[FCU], seed=0, min_fibers=10)[FCU]
    pb2 = build_muscle_beds(fm, method="poisson", labels=[FCU], seed=0, min_fibers=10)[FCU]
    deterministic = (pb.paths.shape == pb2.paths.shape
                     and np.allclose(pb.paths, pb2.paths)
                     and pb.half_mm == pb2.half_mm
                     and pb.half1_mm is None)  # old method leaves harmonic fields unset

    # Full-length MUAP through the untouched global-half path.
    n = min(10, len(pb.paths))
    elecp = np.array([pb.centroid_xy[0] + 25.0, pb.centroid_xy[1] + 25.0,
                      0.5 * (pb.z_vals[0] + pb.z_vals[-1])])
    phi_p = np.array([fake_phi_along(pb.paths[i], elecp) for i in range(n)])
    dz = float(pb.z_vals[1] - pb.z_vals[0])
    tA, mA = compute_per_fiber_muap(phi_p, dz, pb.half_mm,
                                    vs_per_fiber=np.full(n, 4.0),
                                    posz_per_fiber=np.zeros(n))
    tB, mB = compute_per_fiber_muap(phi_p, dz, pb.half_mm,
                                    vs_per_fiber=np.full(n, 4.0),
                                    posz_per_fiber=np.zeros(n))
    # also exercise the Fourier API directly (fast path, no jitter)
    res = generate_muap_from_phi(phi_p, dz, MUAPConfig(len1_mm=pb.half_mm,
                                                       len2_mm=pb.half_mm, w=256))
    ok = (deterministic and np.array_equal(mA, mB)
          and np.all(np.isfinite(mA)) and np.all(np.isfinite(res.muap)))
    check("T5 regression: poisson bed + global-half MUAP unchanged", ok,
          f"bed deterministic={deterministic}, half_mm={pb.half_mm:.1f}, "
          f"per-fibre MUAP reproducible={np.array_equal(mA, mB)}, "
          f"Fourier-API peak={np.abs(res.muap).max():.2e}")
except Exception as e:
    import traceback; traceback.print_exc()
    check("T5 regression", False, f"crashed: {e!r}")

# ---------------------------------------------------------------------------
print("\n" + ("ALL PASS" if all(results) else "SOME FAILED")
      + f"  ({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
