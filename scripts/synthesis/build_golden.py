"""Build the cylindrical reference set — the fixed fixture the spatial engine
must always reproduce.

For a panel of cylindrical geometries we store (φ, geometry, golden_sfap) where
``golden_sfap`` is the **validated Fourier pipeline** (`emgforge.synthesis.engines.fourier`,
Farina 2004, validated r=0.997 vs MATLAB) applied to the *same* φ the spatial
engine will be fed. This is an operator-consistency anchor: a correct spatial
(time-domain) engine must match golden_sfap, because both compute the same
line-source integral ``∫ φ(z) ∂²Vm/∂z²(z−vt) dz`` — just in different domains.

φ sources — clean, deterministic **cylinder-like lead fields** that decay to ~0
at the window edges, so the Fourier reference is artifact-free:
  - monophasic Gaussian (σ = depth proxy);
  - biphasic difference-of-Gaussians (positive hump + negative tails, the shape
    of a real layered-cylinder lead field).

Cases sweep width/depth, L1/L2 (symmetric + asymmetric), v, and NMJ offset posz
(electrode-over-NMJ AND electrode-offset, the realistic propagating case).

NOTE on `analytical_phi_along_fibre`: it is not used here because this fixture is
the fast engine-to-engine gate built from simple, edge-decayed fields. The strict
independent cylinder oracle lives in `tests/validation/test_cylinder_oracle.py`;
it confirms that analytical φ → production Fourier pipeline reproduces the
analytical MUAP in both shape and SI-scaled amplitude.

Output: golden_cylindrical.npz (committed-small) + a Drive copy via the manifest.
Run: python \
        scripts/synthesis/build_golden.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from emgforge.synthesis.engines.fourier import (build_fourier_grids, build_spe2_iap_spectrum,
                                     radon_section, build_time_vector_ms,
                                     compute_C_from_phi_z)
from emgforge.synthesis.preprocessing import resample_centered_line

HERE = Path(__file__).resolve().parent
GOLDEN = Path(__file__).resolve().parents[2] / "tests/synthesis/data/golden_cylindrical.npz"
FS, W = 2048.0, 256


def fourier_golden(phi, dz, L1, L2, v, posz=0.0):
    """Fourier-engine SFAP on φ (the reference SFAP). Centred time axis.

    Includes the posz phase term (NMJ offset) exactly as emgforge.synthesis.engines.fourier."""
    zs = v * 1000.0 / FS
    pu = resample_centered_line(np.asarray(phi, float), delta_s_mm=dz, w_out=W,
                                delta_s_out_mm=zs)
    Ck, _ = compute_C_from_phi_z(pu, delta_s_mm=zs, apply_z_window=False)
    g = build_fourier_grids(w=W, fsamp=FS, v=v)
    ka, kb, kzt, kt = g['kalpha'], g['kbeta'], g['kz_t'], g['kt']
    sp, _ = build_spe2_iap_spectrum(w=W, fsamp=FS, v=v)
    pare = (np.exp(-1j*L1/2*ka)*L1*np.sinc(L1/2*ka/np.pi)
            - np.exp(1j*L2/2*kb)*L2*np.sinc(L2/2*kb/np.pi))
    E = pare*np.exp(1j*kzt*posz)*(np.ones((W, 1))@Ck.reshape(1, -1))
    E1 = (1.0/v)*(np.ones((W, 1))@sp.reshape(1, -1)).T*E*(1j*kzt)
    s = np.flip(radon_section(kt.T/(2*np.pi), kzt.T/(2*np.pi), E1.T, 0.0))
    t = build_time_vector_ms(w=W, fsamp=FS) - W/FS*1000.0/2
    return t, np.flip(s)


def gaussian_phi(sigma_mm, v):
    dz = v * 1000.0 / FS
    z = (np.arange(W) - W // 2) * dz
    return np.exp(-(z**2) / (2 * sigma_mm**2)), dz


def dog_phi(sig_pos_mm, sig_neg_mm, amp_neg, v):
    """Biphasic difference-of-Gaussians lead field (positive hump, negative tails)."""
    dz = v * 1000.0 / FS
    z = (np.arange(W) - W // 2) * dz
    phi = np.exp(-(z**2) / (2 * sig_pos_mm**2)) - amp_neg * np.exp(-(z**2) / (2 * sig_neg_mm**2))
    return phi, dz


# (name, phi_kind, params, L1, L2, v, posz)
CASES = [
    ("gauss_s12_sym",        "gauss", (12.0,),        60, 60,  4.0,   0.0),
    ("gauss_s8_sym",         "gauss", (8.0,),         60, 60,  4.0,   0.0),
    ("gauss_s5_sym",         "gauss", (5.0,),         60, 60,  4.0,   0.0),
    ("gauss_s18_deep",       "gauss", (18.0,),        60, 60,  4.0,   0.0),
    ("gauss_s8_asym40-100",  "gauss", (8.0,),         40, 100, 4.0,   0.0),
    ("gauss_s8_pm_asym",     "gauss", (8.0,),         64, 144, 3.26,  0.0),
    ("gauss_s8_offset+20",   "gauss", (8.0,),         60, 60,  4.0, +20.0),
    ("gauss_s8_offset-30",   "gauss", (8.0,),         60, 60,  4.0, -30.0),
    ("gauss_s8_v3",          "gauss", (8.0,),         60, 60,  3.0,   0.0),
    ("gauss_s8_v5",          "gauss", (8.0,),         60, 60,  5.0,   0.0),
    ("dog_biphasic_sym",     "dog",   (8.0, 22.0, 0.5), 60, 60, 4.0,  0.0),
    ("dog_biphasic_asym",    "dog",   (6.0, 25.0, 0.6), 50, 110, 4.0, 0.0),
]


def main():
    out = {}
    names = []
    for name, kind, p, L1, L2, v, posz in CASES:
        if kind == "gauss":
            phi, dz = gaussian_phi(p[0], v)
        else:
            phi, dz = dog_phi(p[0], p[1], p[2], v)
        t, golden = fourier_golden(phi, dz, L1, L2, v, posz)
        names.append(name)
        out[f"{name}__phi"] = phi.astype(np.float32)
        out[f"{name}__dz_mm"] = np.float32(dz)
        out[f"{name}__L1"] = np.float32(L1)
        out[f"{name}__L2"] = np.float32(L2)
        out[f"{name}__v"] = np.float32(v)
        out[f"{name}__posz"] = np.float32(posz)
        out[f"{name}__golden_t_ms"] = t.astype(np.float32)
        out[f"{name}__golden_sfap"] = golden.astype(np.float32)
        print(f"  {name:22s} L1={L1:>4} L2={L2:>4} v={v:>4} posz={posz:>+6.1f} "
              f"| reference trough@{t[np.argmin(golden)]:.1f}ms ptp={golden.ptp():.2e}")
    out["case_names"] = np.array(names)
    out["fsamp_hz"] = np.float32(FS)
    out["w_samples"] = np.int32(W)
    out["reference"] = np.array("emgforge.synthesis.engines.fourier (validated r=0.997 vs MATLAB)")
    np.savez_compressed(GOLDEN, **out)
    print(f"\n✓ {GOLDEN}  ({len(names)} cases)")


if __name__ == "__main__":
    main()
