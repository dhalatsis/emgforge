"""Dataset D1 — the cylinder SFAP atlas.

radii {33,30,27,25,20,15} mm × fibre angle {0,10,20,30,45}° × NMJ offset {0,−20,−40} mm ×
electrode axial position {−40 … +40 step 10} mm: analytical (Farina 2004) and FEM lead
fields on the engine grid, the SFAP from direct line-source synthesis on each, and the
Farina-generator MUAP for the same case. Written to _results/paper/datasets/cylinder_sfap_atlas.{npz,json}.

Frame: z = 0 is the array centre; the fibre always spans [−60, +60] mm with its NMJ at
nmj_mm[j]; the electrode sits at z_el_mm[k] (a z-invariant cylinder, so electrode at +z ≡
lead field translated by +z). The Farina port places its NMJ at channel z = −zi, so it is
called with zi = −nmj to land in this frame.

Run from the repo root:  python paper/figures/make_dataset_d1.py   (~2 min)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "paper/figures")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.interpolate import CubicSpline

from f1_common import (COND, DZ, FS, G, ROOT, V, W, Z, ana_phi, farina, fem_phi, golden_cfg, record, sfap,
                       shift_phi)

T0 = time.time()
OUT = ROOT / "_results/paper/datasets"; OUT.mkdir(parents=True, exist_ok=True)
RADII = np.array([33.0, 30.0, 27.0, 25.0, 20.0, 15.0])
ANGLES = np.array([0.0, 10.0, 20.0, 30.0, 45.0])
NMJ = np.array([0.0, -20.0, -40.0])
ZEL = np.arange(-40.0, 40.1, 10.0)
L_HALF = 60.0
cfg = golden_cfg()
t_ms = np.arange(W) / FS * 1000.0 + cfg.t_start_ms

Nr, Nth, Nn, Ne = len(RADII), len(ANGLES), len(NMJ), len(ZEL)
phi_a = np.zeros((Nr, Nth, W), np.float32); phi_f = np.zeros((Nr, Nth, W), np.float32)
sf_a = np.zeros((Nr, Nth, Nn, Ne, W), np.float32); sf_f = np.zeros_like(sf_a); mu_F = np.zeros_like(sf_a)
frame_check = None
for i, r in enumerate(RADII):
    for j, th in enumerate(ANGLES):
        pa = ana_phi(r, distfib=th)[0]; pf = fem_phi(r, th)
        phi_a[i, j], phi_f[i, j] = pa, pf
        for m, nmj in enumerate(NMJ):
            len1, len2 = L_HALF + nmj, L_HALF - nmj          # fibre stays [−60, 60]
            tF, sF, zc = farina(depth=r, zi=-nmj, L1=L_HALF, L2=L_HALF, distfib=th, channels=Ne, dint=10.0)
            assert np.allclose(zc, ZEL)
            # the generator's physical axis is window-centred (−31.25 … +31 ms); put it on t_ms
            mu_F[i, j, m] = np.stack([CubicSpline(tF, s, extrapolate=False)(t_ms) for s in sF])
            mu_F[i, j, m] = np.nan_to_num(mu_F[i, j, m])
            for k, zel in enumerate(ZEL):
                sf_a[i, j, m, k] = sfap(shift_phi(pa, DZ, zel), DZ, len1, len2, nmj, cfg)[1]
                sf_f[i, j, m, k] = sfap(shift_phi(pf, DZ, zel), DZ, len1, len2, nmj, cfg)[1]
    print(f"  r = {r:g} mm done  ({time.time() - T0:.0f} s)", flush=True)

# frame check (long fibre so no end-of-fibre potential can be mistaken for the travelling wave):
# NMJ at −20 → the electrode at −40 is 20 mm away (front at 5 ms), the one at +40 is 60 mm away
# (15 ms). Both models must put the propagating lobe in that order.
pa30 = ana_phi(30.0)[0]
tp = [float(t_ms[np.argmin(sfap(shift_phi(pa30, DZ, zel), DZ, 80.0, 120.0, -20.0, cfg)[1])]) for zel in (-40.0, 40.0)]
tFl, sFl, zcl = farina(depth=30.0, zi=+20.0, L1=100.0, L2=100.0, channels=Ne, dint=10.0)
tF = [float(tFl[np.argmax(np.abs(sFl[k]))]) for k in (0, Ne - 1)]
frame_check = dict(nmj_mm=-20.0, electrodes_mm=[-40.0, 40.0], pipeline_neg_lobe_ms=tp,
                   farina_absmax_ms=tF, farina_call="farina(zi=+20, L1=L2=100)",
                   expected_wavefront_ms=[5.0, 15.0],
                   farina_native_t_axis_ms=[float(tFl[0]), float(tFl[-1])])
# the stored Farina MUAP must line up with the pipeline on the common axis (NMJ −20, r = 30, θ = 0)
k_el = int(np.argmin(np.abs(ZEL + 40.0)))
lag_chk = float(t_ms[np.argmax(np.abs(mu_F[1, 0, 1, k_el][t_ms < 7.0]))])
frame_check["stored_farina_absmax_below_7ms_at_el_m40_ms"] = lag_chk

np.savez_compressed(OUT / "cylinder_sfap_atlas.npz",
                    radius_mm=RADII, depth_below_skin_mm=G["r_skin"] - RADII, angle_deg=ANGLES, nmj_mm=NMJ,
                    z_el_mm=ZEL, t_ms=t_ms.astype(np.float32), z_mm=Z.astype(np.float32),
                    phi_analytical=phi_a, phi_fem=phi_f, sfap_analytical=sf_a, sfap_fem=sf_f, farina_muap=mu_F)
size_mb = (OUT / "cylinder_sfap_atlas.npz").stat().st_size / 1e6
runtime = time.time() - T0

recipe = asdict(cfg)
manifest = dict(
    name="cylinder_sfap_atlas", file="cylinder_sfap_atlas.npz", size_MB=round(size_mb, 2), dtype="float32",
    description="Single-fibre action potentials in the 4-layer cylinder (bone 10 / muscle 35 / fat 38 / skin 40 mm; "
                "Farina 2004 conductivities) from the analytical and the FEM lead field through direct line-source "
                "synthesis (the spatial engine's production recipe), plus the Farina-2004 generator MUAP for the same case.",
    axes=dict(radius_mm=RADII.tolist(), depth_below_skin_mm=(G["r_skin"] - RADII).tolist(), angle_deg=ANGLES.tolist(),
              nmj_mm=NMJ.tolist(), z_el_mm=ZEL.tolist(),
              t_ms=f"({W},) {t_ms[0]:g} … {t_ms[-1]:.2f} ms, fs = {FS:g} Hz, t = 0 when the NMJ fires",
              z_mm=f"({W},) centred lead-field grid, dz = {DZ:.4f} mm (= v/fs)"),
    arrays=dict(
        phi_analytical=dict(shape=list(phi_a.shape), dims="(radius, angle, z)", units="a.u. (Farina-2004 port normalisation, "
                            "contains an arbitrary constant)", note="electrode Ø10 mm disc on the skin at θ = 0, z = 0"),
        phi_fem=dict(shape=list(phi_f.shape), dims="(radius, angle, z)", units="V per unit reciprocal source (arbitrary constant)",
                     note="FEniCSx reciprocal solve, σ = 5 mm Gaussian electrode source on the skin at θ = 0; re-gridded from "
                          "the 0.5 mm validation cache (_results/validation/fem_cache/cyl_lines.npz)"),
        sfap_analytical=dict(shape=list(sf_a.shape), dims="(radius, angle, nmj, electrode, t)", units="a.u. (∝ V)",
                             note="direct recipe on phi_analytical translated to the electrode position"),
        sfap_fem=dict(shape=list(sf_f.shape), dims="(radius, angle, nmj, electrode, t)", units="a.u. (∝ V)",
                      note="direct recipe on phi_fem translated to the electrode position"),
        farina_muap=dict(shape=list(mu_F.shape), dims="(radius, angle, nmj, electrode, t)", units="a.u. (Farina port)",
                         note="farina(depth=r, zi=-nmj, L1=L2=60, distfib=angle, channels=9, dint=10), physical time, "
                              "cubic-spline resampled from the generator's window-centred axis (-31.25 … +31 ms) onto t_ms "
                              "(zero outside); the port's NMJ sits at channel z = -zi, hence zi = -nmj to share this frame. Its output "
                              "polarity is inverted and its IAP body leads by ~1-2.5 ms relative to the pipeline "
                              "(validation tiers A0.3, A2.3, B1)"),
    ),
    fibre=dict(span_mm=[-L_HALF, L_HALF], len1_len2_per_nmj={f"{n:g}": [L_HALF + n, L_HALF - n] for n in NMJ},
               v_m_per_s=V, iap="Rosenfalck V_m = 96 z^3 e^-z mV"),
    geometry_mm=G, conductivities_S_per_m=COND,
    recipe=dict(engine="emgforge.synthesis.engines.spatial.compute_sfap_spatial", config=recipe,
                steps=["monopole denoise of φ (3 free poles + offset)", "edge taper 5/10 samples", "2× cubic upsampling",
                       "CSD = σ_in π a² ∂²V_m/∂z² by numerical differentiation of the full bidirectional field",
                       "one-sided tendon window (α = 0.25)",
                       "SFAP = (CSD @ φ) dz, physical time (no 1/v; corrected 2026-09-15)"],
                amplitude_note="sfap_analytical / sfap_fem are on the corrected spatial engine (spurious /v removed "
                               "2026-09-15; engine/oracle amplitude ratio 0.994 at every v): ×v = ×4 larger than the "
                               "v1 atlas; farina_muap is unchanged"),
    frame_check=frame_check, runtime_s=round(runtime, 1), generator="paper/figures/make_dataset_d1.py",
)
(OUT / "cylinder_sfap_atlas.json").write_text(json.dumps(manifest, indent=1))
print(f"wrote {OUT / 'cylinder_sfap_atlas.npz'}  ({size_mb:.2f} MB, {runtime:.0f} s)")
print("frame check:", frame_check)
record("dataset_d1", dict(file=str(OUT / "cylinder_sfap_atlas.npz"), size_MB=size_mb, runtime_s=runtime,
                          shapes={k: list(v.shape) for k, v in dict(phi_analytical=phi_a, phi_fem=phi_f, sfap_analytical=sf_a,
                                                                    sfap_fem=sf_f, farina_muap=mu_F).items()},
                          n_sfap_calls=2 * Nr * Nth * Nn * Ne, n_farina_calls=Nr * Nth * Nn, frame_check=frame_check))
