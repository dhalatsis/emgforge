"""MRI tier: does a fresh forearm_WR solve reproduce the NeuroDec (PM) lead fields?

The NeuroDec digital twin (Maksymenko et al. 2023) distributed, for subject WR, the
reciprocal lead field `phi_raw` of each of its skin electrodes sampled along 100 fibre
paths. We re-solve the same mesh with our solver (z-aligned muscle anisotropy, as the
saved fields were produced), sample φ along the same paths, and report per electrode the
SIGNED correlation and the least-squares amplitude ratio fresh/saved.

Needs the (non-redistributed) reference data and the sanity-suite glue under
``_results/sanity/`` (see ``_results/sanity/mri/neurodec_baseline/validate_wr.py``);
writes ``docs/validation/neurodec_wr.json`` as the committed artefact.

Run:  python scripts/validation/mri_neurodec_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "_results/sanity/mri/neurodec_baseline"))
sys.path.insert(0, str(ROOT / "_results/sanity/lib"))
import pm_lib as pm          # noqa: E402
import fem_mri as fm         # noqa: E402


def main():
    phi_all, Lp, Ld, dz, v, nd_all, t_nd, el, fib = pm.load_pm()
    zz = el[:, 2]
    pts = fib.reshape(-1, 3)
    n_fib, n_z = fib.shape[0], fib.shape[1]
    t0 = time.time()
    model = fm.make_model(mesh="forearm_WR", patch=False, fiber_aligned=False)
    t_model = time.time() - t0
    rows = []
    t0 = time.time()
    for e in range(len(el)):
        uh = model.solve_for_point(el[e].astype(float))
        fresh = fm.fibre_phi_path(model, uh, pts).reshape(n_fib, n_z)
        fin = np.isfinite(fresh).all(1)
        a = fresh[fin].ravel(); b = phi_all[e][fin].ravel()
        a0, b0 = a - a.mean(), b - b.mean()
        r = float((a0 @ b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-30))
        scale = float((a0 @ b0) / (b0 @ b0 + 1e-30))              # fresh ≈ scale · saved
        rows.append(dict(electrode=int(e), z_mm=float(zz[e]), r_signed=r, amp_ratio_fresh_over_saved=scale,
                         fibres_in_mesh=int(fin.sum())))
        print(f"  e={e:3d} z={zz[e]:6.1f}  r={r:+.4f}  scale={scale:+.3f}  ({int(fin.sum())}/{n_fib} fibres)", flush=True)
    t_solve = (time.time() - t0) / len(el)
    rs = np.array([w["r_signed"] for w in rows]); sc = np.array([w["amp_ratio_fresh_over_saved"] for w in rows])
    summary = dict(n_electrodes=len(el), n_fibres=n_fib, samples_per_fibre=n_z,
                   r_signed_mean=float(rs.mean()), r_signed_min=float(rs.min()), r_signed_max=float(rs.max()),
                   abs_r_mean=float(np.abs(rs).mean()), abs_r_min=float(np.abs(rs).min()),
                   amp_ratio_median=float(np.median(sc)), amp_ratio_min=float(sc.min()), amp_ratio_max=float(sc.max()),
                   model_build_s=t_model, solve_and_sample_s_per_electrode=t_solve,
                   mesh="forearm_WR.msh", sigma_mode="uniform z-aligned muscle anisotropy (as the saved fields)",
                   reference="NeuroDec / PM phi_raw_per_electrode for subject WR (Maksymenko et al. 2023)")
    print(json.dumps(summary, indent=1))
    out = ROOT / "docs/validation/neurodec_wr.json"
    out.write_text(json.dumps(dict(summary=summary, per_electrode=rows), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
