"""Phase 1.3 / 2.3 — score a learned VC against the FROZEN MUAP benchmark.

The referee. A model is queried at each benchmark config's MU fibre points -> phi_pred
-> field_to_muap -> compared to the frozen FEM truth. Ranks by MUAP quality, NOT by phi
error (R^2=0.999 already coexisted with bad MUAPs; SFAP is proportional to phi'').

dolfinx-free by construction (torch + numpy + emgforge.synthesis) -> runs on the cluster.

Run: PYTHONPATH=src python scripts/neural_field/04_score_bench.py --ckpt _results/neural_field/cyl_mlp.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from emgforge.fem.geometry import ParametricGeometry
from emgforge.synthesis import FibreBed, SpatialConfig, field_to_muap
from emgforge.synthesis.metrics import align_score, jaggedness, lobe_metrics
from neural_field.models import MLP, SIREN

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"
# must match 01_build_muap_bench.py exactly
CYL = dict(r_bone=10.0, r_muscle=35.0, r_fat=38.0, r_skin=40.0, length=240.0)
N_FIB, MU_RADIUS_MM, MU_SEED = 20, 2.0, 7
NZ, DZ, Z_CENTROID = 200, 1.07, 120.0
LP, LD, V = 65.0, 148.0, 4.0
POSZ = (LP - LD) / 2.0
SPCFG = SpatialConfig(denoise="monopole", denoise_n_poles=3, fiber_window="one_sided",
                      tukey_alpha=0.25, csd_derivative=2, upsample_factor=2, fsamp=2048.0,
                      w=256, edge_taper_left=5, edge_taper_right=10, t_start_ms=-10.0, v=V)


def mu_fibre_paths(g, depth):
    x0, y0 = g.fibre_xy_below_skin(depth, 0.0)
    rng = np.random.default_rng(MU_SEED)
    r = MU_RADIUS_MM * np.sqrt(rng.uniform(0, 1, N_FIB))
    a = rng.uniform(0, 2 * np.pi, N_FIB)
    return np.asarray([g.fibre_points(x0 + ri * np.cos(ai), y0 + ri * np.sin(ai),
                                      Z_CENTROID, NZ, DZ)[0] for ri, ai in zip(r, a)])


def load(ckpt, dev):
    c = torch.load(ckpt, map_location=dev)
    net = (MLP(in_dim=6, hidden_dim=256, n_layers=6, out_dim=1, use_fourier=True,
               n_frequencies=64, fourier_scale=3.0) if c["arch"] == "mlp"
           else SIREN(in_dim=6, hidden_dim=256, n_layers=6, out_dim=1, omega_0=30.0))
    net.load_state_dict(c["state"]); net.to(dev).eval()
    return net, c


def predict_phi(net, c, dev, pts, elec):
    """phi at `pts` (P,3) for one electrode (3,) — inverts the training transform.
    The nets take (coords, condition) separately, condition broadcast across points.
    'phir' mode fits phi*r, so the inverse needs r = |point - electrode|."""
    cs, co, tf = c["cs"], c["co"], c["tf"]
    p = ((pts - co) / cs).astype(np.float32)
    e = np.broadcast_to(((elec - co) / cs).astype(np.float32), p.shape)
    P = torch.from_numpy(p).to(dev)
    E = torch.from_numpy(np.ascontiguousarray(e)).to(dev)
    with torch.no_grad():
        y = net(P, E).reshape(-1).cpu().numpy()
    u = y * tf["sd"] + tf["mu"]
    mode = tf.get("mode", "asinh")
    if mode in ("phir", "phir0"):
        r = np.linalg.norm(pts - elec, axis=-1)
        return u / (np.maximum(r, tf.get("r_min", 0.5)) if mode == "phir"
                    else r + tf.get("r0", 3.0))
    if mode == "raw":
        return u
    return np.sinh(u) * tf["s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "cyl_mlp.pt"))
    ap.add_argument("--bench", default=str(OUT / "muap_bench_cyl.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = np.load(a.bench)
    t_ms, Wt, cfg = b["t_ms"], b["muap"], b["cfg"]
    net, c = load(a.ckpt, dev)
    g = ParametricGeometry(**CYL)
    beds = {d: mu_fibre_paths(g, d) for d in np.unique(cfg[:, 2])}
    sb = FibreBed.from_arrays(np.full(N_FIB, DZ), np.full(N_FIB, LP), np.full(N_FIB, LD),
                              np.full(N_FIB, POSZ), np.full(N_FIB, V))

    rows = []
    for k, (th, z, dep) in enumerate(cfg):
        elec = g.electrode_on_skin(th, z)
        paths = beds[dep]
        phi = np.array([predict_phi(net, c, dev, p, elec) for p in paths])
        m = field_to_muap(phi, sb, SPCFG).muap
        _, r = align_score(t_ms, Wt[k], t_ms, m)
        tr_t, _, eof_t = lobe_metrics(t_ms, Wt[k], 12.0)
        tr_p, _, eof_p = lobe_metrics(t_ms, m, 12.0)
        rows.append(dict(th=th, z=z, dep=dep, r=r,
                         p2p_ratio=float(m.ptp() / (Wt[k].ptp() + 1e-30)),
                         d_jag=float(jaggedness(m) - b["jaggedness"][k]),
                         d_eof=float(eof_p - eof_t),
                         d_lat=float(t_ms[np.argmax(np.abs(m))] - b["latency"][k])))

    print(f"\n=========== MUAP BENCHMARK SCORE — {Path(a.ckpt).name} ===========")
    print(f"{'θ':>4}{'z':>5}{'dep':>5} | {'r':>7} {'p2p×':>7} {'Δjag':>8} {'ΔEOF':>7} {'Δlat ms':>8}")
    print("-" * 56)
    for x in rows:
        print(f"{x['th']:4.0f}{x['z']:5.0f}{x['dep']:5.0f} | {x['r']:+7.3f} {x['p2p_ratio']:7.2f} "
              f"{x['d_jag']:+8.4f} {x['d_eof']:+7.2f} {x['d_lat']:+8.1f}")
    R = np.array([x["r"] for x in rows])
    A = b["p2p"] * 1e6                       # truth amplitude, µV
    print("-" * 56)
    print(f"  MUAP corr : median {np.median(R):+.3f} · mean {R.mean():+.3f} · "
          f"min {R.min():+.3f} · ≥0.94: {(R >= 0.94).sum()}/{len(R)}")
    # --- amplitude-aware scoring -------------------------------------------------
    # A plain median treats a 0.14µV MUAP as equal to a 9.6µV one, but real surface-EMG
    # noise is ~1-5µV RMS, so most of the weak configs are UNDETECTABLE in practice.
    # Weight by truth amplitude, and gate on the detectable subset.
    w = A / A.sum()
    wmean = float((w * R).sum())
    det = A >= 1.0
    print(f"  amp-weighted mean r : {wmean:+.3f}   (w ∝ truth amplitude)")
    if det.any():
        print(f"  DETECTABLE (>1µV, n={det.sum()}) : median r {np.median(R[det]):+.3f} · "
              f"min {R[det].min():+.3f} · p2p med "
              f"{np.median([x['p2p_ratio'] for x, m in zip(rows, det) if m]):.2f}")
    print(f"  p2p ratio : median {np.median([x['p2p_ratio'] for x in rows]):.2f} (1.0 = perfect)")
    print(f"  Δjagged   : median {np.median([x['d_jag'] for x in rows]):+.4f} (0 = as smooth as FEM)")
    print(f"\n  phi rel-L2 on held-out electrodes (from training): {c.get('rel_l2', float('nan')):.4f}")
    print(f"  GATE 2 (plain median >= 0.94)      : "
          f"{'PASS ✅' if np.median(R) >= 0.94 else 'FAIL ❌'}")
    print(f"  GATE 2b (amp-weighted mean >= 0.94) : "
          f"{'PASS ✅' if wmean >= 0.94 else 'FAIL ❌'}   <- the honest gate")
    np.savez(OUT / f"score_{Path(a.ckpt).stem}.npz", **{k: np.array([x[k] for x in rows])
                                                        for k in rows[0]})


if __name__ == "__main__":
    main()
