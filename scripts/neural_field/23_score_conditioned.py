"""Phase 5.4 — score the ANATOMY-conditioned VC on the held-out-anatomy benchmark.

Same referee logic as the cylinder scorer (04_score_bench) but the model is queried with a
5-D condition [electrode_xyz, r_fat, pennation], and each benchmark config carries its own
(r_fat, pennation). A pass here means the network interpolated the VC field to anatomies it
never trained on.

dolfinx-free (torch + numpy + emgforge.synthesis) -> runs on the GPU node.

Run: PYTHONPATH=src python scripts/neural_field/23_score_conditioned.py \
        --ckpt _results/neural_field/cyl_cond_mlp_raw.pt
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
B = import_module("01_build_muap_bench")         # mu_fibre_paths, SPCFG, fibre consts
G = import_module("20_gen_cyl_anatomy")           # geom(r_fat)
from neural_field.models import MLP, SIREN
from emgforge.synthesis import FibreBed, field_to_muap
from emgforge.synthesis.metrics import align_score, jaggedness, lobe_metrics

OUT = Path(__file__).resolve().parents[2] / "_results/neural_field"


def load(ckpt, dev):
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    cd = int(c.get("cond_dim", 3)); mode = "film" if c.get("film") else "concat"
    mk = (lambda: MLP(in_dim=3 + cd, hidden_dim=256, n_layers=6, out_dim=1, use_fourier=True,
                      n_frequencies=64, fourier_scale=3.0, conditioning_mode=mode,
                      condition_dim=cd)) if c["arch"] == "mlp" else \
         (lambda: SIREN(in_dim=3 + cd, hidden_dim=256, n_layers=6, out_dim=1, omega_0=30.0,
                        conditioning_mode=mode, condition_dim=cd))
    net = mk(); net.load_state_dict(c["state"]); net.to(dev).eval()
    return net, c


def predict_phi(net, c, dev, pts, elec, r_fat, penn):
    """phi at `pts` (P,3) for one (electrode, anatomy). Condition = [elec_norm, fat_norm, pen_norm]."""
    co, cs, tf = c["co"], c["cs"], c["tf"]
    p = ((pts - co) / cs).astype(np.float32)
    anat = np.array([(r_fat - c["fat_c"]) / c["fat_s"], (penn - c["pen_c"]) / c["pen_s"]], np.float32)
    cond = np.concatenate([(elec - co) / cs, anat]).astype(np.float32)
    cond = np.broadcast_to(cond, (p.shape[0], cond.shape[0]))
    with torch.no_grad():
        y = net(torch.from_numpy(p).to(dev),
                torch.from_numpy(np.ascontiguousarray(cond)).to(dev)).reshape(-1).cpu().numpy()
    u = y * tf["sd"] + tf["mu"]
    mode = tf.get("mode", "raw")
    if mode == "raw":
        return u
    if mode in ("phir", "phir0"):
        r = np.linalg.norm(pts - elec, axis=-1)
        return u / (np.maximum(r, tf.get("r_min", 0.5)) if mode == "phir" else r + tf.get("r0", 3.0))
    return np.sinh(u) * tf["s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench", default=str(OUT / "muap_bench_anatomy.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = np.load(a.bench)
    t_ms, Wt, cfg = b["t_ms"], b["muap"], b["cfg"]        # cfg: r_fat, penn, theta, z, depth
    A = b["p2p"] * 1e6
    net, c = load(a.ckpt, dev)
    sb = FibreBed.from_arrays(np.full(B.N_FIB, B.DZ), np.full(B.N_FIB, B.LP),
                              np.full(B.N_FIB, B.LD), np.full(B.N_FIB, B.POSZ),
                              np.full(B.N_FIB, B.V))
    # fibre beds depend on r_fat (skin radius); cache per unique fat
    geoms, beds = {}, {}
    for r_fat, penn, th, z, dep in cfg:
        if r_fat not in geoms:
            geoms[r_fat] = G.geom(r_fat); beds[r_fat] = {}
        if dep not in beds[r_fat]:
            beds[r_fat][dep] = B.mu_fibre_paths(geoms[r_fat], dep)

    rows = []
    for k in range(len(cfg)):
        r_fat, penn, th, z, dep = cfg[k]
        g = geoms[r_fat]
        elec = g.electrode_on_skin(th, z)
        phi = np.array([predict_phi(net, c, dev, p, elec, r_fat, penn) for p in beds[r_fat][dep]])
        m = field_to_muap(phi, sb, B.SPCFG).muap
        r = align_score(t_ms, Wt[k], t_ms, m)[1]
        rows.append(dict(r_fat=r_fat, penn=penn, r=r,
                         p2p_ratio=float(np.ptp(m) / (np.ptp(Wt[k]) + 1e-30)),
                         d_jag=float(jaggedness(m) - b["jaggedness"][k])))

    R = np.array([x["r"] for x in rows]); w = A / A.sum(); det = A >= 1.0
    print(f"\n===== ANATOMY BENCHMARK — {Path(a.ckpt).name} =====")
    print(f"{'fat':>5}{'pen':>6}{'µV':>9} | {'r':>8}{'p2p×':>7}")
    print("-" * 38)
    for x, amp in sorted(zip(rows, A), key=lambda t: -t[1]):
        print(f"{x['r_fat']:5.0f}{x['penn']:6.1f}{amp:9.2f} | {x['r']:+8.3f}{x['p2p_ratio']:7.2f}")
    print("-" * 38)
    print(f"  plain median r      : {np.median(R):+.3f} · ≥0.94 {(R>=0.94).sum()}/{len(R)}")
    print(f"  amp-weighted mean r : {float((w*R).sum()):+.3f}")
    print(f"  DETECTABLE >1µV (n={det.sum()}) : median {np.median(R[det]):+.3f} · "
          f"min {R[det].min():+.3f} · p2p med {np.median([x['p2p_ratio'] for x,m in zip(rows,det) if m]):.2f}")
    print(f"  GATE 5 (amp-weighted ≥0.94): {'PASS ✅' if float((w*R).sum())>=0.94 else 'FAIL ❌'}")
    np.savez(OUT / f"scorecond_{Path(a.ckpt).stem}.npz",
             **{k: np.array([x[k] for x in rows]) for k in rows[0]}, amp=A)


if __name__ == "__main__":
    main()
