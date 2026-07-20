"""Phase 5.3 — train the ANATOMY-conditioned learned VC.

phi = f(query_xyz, electrode_xyz, r_fat, pennation). Globs many single-anatomy datasets
(20_gen_cyl_anatomy) and trains ONE network conditioned on the anatomy, so it can be queried
at fat/pennation values it never saw. The referee is muap_bench_anatomy (held-out anatomies).

Reuses the validated Gate-4 recipe: MLP+Fourier, target `raw`, hybrid near-field, coord-norm
from data. The only change is the condition vector: 3-D (electrode) -> 5-D (electrode + fat +
pennation). Condition is concatenated (the validated mode); `--film` switches to FiLM.

Dependency-light (torch + numpy, NO dolfinx) so it runs unchanged on the GPU node.

Run: PYTHONPATH=src python scripts/neural_field/22_train_conditioned.py \
        --data_glob '_results/neural_field/anat/cyl_fat*_pen*_n*.npz'
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
T = import_module("03_train_cyl")               # PhiTransform
from neural_field.models import MLP, SIREN

OUT = Path(__file__).resolve().parents[2] / "_results/neural_field"


def load_anatomy(fp):
    """One anatomy -> (coords (E,M,3), phi (E,M), electrodes (E,3), r_fat, pennation)."""
    d = np.load(fp)
    P, E, PHI = d["points"], d["electrodes"], d["phi"]                 # (M,3)(E,3)(E,M)
    n_el, M = PHI.shape
    pts = np.broadcast_to(P[None], (n_el, M, 3))
    if "near_points" in d:
        pts = np.concatenate([pts, d["near_points"]], 1)
        PHI = np.concatenate([PHI, d["near_phi"]], 1)
    return (pts.astype(np.float32), PHI.astype(np.float32), E.astype(np.float32),
            float(d["r_fat"]), float(d["pennation"]))


def build(arch, cond_dim, film):
    mode = "film" if film else "concat"
    if arch == "mlp":
        return MLP(in_dim=3 + cond_dim, hidden_dim=256, n_layers=6, out_dim=1,
                   use_fourier=True, n_frequencies=64, fourier_scale=3.0,
                   conditioning_mode=mode, condition_dim=cond_dim)
    return SIREN(in_dim=3 + cond_dim, hidden_dim=256, n_layers=6, out_dim=1, omega_0=30.0,
                 conditioning_mode=mode, condition_dim=cond_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default=str(OUT / "anat/cyl_fat*_pen*_n*.npz"))
    ap.add_argument("--arch", default="mlp", choices=["mlp", "siren"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--target", default="raw", choices=["phir", "phir0", "raw", "asinh"])
    ap.add_argument("--film", action="store_true", help="FiLM conditioning instead of concat")
    # anatomy-normalisation ranges (must span the training grid; keep FIXED so a checkpoint's
    # condition scaling is reproducible and matches the scorer)
    ap.add_argument("--fat_lo", type=float, default=36.0)
    ap.add_argument("--fat_hi", type=float, default=46.0)
    ap.add_argument("--pen_lo", type=float, default=0.0)
    ap.add_argument("--pen_hi", type=float, default=25.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    files = sorted(glob.glob(a.data_glob))
    if not files:
        raise SystemExit(f"no anatomy datasets match {a.data_glob!r}")
    anats = [load_anatomy(f) for f in files]
    print(f"{len(anats)} anatomies (r_fat, pennation): "
          f"{sorted(set((round(x[3], 1), round(x[4], 1)) for x in anats))}")

    # coord + electrode norm from ALL points/electrodes (data-derived, as Gate 4)
    allpts = np.concatenate([x[0].reshape(-1, 3) for x in anats] +
                            [x[2] for x in anats], 0)
    lo, hi = allpts.min(0), allpts.max(0)
    co = ((hi + lo) / 2).astype(np.float32); cs = np.maximum((hi - lo) / 2, 1e-6).astype(np.float32)
    fat_c, fat_s = (a.fat_lo + a.fat_hi) / 2, max((a.fat_hi - a.fat_lo) / 2, 1e-6)
    pen_c, pen_s = (a.pen_lo + a.pen_hi) / 2, max((a.pen_hi - a.pen_lo) / 2, 1e-6)

    # flatten to samples; split by (anatomy, electrode) so no electrode leaks across train/val
    rng = np.random.default_rng(a.seed)
    Pl, Cl, Yl, Rl, split = [], [], [], [], []
    tf = None
    for ai, (pts, phi, E, r_fat, penn) in enumerate(anats):
        n_el, M, _ = pts.shape
        R = np.linalg.norm(pts - E[:, None, :], axis=2)               # (n_el, M)
        if tf is None:
            tf = T.PhiTransform(phi, R, mode=a.target)
        cond_el = (E - co) / cs                                        # (n_el, 3)
        anat_vec = np.array([(r_fat - fat_c) / fat_s, (penn - pen_c) / pen_s], np.float32)
        for e in range(n_el):
            p = (pts[e] - co) / cs                                     # (M,3)
            c = np.concatenate([np.repeat(cond_el[e][None], M, 0),
                                np.repeat(anat_vec[None], M, 0)], 1)    # (M,5)
            Pl.append(p.astype(np.float32)); Cl.append(c.astype(np.float32))
            Yl.append(tf.fwd(phi[e], R[e]).astype(np.float32)); Rl.append(R[e].astype(np.float32))
            split.append(rng.uniform() < a.val_frac)
    P = np.concatenate(Pl); C = np.concatenate(Cl)
    Y = np.concatenate(Yl)[:, None]; Rr = np.concatenate(Rl)
    isval = np.repeat(split, [x.shape[0] for x in Pl])
    tr, va = ~isval, isval
    print(f"  {P.shape[0]/1e6:.1f}M samples · cond_dim 5 · target {a.target} · {dev} · "
          f"val {va.mean()*100:.0f}%")

    Pt = torch.from_numpy(P[tr]); Ct = torch.from_numpy(C[tr]); Yt = torch.from_numpy(Y[tr])
    Pv = torch.from_numpy(P[va]).to(dev); Cv = torch.from_numpy(C[va]).to(dev)
    Yv = torch.from_numpy(Y[va]).to(dev); Rv = Rr[va]
    net = build(a.arch, 5, a.film).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n = Pt.shape[0]
    print(f"  {sum(p.numel() for p in net.parameters())/1e3:.0f}K params")

    def val_rel():
        outs = []
        with torch.no_grad():
            for i in range(0, Pv.shape[0], 200_000):
                outs.append(net(Pv[i:i+200_000], Cv[i:i+200_000]).reshape(-1))
        yv = torch.cat(outs).cpu().numpy()
        yp = tf.inv(yv, Rv); yt = tf.inv(Yv.cpu().numpy().ravel(), Rv)
        return np.linalg.norm(yp - yt) / (np.linalg.norm(yt) + 1e-30)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); idx = torch.randperm(n); tot = 0.0
        for i in range(0, n, a.batch):
            b = idx[i:i+a.batch]
            pb, cb, yb = Pt[b].to(dev), Ct[b].to(dev), Yt[b].to(dev)
            opt.zero_grad()
            l = ((net(pb, cb).reshape(-1, 1) - yb) ** 2).mean()
            l.backward(); opt.step(); tot += l.item() * len(b)
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  ep {ep+1:4d} train {tot/n:.5f} relL2(phi) {val_rel():.4f} ({time.time()-t0:.0f}s)")
    rel = val_rel()
    ck = OUT / (f"cyl_cond_{a.arch}_{a.target}" + ("_film" if a.film else "")
                + (f"_{a.tag}" if a.tag else "") + ".pt")
    torch.save(dict(state=net.state_dict(), arch=a.arch, tf=tf.dump(), cs=cs, co=co,
                    cond_dim=5, film=a.film, rel_l2=float(rel),
                    fat_c=fat_c, fat_s=fat_s, pen_c=pen_c, pen_s=pen_s), ck)
    print(f"\nwrote {ck} · rel-L2(phi) held-out electrodes = {rel:.4f}")


if __name__ == "__main__":
    main()
