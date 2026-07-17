"""Phase 2.3 — train the learned VC on the cylinder electrode sweep.

u(x,y,z | electrode_xyz) -> phi. Deliberately dependency-light: torch + numpy only, NO
dolfinx — this is the trainer that must run on the GPU cluster unchanged.

phi spans orders of magnitude (~1/r near the source), so we fit an asinh-compressed,
standardised target and invert at inference; fitting raw phi lets the near-source points
dominate the loss and starves the far field the MUAPs actually need.

Run: PYTHONPATH=src python scripts/neural_field/03_train_cyl.py [--arch mlp|siren]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neural_field.models import MLP, SIREN

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/neural_field"


R_MIN = 0.5   # mm — guard the 1/r inverse near the source
R0 = 3.0      # mm — the Gaussian source sigma: phi ~ 1/(r+R0), NOT 1/r


class PhiTransform:
    """Target transform. Two modes:

    'phir' (default) — fit phi*r, standardised. The physics: phi ~ C/r dominates, and
        measured |phi|*r is flat (0.26/0.30/0.28/0.23 across r=2-40mm) while |phi| itself
        varies 8.6x. Factoring the known 1/r out flattens the target ~7x, so neither the
        near-source peak nor the far field dominates the loss.
    'asinh' — the original. KEPT ONLY FOR COMPARISON: it is wrong for this data. The range
        is 30x (~1.5 decades), not the 6+ decades asinh is for, and d(asinh)/dphi collapses
        21x at the peak (0.033 vs 0.707 at the median) -> the loss cannot feel the tallest
        values, then sinh() amplifies the residual. That is the near-field amplitude bug.
    """

    def __init__(self, phi, r, mode="phir"):
        self.mode = mode
        if mode == "phir":
            y = phi * np.maximum(r, R_MIN)
            self.s = 1.0
        elif mode == "phir0":
            # phi*(r+R0). Keeps phir's flat target, but the inverse divides by (r+R0) >= R0,
            # so it cannot amplify near the source the way 1/r does. Physically right: the
            # source is a Gaussian of sigma=R0, so phi ~ 1/(r+R0).
            y = phi * (r + R0)
            self.s = 1.0
        elif mode == "raw":
            # no transform at all — the range is only 30x (~1.5 decades), so maybe none is needed
            y = phi.copy()
            self.s = 1.0
        else:
            self.s = float(np.percentile(np.abs(phi), 50)) or 1.0
            y = np.arcsinh(phi / self.s)
        self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0

    def fwd(self, phi, r):
        if self.mode == "phir":
            y = phi * np.maximum(r, R_MIN)
        elif self.mode == "phir0":
            y = phi * (r + R0)
        elif self.mode == "raw":
            y = phi
        else:
            y = np.arcsinh(phi / self.s)
        return (y - self.mu) / self.sd

    def inv(self, y, r):
        u = y * self.sd + self.mu
        if self.mode == "phir":
            return u / np.maximum(r, R_MIN)
        if self.mode == "phir0":
            return u / (r + R0)
        if self.mode == "raw":
            return u
        return np.sinh(u) * self.s

    def dump(self):
        return dict(s=self.s, mu=self.mu, sd=self.sd, mode=self.mode, r_min=R_MIN, r0=R0)


def build(arch, in_dim=6):
    if arch == "mlp":     # the prior work's best MUAP performer
        return MLP(in_dim=in_dim, hidden_dim=256, n_layers=6, out_dim=1,
                   use_fourier=True, n_frequencies=64, fourier_scale=3.0)
    return SIREN(in_dim=in_dim, hidden_dim=256, n_layers=6, out_dim=1, omega_0=30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "cyl_elec_64.npz"))
    ap.add_argument("--arch", default="mlp", choices=["mlp", "siren"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--target", default="raw", choices=["phir", "phir0", "raw", "asinh"])
    # loss weight = |phi|^wpow. The MUAP is driven by phi'' where phi is high+sharp, and
    # high-|phi| points are RARE (top decile, near the electrode), so plain MSE under-weights
    # them by scarcity. wpow>0 counteracts that: be accurate where it matters.
    ap.add_argument("--wpow", type=float, default=0.0)
    # Data-scaling sweep: take the first n_elec electrodes. Free — the dataset's electrodes
    # are already i.i.d. random, so a prefix IS a valid smaller draw, no new FEM needed. The
    # referee (frozen benchmark) is held out by construction at every N, so N is comparable.
    ap.add_argument("--n_elec", type=int, default=0, help="0 = use all")
    ap.add_argument("--tag", default="", help="checkpoint suffix")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    d = np.load(a.data)
    P, E, PHI = d["points"], d["electrodes"], d["phi"]        # (M,3) (N,3) (N,M)
    if a.n_elec and a.n_elec < len(E):
        E, PHI = E[:a.n_elec], PHI[:a.n_elec]
        d = {k: (d[k][:a.n_elec] if k in ("near_points", "near_phi") else d[k])
             for k in d.files}
        print(f"  N-sweep: using {a.n_elec} electrodes")
    N, M = PHI.shape
    # Hybrid dataset: a shared/cached coverage set + a per-electrode NEAR-FIELD set.
    # Uniform-in-volume sampling puts only ~0.1% of points within 10mm of the electrode,
    # starving exactly the high-phi regime the MUAP depends on. The near set fixes that.
    PTS = np.broadcast_to(P[None], (N, M, 3))                 # (N,M,3) shared
    if "near_points" in d:
        PTS = np.concatenate([PTS, d["near_points"]], axis=1)         # (N,M+K,3)
        PHI = np.concatenate([PHI, d["near_phi"]], axis=1)            # (N,M+K)
        print(f"  hybrid: {M} shared + {d['near_points'].shape[1]} near-field pts/electrode")
    M = PHI.shape[1]
    # split by ELECTRODE (never leak an electrode across the split)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = max(1, int(a.val_frac * N))
    va, tr = perm[:n_val], perm[n_val:]
    print(f"{N} electrodes × {M} pts · train {len(tr)} / val {len(va)} electrodes · {dev}")

    # r = |point - electrode| per (electrode, point) pair — needed by the phir target
    R = np.linalg.norm(PTS - E[:, None, :], axis=2)                    # (N, M)
    tf = PhiTransform(PHI[tr], R[tr], mode=a.target)
    print(f"  target: {a.target} · loss weight |phi|^{a.wpow}")
    # normalise coords to ~[-1,1] — derived FROM THE DATA, not hardcoded, so the same
    # trainer works on the cylinder (centred at origin) and on MRI/WR (offset anatomy).
    _all = np.concatenate([PTS.reshape(-1, 3), E], 0)
    lo, hi = _all.min(0), _all.max(0)
    co = ((hi + lo) / 2).astype(np.float32)
    cs = np.maximum((hi - lo) / 2, 1e-6).astype(np.float32)
    print(f"  coord norm: centre {np.round(co,1)} scale {np.round(cs,1)}")

    def make(idx):
        """The ported nets take (coords, condition) separately, not a concatenated 6-D input."""
        c = np.repeat(((E[idx] - co) / cs)[:, None, :], M, 1).reshape(-1, 3)   # (n*M,3) electrode
        p = ((PTS[idx] - co) / cs).reshape(-1, 3)                     # (n*M,3) point
        Y = tf.fwd(PHI[idx], R[idx]).reshape(-1, 1).astype(np.float32)
        w = np.abs(PHI[idx]).reshape(-1) ** a.wpow if a.wpow > 0 else np.ones(Y.shape[0])
        w = (w / w.mean()).astype(np.float32)          # mean-1 so lr stays comparable
        return (torch.from_numpy(p.astype(np.float32)),
                torch.from_numpy(c.astype(np.float32)), torch.from_numpy(Y),
                R[idx].reshape(-1).astype(np.float32),
                torch.from_numpy(w).reshape(-1, 1))

    Ptr, Ctr, Ytr, Rtr, Wtr = make(tr); Pva, Cva, Yva, Rva, _ = make(va)
    Pva, Cva, Yva = Pva.to(dev), Cva.to(dev), Yva.to(dev)
    net = build(a.arch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.MSELoss()
    n = Ptr.shape[0]
    print(f"  {sum(p.numel() for p in net.parameters())/1e3:.0f}K params · {n/1e6:.1f}M samples")

    def val_chunked(chunk=200_000):
        """Val set is large; evaluate in chunks to stay inside GPU memory."""
        outs = []
        with torch.no_grad():
            for i in range(0, Pva.shape[0], chunk):
                outs.append(net(Pva[i:i + chunk], Cva[i:i + chunk]).reshape(-1))
        return torch.cat(outs)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        idx = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            b = idx[i:i + a.batch]
            pb, cb, yb, wb = (Ptr[b].to(dev, non_blocking=True), Ctr[b].to(dev, non_blocking=True),
                              Ytr[b].to(dev, non_blocking=True), Wtr[b].to(dev, non_blocking=True))
            opt.zero_grad()
            l = (wb * (net(pb, cb).reshape(-1, 1) - yb) ** 2).mean()
            l.backward(); opt.step()
            tot += l.item() * len(b)
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            net.eval()
            yv = val_chunked()
            vl = float(((yv - Yva.reshape(-1)) ** 2).mean())
            yp = tf.inv(yv.cpu().numpy(), Rva); yt = tf.inv(Yva.cpu().numpy().ravel(), Rva)
            rel = np.linalg.norm(yp - yt) / np.linalg.norm(yt)
            print(f"  ep {ep+1:4d} train {tot/n:.5f} val {vl:.5f} relL2(phi) {rel:.4f} "
                  f"({time.time()-t0:.0f}s)")
    ck = OUT / (f"cyl_{a.arch}_{a.target}" + (f"_w{a.wpow:g}" if a.wpow else "")
                + (f"_{a.tag}" if a.tag else "") + ".pt")
    torch.save(dict(state=net.state_dict(), arch=a.arch, tf=tf.dump(),
                    cs=cs, co=co, val_elec=va, rel_l2=float(rel)), ck)
    print(f"\nwrote {ck} · final rel-L2(phi) on held-out electrodes = {rel:.4f}")


if __name__ == "__main__":
    main()
