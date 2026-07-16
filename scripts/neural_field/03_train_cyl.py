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


class PhiTransform:
    """asinh(phi/s)/k, standardised. Compresses the 1/r dynamic range, invertible."""

    def __init__(self, phi):
        self.s = float(np.percentile(np.abs(phi), 50)) or 1.0
        y = np.arcsinh(phi / self.s)
        self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0

    def fwd(self, phi):
        return (np.arcsinh(phi / self.s) - self.mu) / self.sd

    def inv(self, y):
        return np.sinh(y * self.sd + self.mu) * self.s

    def inv_t(self, y):
        return torch.sinh(y * self.sd + self.mu) * self.s

    def dump(self):
        return dict(s=self.s, mu=self.mu, sd=self.sd)


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
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    d = np.load(a.data)
    P, E, PHI = d["points"], d["electrodes"], d["phi"]        # (M,3) (N,3) (N,M)
    N, M = PHI.shape
    # split by ELECTRODE (never leak an electrode across the split)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = max(1, int(a.val_frac * N))
    va, tr = perm[:n_val], perm[n_val:]
    print(f"{N} electrodes × {M} pts · train {len(tr)} / val {len(va)} electrodes · {dev}")

    tf = PhiTransform(PHI[tr])
    # normalise coords to ~[-1,1] (mm → cylinder scale)
    cs = np.array([40.0, 40.0, 120.0], dtype=np.float32)
    co = np.array([0.0, 0.0, 120.0], dtype=np.float32)

    def make(idx):
        """The ported nets take (coords, condition) separately, not a concatenated 6-D input."""
        c = np.repeat(((E[idx] - co) / cs)[:, None, :], M, 1).reshape(-1, 3)   # (n*M,3) electrode
        p = np.broadcast_to(((P - co) / cs)[None], (len(idx), M, 3)).reshape(-1, 3)  # (n*M,3) point
        Y = tf.fwd(PHI[idx]).reshape(-1, 1).astype(np.float32)
        return (torch.from_numpy(p.astype(np.float32)),
                torch.from_numpy(c.astype(np.float32)), torch.from_numpy(Y))

    Ptr, Ctr, Ytr = make(tr); Pva, Cva, Yva = make(va)
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
            pb, cb, yb = (Ptr[b].to(dev, non_blocking=True), Ctr[b].to(dev, non_blocking=True),
                          Ytr[b].to(dev, non_blocking=True))
            opt.zero_grad(); l = lossf(net(pb, cb).reshape(-1, 1), yb); l.backward(); opt.step()
            tot += l.item() * len(b)
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            net.eval()
            yv = val_chunked()
            vl = float(((yv - Yva.reshape(-1)) ** 2).mean())
            yp = tf.inv(yv.cpu().numpy()); yt = tf.inv(Yva.cpu().numpy().ravel())
            rel = np.linalg.norm(yp - yt) / np.linalg.norm(yt)
            print(f"  ep {ep+1:4d} train {tot/n:.5f} val {vl:.5f} relL2(phi) {rel:.4f} "
                  f"({time.time()-t0:.0f}s)")
    ck = OUT / f"cyl_{a.arch}.pt"
    torch.save(dict(state=net.state_dict(), arch=a.arch, tf=tf.dump(),
                    cs=cs, co=co, val_elec=va, rel_l2=float(rel)), ck)
    print(f"\nwrote {ck} · final rel-L2(phi) on held-out electrodes = {rel:.4f}")


if __name__ == "__main__":
    main()
