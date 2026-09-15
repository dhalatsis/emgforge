"""Shared matplotlib style for every paper figure. Import and call ``use()`` first.

    from style import use, W1, W2, save
    use(); fig, ax = plt.subplots(figsize=(W1, 2.6)); ...; save(fig, "fig_name")

Conventions: single-column width W1 = 3.4 in, double-column W2 = 7.0 in; 8-pt text;
one colour per *model* (see COL); vector PDF + PNG preview; no titles inside axes
(captions carry them); panel letters via ``letter(ax, "a")``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
W1, W2 = 3.4, 7.0

# one colour per model / route, used consistently across all figures
COL = {
    "first":     "#111111",   # first-principles / closed-form reference (black)
    "analytical": "#1f5fbf",  # Farina-2004 analytical cylinder (blue)
    "fem":       "#d95f02",   # FEM lead field (orange)
    "golden":    "#1b9e77",   # golden spatial recipe (green)
    "fourier":   "#c0392b",   # Fourier / Farina generator (red)
    "raw":       "#8c8c8c",   # unprocessed / no denoise (grey)
    "mono":      "#1f5fbf", "sd": "#d95f02", "dd": "#7570b3",
    "poisson":   "#7570b3", "harmonic": "#1b9e77",
    "lit":       "#111111",   # literature reference points
}


def use():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False,
        "mathtext.fontset": "dejavusans",
    })


def letter(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name, out=HERE):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


__all__ = ["use", "letter", "save", "COL", "W1", "W2", "HERE"]
