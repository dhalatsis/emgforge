"""Diagnostic plots for the regression bench.

Produces four panels into tests/regression/figures/:

  1. summary_pass_rate.png    — bar chart of pass/fail per config
  2. delta_r_scatter.png      — per-case Δr (adaptive_w_auto vs default)
  3. tier_a_recovery.png      — overlaid MUAPs for ~6 Tier-A cases that
                                 went from fail (default) → pass (auto)
  4. neumann_budget_grid.png  — phi(z) for L240 vs L400 in the Neumann
                                 budget study, plus MUAP r heatmap.

Usage:
  PYTHONPATH=src python tests/regression/make_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "figures"
SNAP_DIR = Path(__file__).resolve().parent / "snapshots"


def _load(name: str) -> list[dict]:
    with open(SNAP_DIR / name) as f:
        return json.load(f)["results"]


# ────────────────────────── plot 1: pass rate ──────────────────────────

def plot_pass_rate():
    configs = {
        "default":                   _load("baseline_default.json"),
        "edge_taper":                _load("baseline_edge_taper.json"),
        "adaptive_w":                _load("adaptive_w.json"),
        "adaptive_w_taper":          _load("adaptive_w_taper.json"),
        "adaptive_w_no_smoothing":   _load("adaptive_w_no_smoothing.json"),
        "adaptive_w_auto":           _load("adaptive_w_auto.json"),
    }

    labels = list(configs)
    n_sanity = []
    p_sanity = []
    n_chal = []
    p_chal = []
    for cfg_name, rs in configs.items():
        s = [r for r in rs if r["category"] == "sanity"]
        c = [r for r in rs if r["category"] == "challenging"]
        n_sanity.append(len(s))
        p_sanity.append(sum(1 for r in s if r["r_vs_ref"] >= r["target_r"]))
        n_chal.append(len(c))
        p_chal.append(sum(1 for r in c if r["r_vs_ref"] >= r["target_r"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, [p / n * 100 for p, n in zip(p_sanity, n_sanity)],
           width=w, label="Sanity (n=90)", color="steelblue")
    ax.bar(x + w / 2, [p / n * 100 for p, n in zip(p_chal, n_chal)],
           width=w, label="Challenging (n=110)", color="coral")

    for i, (ps, ns, pc, nc) in enumerate(zip(p_sanity, n_sanity, p_chal, n_chal)):
        ax.text(i - w / 2, ps / ns * 100 + 1, f"{ps}/{ns}",
                ha="center", fontsize=8)
        ax.text(i + w / 2, pc / nc * 100 + 1, f"{pc}/{nc}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Bench pass rate by config (target = case-specific r threshold)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = OUT_DIR / "01_summary_pass_rate.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ────────────────────────── plot 2: Δr scatter ──────────────────────────

def plot_delta_r():
    baseline = {r["name"]: r for r in _load("baseline_default.json")}
    new = {r["name"]: r for r in _load("adaptive_w_auto.json")}

    names = sorted(set(baseline) & set(new))
    cats = [baseline[n]["category"] for n in names]
    r_base = np.array([baseline[n]["r_vs_ref"] for n in names])
    r_new = np.array([new[n]["r_vs_ref"] for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: r_new vs r_base
    ax = axes[0]
    for cat, color in (("sanity", "steelblue"), ("challenging", "coral")):
        m = [c == cat for c in cats]
        ax.scatter(r_base[m], r_new[m], s=20, color=color,
                   label=f"{cat} (n={sum(m)})", alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="no change")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("r vs analytical, baseline (MUAPConfig() default)")
    ax.set_ylabel("r vs analytical, adaptive_w_auto")
    ax.set_title("Per-case r: above the diagonal = improved")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Right: Δr histogram, challenging only
    ax = axes[1]
    delta = r_new - r_base
    cd = delta[np.array(cats) == "challenging"]
    bins = np.linspace(-0.05, 0.5, 40)
    ax.hist(cd, bins=bins, color="coral", edgecolor="k")
    ax.axvline(0, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("Δr (adaptive_w_auto − default)")
    ax.set_ylabel("count")
    ax.set_title(f"Δr on challenging cases (n={len(cd)})\n"
                 f"mean Δr {cd.mean():+.4f}, max {cd.max():+.4f}, "
                 f"# improved {sum(cd > 0.01)}, # regressed {sum(cd < -0.01)}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "02_delta_r_scatter.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ────────────────────────── plot 3: Tier A recovery ──────────────────────────

def plot_tier_a_recovery():
    baseline = {r["name"]: r for r in _load("baseline_default.json")}
    new = {r["name"]: r for r in _load("adaptive_w_auto.json")}

    # Pick the most-improved Tier-A challenging cases
    delta = []
    for name in baseline:
        if name not in new:
            continue
        b = baseline[name]
        n = new[name]
        if b["tier"] != "A" or b["category"] != "challenging":
            continue
        delta.append((name, n["r_vs_ref"] - b["r_vs_ref"]))
    delta.sort(key=lambda x: -x[1])
    picks = [d[0] for d in delta[:6]]

    # Two rows of 3 plots: top = baseline (w=256), bottom = adaptive_w_auto
    # (typically w=512 with its own matched reference). Each cell shows
    # production vs its matched analytical reference, so the comparison is
    # apples-to-apples per row.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for col, name in enumerate(picks[:3]):
        b = baseline[name]
        n = new[name]
        # Top row: baseline
        ax = axes[0, col]
        t_b = np.asarray(b["t_ms"]); mref = np.asarray(b["muap_ref"]); mpr = np.asarray(b["muap_prod"])
        nb = min(len(t_b), len(mref), len(mpr))
        ax.plot(t_b[:nb], mref[:nb], "k-", lw=1.4, label="analytical")
        ax.plot(t_b[:nb], mpr[:nb], "C1-", lw=0.9, label=f"baseline r={b['r_vs_ref']:.3f}")
        ax.set_title(f"{name}  [baseline, w={b['case_params']['w']}]", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        # Bottom row: adaptive_w_auto
        ax = axes[1, col]
        t_n = np.asarray(n["t_ms"]); nref = np.asarray(n["muap_ref"]); npr = np.asarray(n["muap_prod"])
        nn = min(len(t_n), len(nref), len(npr))
        ax.plot(t_n[:nn], nref[:nn], "k-", lw=1.4, label="analytical")
        ax.plot(t_n[:nn], npr[:nn], "C2-", lw=0.9, label=f"auto r={n['r_vs_ref']:.3f}")
        # Find chosen w from output length (production output is exactly w samples)
        w_chosen = len(npr)
        ax.set_title(f"{name}  [adaptive_w_auto, w={w_chosen}]", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_xlabel("t (ms)")
    fig.suptitle(
        "Tier-A challenging cases — baseline (top) vs adaptive_w_auto (bottom)\n"
        "Each row: production MUAP overlaid on its matched-w analytical reference",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "03_tier_a_recovery.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ────────────────────────── plot 4: Neumann budget heatmap ──────────────────────────

def plot_neumann_budget():
    with open(Path(__file__).resolve().parent / "NEUMANN_BUDGET.json") as f:
        data = json.load(f)
    results = data["results"]

    depths = sorted({r["depth_mm"] for r in results})
    src_zs = sorted({r["source_z_mm"] for r in results})

    rgrid = np.full((len(depths), len(src_zs)), np.nan)
    l2grid = np.full((len(depths), len(src_zs)), np.nan)
    for r in results:
        i = depths.index(r["depth_mm"])
        j = src_zs.index(r["source_z_mm"])
        rgrid[i, j] = r["muap_r_240_vs_400"]
        l2grid[i, j] = r["phi_l2_rel"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    im0 = axes[0].imshow(rgrid, aspect="auto", cmap="RdYlGn",
                          vmin=0.985, vmax=1.0, origin="lower")
    axes[0].set_xticks(range(len(src_zs)))
    axes[0].set_xticklabels([f"{z:.0f} mm" for z in src_zs])
    axes[0].set_yticks(range(len(depths)))
    axes[0].set_yticklabels([f"{d:.0f} mm" for d in depths])
    axes[0].set_xlabel("source z (absolute)")
    axes[0].set_ylabel("fibre depth")
    axes[0].set_title("MUAP r (L240 vs L400) — gate metric")
    for i in range(rgrid.shape[0]):
        for j in range(rgrid.shape[1]):
            axes[0].text(j, i, f"{rgrid[i, j]:.3f}", ha="center", va="center",
                         fontsize=8,
                         color="white" if rgrid[i, j] < 0.992 else "black")
    plt.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(l2grid, aspect="auto", cmap="viridis",
                          vmin=0, vmax=l2grid.max(), origin="lower")
    axes[1].set_xticks(range(len(src_zs)))
    axes[1].set_xticklabels([f"{z:.0f} mm" for z in src_zs])
    axes[1].set_yticks(range(len(depths)))
    axes[1].set_yticklabels([f"{d:.0f} mm" for d in depths])
    axes[1].set_xlabel("source z (absolute)")
    axes[1].set_ylabel("fibre depth")
    axes[1].set_title("phi rel-L2 (along fibre) — BC contamination")
    for i in range(l2grid.shape[0]):
        for j in range(l2grid.shape[1]):
            axes[1].text(j, i, f"{l2grid[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white")
    plt.colorbar(im1, ax=axes[1], shrink=0.85)

    fig.suptitle(f"Neumann error budget — gate min(r)={data['min_muap_r']:.4f}, "
                 f"decision: {data['verdict']}", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "04_neumann_budget_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ────────────────────────── main ──────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Writing plots to", OUT_DIR)
    plot_pass_rate()
    plot_delta_r()
    plot_tier_a_recovery()
    plot_neumann_budget()
    print("done")


if __name__ == "__main__":
    main()
