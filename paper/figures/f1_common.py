"""Shared helpers for the F1 figure scripts (figs 2, 3, 4, 5, 9 and dataset D1).

Re-exports the validation harness (the direct line-source synthesis recipe, analytical oracle, signal helpers),
opens the cylinder FEM cache, and provides ``record()`` which merges each script's
printed numbers into ``paper/figures/key_numbers_f1.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts/validation", ROOT / "paper/figures"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from harness import *            # noqa: F401,F403  (V, FS, W, golden_cfg, sfap, ana_phi, …)
from harness import V, FS, W     # noqa: F401  (explicit for linters)
import cyl_fem                   # noqa: E402

DZ = V * 1000.0 / FS                         # 0.977 mm — the cylinder-tier φ grid
Z = (np.arange(W) - W // 2) * DZ             # centred z grid of a φ line (mm)
T_AX = np.arange(W) / FS * 1000.0 - 10.0     # the direct recipe's time axis (ms), t = 0 at the NMJ
CACHE = cyl_fem.build_cache()
Z_ABS, RADII, THETAS, ZE = CACHE["z_abs"], CACHE["radii"], CACHE["thetas"], float(CACHE["ze"])
KEY = ROOT / "paper/figures/key_numbers_f1.json"


def fem_phi(r, th=0.0, key="phi", centre=None):
    """FEM φ(z) on the engine's 256-point grid for the fibre at radius r, angle θ."""
    i, j = int(np.argmin(np.abs(RADII - r))), int(np.argmin(np.abs(THETAS - th)))
    return cyl_fem.window(CACHE[key][i, j], Z_ABS, ZE if centre is None else centre, W, DZ)


def dc_free(p, n_edge=12):
    """Remove the (arbitrary) constant of a lead field: mean of the far-z samples."""
    p = np.asarray(p, float)
    return p - np.mean(np.r_[p[:n_edge], p[-n_edge:]])


def normed(x):
    x = np.asarray(x, float)
    return x / np.abs(x).max()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def record(section: str, data: dict):
    """Merge ``data`` under ``section`` in key_numbers_f1.json and echo it."""
    store = json.loads(KEY.read_text()) if KEY.exists() else {}
    store[section] = _jsonable(data)
    KEY.write_text(json.dumps(store, indent=1))
    print(f"[key numbers → {KEY.name} :: {section}]")
    print(json.dumps(store[section], indent=1))
