"""Simulator — the one-call facade over the whole chain.

Wires the motoneuron pool + twitch/force + the per-MU MUAPs into a single object so a
contraction becomes one line::

    sim = Simulator.from_pipeline("_results/pipeline/L8_5x5_ied10_mu20/pipeline_output.npz")
    rec = sim.run(drive.trapezoid(0.4, 2, 2, 2, fs=2048))      # → EMG + force + spikes

The MUAPs come from the FEM + ``emgforge.synthesis`` pipeline. The validated route is
``scripts/run_pipeline.py`` (segmentation → mesh → lead fields → the direct line-source
recipe on a regular skin grid), whose ``pipeline_output.npz`` ``from_pipeline`` reloads;
``from_tensor`` takes a released ``(N, E, w)`` tensor. The Simulator owns the neural side
and the summation; it does not re-solve the FEM.

``from_mri`` is the convenience lookup for a muscle + grid. It prefers a
``run_pipeline.py`` output (or an explicit ``path``) and only falls back — with a
``UserWarning`` — to the legacy ``_results/mu_pool/electrode_grid`` tensor of
``scripts/activation/build_grid_tensor.py``, whose vertex-snapped ``_phigrid`` electrode
grid is known to be wrong (centre ~17 mm off the muscle, duplicated columns).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emgforge.activation import (
    MotoneuronPool, TwitchPool, compound_emg, compound_emg_multi,
)


@dataclass
class Recording:
    """The output of one :meth:`Simulator.run` — a labelled synthetic recording."""

    t: np.ndarray                 # (T,) seconds
    drive: np.ndarray             # (T,) excitation
    emg: np.ndarray               # (T,) single-channel or (E, T) multichannel
    spikes: list                  # per-MU spike sample indices (ground truth)
    force: np.ndarray | None      # (T,) %MVC, or None
    fs: float
    grid: tuple | None            # (M, M) electrode layout, or None

    @property
    def multichannel(self) -> bool:
        return self.emg.ndim == 2

    @property
    def n_active(self) -> int:
        return int(sum(len(s) > 0 for s in self.spikes))

    def grid_view(self) -> np.ndarray:
        """Reshape a multichannel EMG to ``(M, M, T)``."""
        if self.grid is None:
            raise ValueError("recording has no grid layout")
        return self.emg.reshape(*self.grid, -1)


class Simulator:
    """Neural pool + twitch + MUAPs → EMG (+force) from a drive."""

    def __init__(self, muaps: np.ndarray, fs: float = 2048.0, grid: tuple | None = None,
                 twitch: bool = True, pool_kwargs: dict | None = None):
        self.muaps = np.asarray(muaps, dtype=float)        # (N, w) or (N, E, w)
        self.multichannel = self.muaps.ndim == 3
        self.n_mu = self.muaps.shape[0]
        self.fs = float(fs)
        self.grid = grid
        self.pool = MotoneuronPool(n_mu=self.n_mu, fs=self.fs, **(pool_kwargs or {}))
        self.twitch = TwitchPool(self.pool, fs=self.fs) if twitch else None

    def run(self, drive: np.ndarray, seed: int = 0) -> Recording:
        """Drive the pool and synthesise the recording."""
        drive = np.asarray(drive, dtype=float)
        T = drive.shape[-1]
        spikes = self.pool.spike_trains(drive, seed=seed)
        if self.multichannel:
            emg = compound_emg_multi(spikes, self.muaps, n_samples=T)
        else:
            emg = compound_emg(spikes, self.muaps, n_samples=T)
        force = self.twitch.force(spikes, n_samples=T) if self.twitch else None
        return Recording(np.arange(T) / self.fs, drive, emg, spikes, force, self.fs, self.grid)

    # ---- constructors from saved MUAPs ----
    @classmethod
    def from_tensor(cls, npz_path, fs: float = 2048.0, **kw) -> "Simulator":
        """Multichannel Simulator from a saved ``(N, E, w)`` grid MUAP tensor (``W`` in V,
        optional ``M`` for an M×M layout) — the released-tensor format."""
        d = np.load(npz_path)
        M = int(d["M"]) if "M" in d.files else None
        return cls(d["W"], fs=fs, grid=(M, M) if M else None, **kw)

    @classmethod
    def from_single(cls, npz_path, key: str = "muap_wave", fs: float = 2048.0, **kw) -> "Simulator":
        """Single-channel Simulator from a saved pool run (``muap_wave`` = ``(N, w)``)."""
        return cls(np.load(npz_path)[key], fs=fs, **kw)

    @classmethod
    def from_pipeline(cls, npz_path, **kw) -> "Simulator":
        """Multichannel Simulator from a ``scripts/run_pipeline.py`` output
        (``muap_grid`` in µV, ``grid_shape``, ``fs``) — the cold-built tensor."""
        d = np.load(npz_path)
        return cls(np.asarray(d["muap_grid"], dtype=float) * 1e-6, fs=float(d["fs"]),
                   grid=tuple(int(x) for x in d["grid_shape"]), **kw)

    @classmethod
    def from_file(cls, npz_path, fs: float | None = None, **kw) -> "Simulator":
        """Multichannel Simulator from either saved format, told apart by its keys:
        a ``run_pipeline.py`` ``pipeline_output.npz`` (``muap_grid`` in µV, ``grid_shape``,
        ``fs``) or a released ``(N, E, w)`` tensor (``W`` in V, ``M``). ``fs`` is only
        used for the tensor format (default 2048 Hz); a pipeline output carries its own
        and a conflicting ``fs`` raises."""
        npz_path = Path(npz_path)
        d = np.load(npz_path)
        if "muap_grid" in d.files:
            if fs is not None and not np.isclose(float(fs), float(d["fs"])):
                raise ValueError(f"{npz_path} was synthesised at fs={float(d['fs']):g} Hz, "
                                 f"not the requested {fs:g} Hz")
            return cls.from_pipeline(npz_path, **kw)
        if "W" in d.files:
            return cls.from_tensor(npz_path, fs=2048.0 if fs is None else float(fs), **kw)
        raise ValueError(f"{npz_path} is neither a run_pipeline.py output (muap_grid) "
                         f"nor a grid MUAP tensor (W); keys: {d.files}")

    @classmethod
    def from_mri(cls, muscle: int = 8, m: int = 5, root: str | Path | None = None,
                 fs: float | None = None, *, path: str | Path | None = None,
                 n: int | None = None, ied: float = 10.0, **kw) -> "Simulator":
        """Multichannel Simulator for a muscle + grid from the standard artefacts.

        Resolution order:

        1. ``path`` — an explicit file: a ``scripts/run_pipeline.py``
           ``pipeline_output.npz`` or a released ``(N, E, w)`` tensor (``W``/``M``).
        2. ``<root>/_results/pipeline/L{muscle}_{m}x{n}_ied{ied}_mu*/pipeline_output.npz``
           — the output of ``scripts/run_pipeline.py`` for that muscle and grid
           (``n`` defaults to ``m``; the most recently written run wins if several).
           This is the validated route: regular grid ray-cast onto the skin, direct
           line-source recipe.
        3. ``<root>/_results/mu_pool/electrode_grid/muap_tensor_L{muscle}_M{m}.npz`` —
           the **legacy** cache of ``scripts/activation/build_grid_tensor.py``, built on
           the vertex-snapped ``_phigrid`` electrode grid (centre ~17 mm off the muscle,
           duplicated columns). Loaded with a ``UserWarning``; do not use it for results.

        ``fs`` applies to tensor files only (default 2048 Hz); a pipeline output carries
        its own. Raises ``FileNotFoundError`` when nothing is found.
        """
        if path is not None:
            return cls.from_file(path, fs=fs, **kw)
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        n = m if n is None else int(n)
        pattern = f"L{int(muscle)}_{int(m)}x{n}_ied{float(ied):g}_mu*/pipeline_output.npz"
        # prefer the largest pool, then the newest run of that size
        def _rank(p: Path):
            m_ = re.search(r"_mu(\d+)", p.parent.name)
            return (int(m_.group(1)) if m_ else 0, p.stat().st_mtime)
        runs = sorted((root / "_results/pipeline").glob(pattern), key=_rank)
        if runs:
            return cls.from_file(runs[-1], fs=fs, **kw)
        legacy = root / f"_results/mu_pool/electrode_grid/muap_tensor_L{int(muscle)}_M{int(m)}.npz"
        hint = (f"build the validated tensor with `python scripts/run_pipeline.py --muscle {muscle} "
                f"--grid {m}x{n} --ied {ied:g}` and load it with Simulator.from_pipeline(<out>/"
                f"pipeline_output.npz) or Simulator.from_mri(path=...)")
        if legacy.exists():
            warnings.warn(
                f"Simulator.from_mri: no run_pipeline.py output under {root / '_results/pipeline'} "
                f"matching {pattern!r}; falling back to the LEGACY electrode-grid tensor {legacy} "
                f"(scripts/activation/build_grid_tensor.py on the vertex-snapped _phigrid cache — "
                f"a bad grid: centre ~17 mm off the muscle, duplicated columns). Comparison only: {hint}.",
                UserWarning, stacklevel=2)
            return cls.from_tensor(legacy, fs=2048.0 if fs is None else float(fs), **kw)
        raise FileNotFoundError(
            f"no MUAP tensor for muscle {muscle} on a {m}x{n} grid under {root}: neither a "
            f"run_pipeline.py output ({pattern}) nor the legacy {legacy.name}; {hint}")


__all__ = ["Simulator", "Recording"]
