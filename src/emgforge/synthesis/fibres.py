"""Fibre + FibreBed — the motor-unit fibre bed as an inspectable value.

A ``Fibre`` is one muscle fibre's geometry + conduction — **not** the lead field
sampled along it. φ comes off the fibre (the synthesis contract is
``field_to_muap(field, bed, cfg)``), so one bed serves many electrodes without being
redrawn. A ``FibreBed`` is a validated, frozen collection with:

  * array views for the vectorised Fourier path (``.dz_mm`` … ``.v``),
  * ``.uniform_dz`` — the single scalar dz the Fourier engine needs, or ``NonUniformDz``,
  * factories: ``from_arrays`` (measured PM / FEM beds), ``uniform`` (identical fibres),
    ``jittered`` (drawn from per-fibre Gaussians with a seed — the bed the engine used to
    hide inside its own RNG, now a reproducible, inspectable value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NonUniformDz(ValueError):
    """A single scalar dz was requested from a bed whose fibres have different dz."""


@dataclass(frozen=True)
class Fibre:
    """One fibre's geometry + conduction (no lead field — φ is sampled separately)."""

    dz_mm: float            # arc-length step of the φ samples along THIS fibre
    len1_mm: float          # NMJ → proximal tendon
    len2_mm: float          # NMJ → distal tendon
    posz_mm: float = 0.0    # NMJ offset along the detector z-axis
    v: float = 4.0          # conduction velocity, mm/ms (≡ m/s)

    def __post_init__(self) -> None:
        if self.dz_mm <= 0:
            raise ValueError(f"dz_mm must be > 0, got {self.dz_mm}")
        if self.v <= 0:
            raise ValueError(f"v must be > 0, got {self.v}")
        if self.len1_mm + self.len2_mm <= 0:
            raise ValueError(
                f"len1_mm+len2_mm must be > 0, got {self.len1_mm}+{self.len2_mm}")


@dataclass(frozen=True)
class FibreBed:
    """A validated, immutable collection of fibres — one motor unit's bed."""

    fibres: tuple[Fibre, ...]

    def __post_init__(self) -> None:
        if len(self.fibres) == 0:
            raise ValueError("FibreBed needs at least one fibre")

    def __len__(self) -> int:
        return len(self.fibres)

    def __iter__(self):
        return iter(self.fibres)

    def __getitem__(self, i: int) -> Fibre:
        return self.fibres[i]

    # ---- array views (the vectorised Fourier path reads these) ----
    @property
    def dz_mm(self) -> np.ndarray:
        return np.array([f.dz_mm for f in self.fibres], dtype=float)

    @property
    def len1_mm(self) -> np.ndarray:
        return np.array([f.len1_mm for f in self.fibres], dtype=float)

    @property
    def len2_mm(self) -> np.ndarray:
        return np.array([f.len2_mm for f in self.fibres], dtype=float)

    @property
    def posz_mm(self) -> np.ndarray:
        return np.array([f.posz_mm for f in self.fibres], dtype=float)

    @property
    def v(self) -> np.ndarray:
        return np.array([f.v for f in self.fibres], dtype=float)

    @property
    def uniform_dz(self) -> float:
        """The bed's shared dz — or raise ``NonUniformDz``.

        The Fourier engine needs one scalar dz (it feeds both the (kt, kz) resample and
        ``C(kz) = dz·fftc(φ)``), so a bed whose fibres have different dz — e.g. PM's
        curved 200-pt polylines, each with its own arc length — cannot go through it.
        This raises rather than silently taking ``dz[0]``.
        """
        dz = self.dz_mm
        if not np.allclose(dz, dz[0]):
            n_distinct = len(np.unique(np.round(dz, 9)))
            raise NonUniformDz(
                f"bed has {n_distinct} distinct dz values; the Fourier engine needs a "
                "single scalar dz — use the spatial engine for a per-fibre-dz bed")
        return float(dz[0])

    # ---- factories ----
    @classmethod
    def from_arrays(cls, dz_mm, len1_mm, len2_mm, posz_mm=0.0, v=4.0) -> "FibreBed":
        """Build a bed from measured / derived per-fibre arrays (PM npz, FEM tiers).

        Each argument may be a scalar (broadcast) or a 1-D array; the bed size is the
        longest array. Scalars broadcast; mismatched array lengths raise.
        """
        arrs = [np.atleast_1d(np.asarray(a, dtype=float))
                for a in (dz_mm, len1_mm, len2_mm, posz_mm, v)]
        n = max(a.shape[0] for a in arrs)
        cols = []
        for name, a in zip(("dz_mm", "len1_mm", "len2_mm", "posz_mm", "v"), arrs):
            if a.shape[0] == 1:
                a = np.broadcast_to(a, (n,))
            elif a.shape[0] != n:
                raise ValueError(f"{name} length {a.shape[0]} != bed size {n}")
            cols.append(a)
        dz, l1, l2, pz, vv = cols
        return cls(tuple(
            Fibre(float(dz[i]), float(l1[i]), float(l2[i]), float(pz[i]), float(vv[i]))
            for i in range(n)))

    @classmethod
    def uniform(cls, n: int, dz_mm: float, len1_mm: float = 60.0, len2_mm: float = 60.0,
                posz_mm: float = 0.0, v: float = 4.0) -> "FibreBed":
        """``n`` identical fibres — the old ``MUAPConfig`` defaults."""
        return cls(tuple(Fibre(dz_mm, len1_mm, len2_mm, posz_mm, v) for _ in range(n)))

    @classmethod
    def jittered(cls, n: int, dz_mm: float, *, len1_mm: float = 60.0, len2_mm: float = 60.0,
                 v: float = 4.0, nmj_sigma_mm: float = 0.0, cv_sigma: float = 0.0,
                 tendon_sigma_mm: float = 0.0, fibre_length_sigma_mm: float = 0.0,
                 seed: int = 42) -> "FibreBed":
        """A bed drawn from per-fibre Gaussians — the jitter the engine used to hide.

        Reproduces ``api._resolve_per_fiber_jitter`` exactly: same seeded draw ORDER
        (NMJ offset → velocity → symmetric length Δ → asymmetric tendon Δ×2) and the
        same clips (v ∈ [1.5, 7.0]; each semi-length ≥ 10 mm). So making the bed a value
        is byte-identical to the old inline draw — the migration can be verified, not
        just asserted.
        """
        rng = np.random.default_rng(int(seed))
        posz = (rng.normal(0.0, float(nmj_sigma_mm), n) if nmj_sigma_mm > 0.0
                else np.zeros(n))
        if cv_sigma > 0.0:
            vv = np.clip(rng.normal(float(v), float(cv_sigma), n), 1.5, 7.0)
        else:
            vv = np.full(n, float(v))
        d_sym = (rng.normal(0.0, float(fibre_length_sigma_mm), n)
                 if fibre_length_sigma_mm > 0.0 else np.zeros(n))
        if tendon_sigma_mm > 0.0:
            d_a1 = rng.normal(0.0, float(tendon_sigma_mm), n)
            d_a2 = rng.normal(0.0, float(tendon_sigma_mm), n)
        else:
            d_a1 = d_a2 = np.zeros(n)
        l1 = np.maximum(float(len1_mm) + d_sym + d_a1, 10.0)
        l2 = np.maximum(float(len2_mm) + d_sym + d_a2, 10.0)
        return cls(tuple(
            Fibre(float(dz_mm), float(l1[i]), float(l2[i]), float(posz[i]), float(vv[i]))
            for i in range(n)))


__all__ = ["Fibre", "FibreBed", "NonUniformDz"]
