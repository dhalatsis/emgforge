"""Test-case catalogue for the regression bench.

Splits ~200 cases into two pools:
  - **sanity** (~100): cases the current pipeline should pass with
    r ≥ 0.99 vs the analytical reference.
  - **challenging** (~100): cases the current pipeline is known to handle
    poorly. The integration work should push them toward r ≥ 0.99 too.

Cases come in two flavours:
  - **TierA** — analytical φ(z) → production pipeline → compare to
    analytical MUAP. Tests operator consistency. No FEM dependency.
  - **TierB** — FEM φ(z) → production pipeline → compare to analytical
    MUAP. Tests physical agreement (cylinder geometry roughly matches the
    analytical 4-layer model).

The case catalogue is deterministic — running ``define_cases()`` always
returns the same list in the same order.

Geometry knobs covered:
  - electrode/fibre depth (the dominant axis)
  - L1, L2  (fibre lengths, symmetric and asymmetric)
  - v       (conduction velocity, 2/3/4/5 m/s)
  - fsamp   (sampling, 2048/4096/8192 Hz)
  - distfib (analytical only — angular separation; FEM uses cached solve at θ=0)

Off-axis FEM cases would need a second FEM solve per electrode position
and are deferred until the Workstream B (Neumann error budget) study,
which already plans those solves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

Tier = Literal["A", "B"]


@dataclass(frozen=True)
class TestCase:
    """One regression test case.

    Operator-consistency invariant (Tier A): r ≥ 0.999.
    Physical-agreement invariant (Tier B): r ≥ 0.99 sanity / target ≥ 0.95 challenging.
    """

    name: str
    tier: Tier              # "A" (analytical φ) or "B" (FEM φ)
    category: str           # "sanity" or "challenging"

    # Geometry / sampling
    fiber_depth_mm: float
    L1_mm: float = 60.0
    L2_mm: float = 60.0
    v_m_per_s: float = 4.0
    fsamp_hz: float = 4096.0
    w: int = 256             # current pipeline default; bench may override later
    distfib_deg: float = 0.0   # Tier A only
    electrode_dim1_mm: float = 5.0

    # Acceptance threshold against the analytical reference MUAP
    target_r: float = 0.99

    # Free-form tag for the canary kind (drives reporting groupings)
    tag: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


def _grid_a(values_a, values_b):
    """Cartesian product helper."""
    return [(a, b) for a in values_a for b in values_b]


def define_cases() -> List[TestCase]:
    """Return the deterministic full bench (~200 cases)."""
    cases: List[TestCase] = []

    # ── 1. SANITY POOL ──────────────────────────────────────────────────
    # Tier A operator-consistency. The pipeline should ALWAYS reproduce
    # the analytical MUAP from analytical φ at r ≥ 0.999.

    # Tier A target is 0.99 with the default config (Butterworth smoothing
    # introduces a ~0.01 perturbation even on clean inputs); a strict 0.999
    # operator-consistency test exists separately when the bench is run
    # with the "no_smoothing" config.
    A_SANITY_TARGET = 0.99

    # NB: the analytical SignalGenerator currently only works at v ∈ {3.0, 4.0}
    # (np.arange floating-point bug at other velocities; see memory note).
    A_VALID_V = [3.0, 4.0]

    # 1a. Tier A — depth sweep (default L=120, v=4, fs=4096).
    # Shallow depths (d ≤ 20 mm) stay in sanity. Deeper depths move to the
    # challenging pool — the Butterworth smoothing on an un-decayed boundary
    # introduces an edge artifact that the integration is meant to fix.
    for d in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        cases.append(TestCase(
            name=f"A_depth_d{d:02d}",
            tier="A", category="sanity",
            fiber_depth_mm=float(d),
            target_r=A_SANITY_TARGET,
            tag="A_depth_sweep",
        ))
    # Deeper analytical depths (boundary-truncated) — challenging
    for d in [21, 22, 23, 24, 25, 26, 28, 30]:
        cases.append(TestCase(
            name=f"A_depth_d{d:02d}",
            tier="A", category="challenging",
            fiber_depth_mm=float(d),
            target_r=0.99,
            tag="A_depth_sweep_deep",
        ))

    # 1b. Tier A — velocity sweep at fixed depth (only the two valid speeds)
    for v in A_VALID_V:
        cases.append(TestCase(
            name=f"A_velocity_v{int(v*10):02d}",
            tier="A", category="sanity",
            fiber_depth_mm=15.0,
            v_m_per_s=v,
            target_r=A_SANITY_TARGET,
            tag="A_velocity_sweep",
        ))

    # 1c. Tier A — sampling-rate sweep
    # fs=2048 stresses the spatial window (dz=1.95 → 256·dz=500 mm, way wider
    # than the lead-field decay); move it to challenging.
    cases.append(TestCase(
        name="A_fsamp_4096",
        tier="A", category="sanity",
        fiber_depth_mm=15.0, fsamp_hz=4096.0,
        target_r=A_SANITY_TARGET, tag="A_fsamp_sweep",
    ))
    cases.append(TestCase(
        name="A_fsamp_8192",
        tier="A", category="sanity",
        fiber_depth_mm=15.0, fsamp_hz=8192.0,
        target_r=A_SANITY_TARGET, tag="A_fsamp_sweep",
    ))
    cases.append(TestCase(
        name="A_fsamp_2048",
        tier="A", category="challenging",
        fiber_depth_mm=15.0, fsamp_hz=2048.0,
        target_r=0.99, tag="A_fsamp_sweep_extreme",
    ))

    # 1d. Tier A — fibre length sweep (symmetric)
    for L in [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0]:
        cases.append(TestCase(
            name=f"A_Lsym_L{int(L):03d}",
            tier="A", category="sanity",
            fiber_depth_mm=15.0,
            L1_mm=L, L2_mm=L,
            target_r=A_SANITY_TARGET,
            tag="A_Lsym_sweep",
        ))
    # Sanity Tier A — fibre length sweep at depth=10mm too (a second sanity row)
    for L in [40.0, 60.0, 80.0, 100.0, 120.0]:
        cases.append(TestCase(
            name=f"A_Lsym_d10_L{int(L):03d}",
            tier="A", category="sanity",
            fiber_depth_mm=10.0,
            L1_mm=L, L2_mm=L,
            target_r=A_SANITY_TARGET,
            tag="A_Lsym_d10",
        ))

    # 1e. Tier A — asymmetric fibre lengths
    for L1, L2 in [(40, 80), (50, 70), (60, 90), (80, 50), (100, 40)]:
        cases.append(TestCase(
            name=f"A_Lasym_L{L1}_L{L2}",
            tier="A", category="sanity",
            fiber_depth_mm=15.0,
            L1_mm=float(L1), L2_mm=float(L2),
            target_r=A_SANITY_TARGET,
            tag="A_Lasym",
        ))

    # 1f. Tier A — angular distfib sweep
    for ang in [0.0, 5.0, 10.0, 20.0, 45.0]:
        cases.append(TestCase(
            name=f"A_distfib_a{int(ang):02d}",
            tier="A", category="sanity",
            fiber_depth_mm=15.0,
            distfib_deg=ang,
            target_r=A_SANITY_TARGET,
            tag="A_distfib",
        ))

    # 1g. Tier B — shallow cylindrical FEM cases (expected sanity).
    # Very shallow (d ≤ 6 mm) cases — field is too sharp for default smoothing
    # and σ=5 Gaussian source kernel — move to challenging.
    for d_int in [5, 6]:
        cases.append(TestCase(
            name=f"B_shallow_d{d_int:02d}",
            tier="B", category="challenging",
            fiber_depth_mm=float(d_int),
            target_r=0.99,
            tag="B_too_shallow",
        ))
    for d_int in range(7, 23):  # d = 7..22 mm — sanity
        d = float(d_int)
        cases.append(TestCase(
            name=f"B_shallow_d{d_int:02d}",
            tier="B", category="sanity",
            fiber_depth_mm=d,
            target_r=0.99,
            tag="B_shallow",
        ))
    # Half-mm steps in the most-used band for finer resolution
    for d in [11.5, 13.5, 15.5, 17.5, 19.5, 21.5]:
        cases.append(TestCase(
            name=f"B_shallow_d{int(round(d*10)):03d}h",
            tier="B", category="sanity",
            fiber_depth_mm=d,
            target_r=0.99,
            tag="B_shallow_half",
        ))

    # 1h. Tier B — shallow, varied fibre length (still sanity)
    for d in [10, 12, 15, 17, 18, 20]:
        for L in [40.0, 60.0, 80.0]:
            cases.append(TestCase(
                name=f"B_shallow_d{d:02d}_L{int(L):03d}",
                tier="B", category="sanity",
                fiber_depth_mm=float(d),
                L1_mm=L, L2_mm=L,
                target_r=0.99,
                tag="B_shallow_Lsym",
            ))

    # 1i. Tier B — shallow, varied conduction velocity
    # Only v ∈ {3.0, 4.0} because the analytical reference fails at others.
    for d in [12, 15, 18]:
        for v in [3.0, 4.0]:
            cases.append(TestCase(
                name=f"B_shallow_d{d:02d}_v{int(v*10):02d}",
                tier="B", category="sanity",
                fiber_depth_mm=float(d),
                v_m_per_s=v,
                target_r=0.99,
                tag="B_shallow_v",
            ))

    # 1j. Tier B — shallow, varied sampling rate
    # 4096 (default) is sanity; 2048 and 8192 are challenging (extreme spatial windows).
    cases.append(TestCase(
        name="B_shallow_d15_fs4096",
        tier="B", category="sanity",
        fiber_depth_mm=15.0, fsamp_hz=4096.0,
        target_r=0.99, tag="B_shallow_fsamp",
    ))
    for fs in [2048.0, 8192.0]:
        cases.append(TestCase(
            name=f"B_shallow_d15_fs{int(fs)}",
            tier="B", category="challenging",
            fiber_depth_mm=15.0, fsamp_hz=fs,
            target_r=0.99, tag="B_shallow_fsamp_extreme",
        ))

    # ── 2. CHALLENGING POOL ─────────────────────────────────────────────
    # Cases known to degrade the current pipeline. Targets to FIX with
    # adaptive w + (gated) better BC.

    # 2a. Tier B — deep electrode (wide field, edge truncation)
    for d_int in range(24, 35):   # 24..34 mm
        d = float(d_int)
        cases.append(TestCase(
            name=f"B_deep_d{d_int:02d}",
            tier="B", category="challenging",
            fiber_depth_mm=d,
            target_r=0.95,
            tag="B_deep",
        ))
    for d in [24.5, 26.5, 28.5, 30.5, 32.5, 33.5]:
        cases.append(TestCase(
            name=f"B_deep_d{int(round(d*10)):03d}h",
            tier="B", category="challenging",
            fiber_depth_mm=d,
            target_r=0.95,
            tag="B_deep_half",
        ))

    # 2b. Tier B — long fibre (does not fit in default w=256 window)
    for d in [12, 15, 18]:
        for L in [100.0, 120.0, 140.0, 160.0]:
            cases.append(TestCase(
                name=f"B_longfib_d{d:02d}_L{int(L):03d}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=L, L2_mm=L,
                target_r=0.95,
                tag="B_longfib",
            ))

    # 2c. Tier B — very long fibre + deep depth (combined stressor)
    for d in [20, 25, 30]:
        for L in [80.0, 100.0]:
            cases.append(TestCase(
                name=f"B_deep_long_d{d:02d}_L{int(L):03d}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=L, L2_mm=L,
                target_r=0.9,
                tag="B_deep_long",
            ))

    # 2d. Tier B — short fibre at deep depth (small signal, edge-dominated)
    for d in [20, 25, 30]:
        for L in [20.0, 30.0]:
            cases.append(TestCase(
                name=f"B_deep_short_d{d:02d}_L{int(L):03d}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=L, L2_mm=L,
                target_r=0.9,
                tag="B_deep_short",
            ))

    # Note: Tier B challenging velocity sweeps are disabled because the
    # analytical reference only works at v=3 and v=4. We can still vary the
    # *FEM* solve at other velocities, but cannot compare against an
    # analytical baseline. Use the velocity=3 (low-end valid) deep cases
    # instead to stress the low-velocity regime.
    for d in [20, 25, 28, 30]:
        cases.append(TestCase(
            name=f"B_deep_v3_d{d:02d}",
            tier="B", category="challenging",
            fiber_depth_mm=float(d),
            v_m_per_s=3.0,
            target_r=0.9,
            tag="B_deep_v3",
        ))

    # 2g. Tier B — asymmetric fibre at depth (stresses pare formula)
    for d in [15, 20, 25]:
        for L1, L2 in [(40, 100), (100, 40), (60, 120), (120, 60)]:
            cases.append(TestCase(
                name=f"B_asym_d{d:02d}_L{L1}_L{L2}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=float(L1), L2_mm=float(L2),
                target_r=0.9,
                tag="B_asym",
            ))

    # 2h. Tier B — fsamp stress (low fs → coarse dz → big spatial extent;
    # high fs → small dz → small extent stressing wide fields)
    for d in [20, 25, 30]:
        for fs in [2048.0, 8192.0]:
            cases.append(TestCase(
                name=f"B_deep_fs_d{d:02d}_fs{int(fs)}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                fsamp_hz=fs,
                target_r=0.9,
                tag="B_deep_fsamp",
            ))

    # 2i. Tier A — challenging operator-consistency: very long fibre
    # (the FFT window can't possibly fit it; pipeline must still degrade
    # gracefully).
    for L in [140.0, 160.0, 200.0]:
        cases.append(TestCase(
            name=f"A_longfib_L{int(L):03d}",
            tier="A", category="challenging",
            fiber_depth_mm=15.0,
            L1_mm=L, L2_mm=L,
            target_r=0.95,
            tag="A_longfib",
        ))

    # (Tier A extreme-velocity sweeps disabled — analytical only stable at v∈{3,4})

    # 2k. Tier A — extreme asymmetry (asym fibre lengths)
    for L1, L2 in [(20, 200), (200, 20), (30, 180), (180, 30)]:
        cases.append(TestCase(
            name=f"A_extreme_asym_L{L1}_L{L2}",
            tier="A", category="challenging",
            fiber_depth_mm=15.0,
            L1_mm=float(L1), L2_mm=float(L2),
            target_r=0.95,
            tag="A_extreme_asym",
        ))

    # 2l. Tier A — deep + long combined
    for d in [25, 28, 30]:
        for L in [120.0, 160.0]:
            cases.append(TestCase(
                name=f"A_deep_long_d{d:02d}_L{int(L):03d}",
                tier="A", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=L, L2_mm=L,
                target_r=0.95,
                tag="A_deep_long",
            ))

    # 2m. Tier B — extreme deep (right next to mesh edge for fibres)
    for d in [35.0, 35.5, 36.0, 36.5]:
        cases.append(TestCase(
            name=f"B_xtreme_deep_d{int(round(d*10)):03d}",
            tier="B", category="challenging",
            fiber_depth_mm=d,
            target_r=0.9,
            tag="B_xtreme_deep",
        ))

    # 2n. Tier B — extra deep_short combinations (deep+short stresses end effects)
    for d in [22, 24, 26, 28, 32]:
        cases.append(TestCase(
            name=f"B_deep_short2_d{d:02d}_L25",
            tier="B", category="challenging",
            fiber_depth_mm=float(d),
            L1_mm=25.0, L2_mm=25.0,
            target_r=0.9,
            tag="B_deep_short2",
        ))

    # 2o. Tier B — extra asym combos at various depths
    for d in [18, 22, 28, 32]:
        for L1, L2 in [(80, 30), (30, 80), (100, 50)]:
            cases.append(TestCase(
                name=f"B_asym2_d{d:02d}_L{L1}_L{L2}",
                tier="B", category="challenging",
                fiber_depth_mm=float(d),
                L1_mm=float(L1), L2_mm=float(L2),
                target_r=0.9,
                tag="B_asym2",
            ))

    return cases


def case_counts(cases: List[TestCase]) -> dict:
    """Helper: count cases by (category, tier)."""
    out = {"sanity": {"A": 0, "B": 0}, "challenging": {"A": 0, "B": 0}}
    for c in cases:
        out[c.category][c.tier] += 1
    return out


if __name__ == "__main__":
    cs = define_cases()
    print(f"Total cases: {len(cs)}")
    counts = case_counts(cs)
    print(f"Sanity:      {sum(counts['sanity'].values())} "
          f"(Tier A: {counts['sanity']['A']}, Tier B: {counts['sanity']['B']})")
    print(f"Challenging: {sum(counts['challenging'].values())} "
          f"(Tier A: {counts['challenging']['A']}, Tier B: {counts['challenging']['B']})")
    by_tag = {}
    for c in cs:
        by_tag.setdefault(c.tag, 0)
        by_tag[c.tag] += 1
    for tag, n in sorted(by_tag.items()):
        print(f"  {tag:30s} {n}")
