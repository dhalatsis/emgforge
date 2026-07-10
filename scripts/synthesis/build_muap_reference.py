"""Build the MUAP regression reference — the 21 expected outputs the test gates against.

The case definitions live in tests/synthesis/muap_cases.py (shared with the test); this
script only runs them and freezes the outputs. Rerun it ONLY when a change to the MUAP
outputs is intended, and commit the refreshed data/muap_reference.npz alongside the code
change that justifies it.

    python scripts/synthesis/build_muap_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))  # import the shared case module

from synthesis.muap_cases import CASES, run_case  # noqa: E402

OUT = ROOT / "tests" / "synthesis" / "data" / "muap_reference.npz"


def main() -> int:
    store = {}
    print(f"{'case':22s} {'engine':8s} {'nz':>10} {'peak':>12}")
    print("-" * 58)
    for c in CASES:
        t, m = run_case(c)
        store[f"{c.name}__t"] = t.astype(np.float64)
        store[f"{c.name}__muap"] = m.astype(np.float64)
        phi_shape = "x".join(str(s) for s in np.atleast_2d(c.phi).shape)
        print(f"{c.name:22s} {c.engine:8s} {phi_shape:>10} {np.abs(m).max():>12.4e}")
    store["case_names"] = np.array([c.name for c in CASES])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **store)
    print("-" * 58)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
