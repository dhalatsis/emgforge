"""Run the whole validation suite and write _results/validation/VALIDATION_REPORT.md.

  tier A  cylinder: first principles → analytical → FEM → pipeline   (scripts/validation/tier_a_cylinder.py)
  tier B  MUAP features vs the literature                            (scripts/validation/tier_b_features.py)
  tier C  interference EMG & motor-unit pool                         (scripts/validation/tier_c_interference.py)
  tier S  the existing chain-level sanity checks                     (scripts/sanity/simulator_sanity.py)

Run:  python scripts/validation/run_all.py          (~4 min; first run builds the FEM cache, +2 min)
      python scripts/validation/run_all.py --report  (rebuild the report from the saved JSON only)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_results/validation"
HERE = Path(__file__).resolve().parent
TIERS = [("A", HERE / "tier_a_cylinder.py", "Cylinder — analytical vs pipeline", "tier_a_cylinder.png"),
         ("B", HERE / "tier_b_features.py", "MUAP features vs the literature", "tier_b_features.png"),
         ("C", HERE / "tier_c_interference.py", "Interference EMG & motor-unit pool", "tier_c_interference.png")]


def run(script: Path) -> int:
    t0 = time.time()
    p = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    print(f"--- {script.name}: exit {p.returncode} in {time.time() - t0:.0f}s\n", flush=True)
    return p.returncode


def report() -> str:
    lines = ["# emgforge validation report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/validation/run_all.py`. "
             "Plan and rationale: `docs/validation/PLAN.md`; literature: `docs/validation/BIBLIOGRAPHY.md`.", ""]
    totals = []
    for tier, _, title, fig in TIERS:
        recs = [r for r in json.loads((OUT / f"{tier}.json").read_text()) if r["tier"] == tier]
        n = [r for r in recs if not r["known"]]
        n_pass = sum(r["passed"] for r in n)
        n_known = sum(1 for r in recs if r["known"] and not r["passed"])
        totals.append((tier, n_pass, len(n), n_known))
        lines += [f"## Tier {tier} — {title}", "",
                  f"**{n_pass}/{len(n)} gated checks pass**" + (f", {n_known} flagged as known limitations" if n_known else "") + ".",
                  "", f"![tier {tier}]({fig})", "",
                  "| | check | measured | expected | refs |", "|---|---|---|---|---|"]
        for r in recs:
            tag = "⚠ known" if (r["known"] and not r["passed"]) else ("✓" if r["passed"] else "✗ FAIL")
            esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {tag} | {esc(r['name'])} | {esc(r['measured'])} | {esc(r['expect'])} | {esc(r['refs'])} |")
        lines.append("")
    lines += ["## Summary", "", "| tier | pass | gated | known |", "|---|---|---|---|"]
    lines += [f"| {t} | {p} | {g} | {k} |" for t, p, g, k in totals]
    lines += ["", "Tier S (chain-level sanity, `scripts/sanity/simulator_sanity.py`): see "
              "`_results/sanity/simulator/simulator_sanity.png`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    if "--report" not in sys.argv:
        codes = [run(s) for _, s, _, _ in TIERS]
        codes.append(run(ROOT / "scripts/sanity/simulator_sanity.py"))
    (OUT / "VALIDATION_REPORT.md").write_text(report())
    print("wrote", OUT / "VALIDATION_REPORT.md")
