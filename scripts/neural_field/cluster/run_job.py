"""Single-job entrypoint — what a scheduler array task invokes.

    python scripts/neural_field/cluster/run_job.py --list
    python scripts/neural_field/cluster/run_job.py --where cpu --index $PBS_ARRAY_INDEX
    python scripts/neural_field/cluster/run_job.py --name train_cyl

Resolves a job from experiment.jobs(), checks its dependencies' sentinel files exist on the
shared FS, runs the command from the REPO ROOT, and writes a `<job>.done` sentinel. Idempotent:
a job whose sentinel exists is skipped, so requeued/restarted arrays don't recompute.

The scheduler-specific glue (walltime, queue, module load, env activate) lives in the submit
templates; this file is scheduler-agnostic so it works identically under PBS and SLURM.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import jobs                     # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SENTINEL_DIR = REPO / "_results/neural_field/_cluster_state"


def sentinel(name: str) -> Path:
    return SENTINEL_DIR / f"{name}.done"


def run_one(job, force: bool) -> int:
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    if sentinel(job.name).exists() and not force:
        print(f"[skip] {job.name} — sentinel exists")
        return 0
    missing = [n for n in job.needs if not sentinel(n).exists()]
    if missing:
        print(f"[abort] {job.name} — unmet deps: {missing}", file=sys.stderr)
        return 3
    print(f"[run ] {job.name} ({job.where}): {' '.join(job.cmd)}", flush=True)
    env_note = REPO  # commands are relative to repo root; PYTHONPATH=src set by the submit script
    r = subprocess.run(job.cmd, cwd=str(env_note))
    if r.returncode == 0:
        sentinel(job.name).write_text("ok\n")
        print(f"[done] {job.name}")
    else:
        print(f"[FAIL] {job.name} rc={r.returncode}", file=sys.stderr)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--where", choices=["cpu", "gpu"], help="filter to one node type (array jobs)")
    ap.add_argument("--index", type=int, help="0-based index within the --where filter")
    ap.add_argument("--name", help="run one job by name")
    ap.add_argument("--list", action="store_true", help="print the resolved job list and exit")
    ap.add_argument("--force", action="store_true", help="ignore existing sentinels")
    a = ap.parse_args()
    J = jobs()

    if a.list:
        for i, j in enumerate(J):
            done = "✓" if sentinel(j.name).exists() else " "
            print(f"[{done}] {i:2d} ({j.where}) {j.name}")
        cpu = [j for j in J if j.where == "cpu"]; gpu = [j for j in J if j.where == "gpu"]
        print(f"\n{len(cpu)} cpu, {len(gpu)} gpu. Array sizes: PBS -J 0-{len(cpu)-1} (cpu), "
              f"0-{len(gpu)-1} (gpu).")
        return 0

    if a.name:
        m = [j for j in J if j.name == a.name]
        if not m:
            print(f"no job named {a.name!r}", file=sys.stderr); return 2
        return run_one(m[0], a.force)

    if a.where is not None and a.index is not None:
        pool = [j for j in J if j.where == a.where]
        if not 0 <= a.index < len(pool):
            print(f"index {a.index} out of range for {a.where} (0..{len(pool)-1})", file=sys.stderr)
            return 2
        return run_one(pool[a.index], a.force)

    ap.error("give --list, --name, or (--where and --index)")


if __name__ == "__main__":
    raise SystemExit(main())
