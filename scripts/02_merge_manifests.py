#!/usr/bin/env python3
"""
Merge chunk manifests produced by scripts/00_generate_meshes.py into a single manifest.csv.

Chunked mesh generation writes:
  manifest_<start>_<end>.csv

This script concatenates them (single header) and sorts by sample_id.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        rows = list(reader)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Mesh output directory containing manifest_*.csv (and where manifest.csv will be written).",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="manifest_[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].csv",
        help="Glob for chunk manifests (default matches manifest_000000_000049.csv style).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output manifest path. Defaults to <out_dir>/manifest.csv",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_path = Path(args.out) if args.out else (out_dir / "manifest.csv")

    manifest_paths = sorted(out_dir.glob(args.glob))
    if not manifest_paths:
        raise SystemExit(f"No chunk manifests found in {out_dir} matching glob: {args.glob}")

    all_rows: List[Dict[str, str]] = []
    fieldnames = None
    for mp in manifest_paths:
        rows = read_manifest(mp)
        if not rows:
            continue
        # Validate consistent columns
        with open(mp, "r", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"Header mismatch in {mp}: {reader.fieldnames} != {fieldnames}")
        all_rows.extend(rows)

    if fieldnames is None:
        raise SystemExit("All chunk manifests were empty; nothing to write.")

    # Sort by sample_id if present
    if "sample_id" in fieldnames:
        all_rows.sort(key=lambda r: r.get("sample_id", ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote merged manifest with {len(all_rows)} rows: {out_path}")


if __name__ == "__main__":
    main()

