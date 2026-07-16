# Handoff items — things this agent must NOT edit (tracked files)

This agent works only inside `_investigations/learned_vc/` (gitignored). Anything needing a
**tracked** file goes here for the fork / a coordinated single-agent step.

## ⚠️ Immediate — dirty working tree, resolve BEFORE the fork commits

The shared checkout currently has uncommitted work from this session's MU-pool thread:

| file | state | what it is |
|---|---|---|
| `scripts/mri/sample_mu_pool.py` | **modified** | engine-param (`--engine spatial\|fourier`), µV units, MUAP overlay, **the IZ-geometry fix** (IZ at 0.305, asymmetric Lp/Ld, `posz=(Lp−Ld)/2`, arc-length dz), field-viz (`_plot_fields`) |
| `scripts/mri/mu_electrode_grid.py` | **untracked** | HD-EMG M×M electrode-grid script (written, parses, run was interrupted — unverified) |

**Risk:** a `git add -A` from the fork sweeps these into an unrelated commit.
**Options:** (a) commit them now as their own commit (they're finished/tested apart from the grid
run), (b) stash, or (c) tell the fork to stage explicit paths only, never `-A`.

Also unpushed on `main`: the A-06/07/08 + segmentation + pool commits (see `git log origin/main..HEAD`).

## 🔥 High-value, verified: cache `cell_ids` in `LeadField.phi` (~18×, byte-identical)

**Not learned-VC work — this speeds up the *existing* pipeline today.** Measured on WR with a
127k-point fibre query set (`scripts/neural_field/00_baseline_fem_timing.py`):

```
solve_for_point         0.39 s
compute_closest_entity  7.17 s   <- 99.5% of the cost
uh.eval                 0.036 s
```

`LeadField.phi()` (`src/emgforge/fem/leadfield.py:121`) recomputes `compute_closest_entity` on
**every** call, but `cell_ids` depends only on `(mesh, points)` — **not** on `uh`. For a fixed
query set swept over many electrodes it is pure waste.

**Verified:** reusing cached `cell_ids` gives **byte-identical** φ (`max|Δ| = 0.00e+00`) across 3
electrodes, **165× faster** sampling → **7.6 s → 0.43 s per electrode (~18×)**.

**Suggested API** (fork owns `emgforge/fem/`): keep `phi(points, uh)` as-is, add an opt-in
precomputed-locator path, e.g. `phi(points, uh, cell_ids=None)` or a small
`PointSampler(leadfield, points)` object that caches `cell_ids` once and exposes `.phi(uh)`.

**Who benefits immediately:** `scripts/mri/mu_electrode_grid.py` (M×M solves × 637 fibres — the
5×5 grid would drop from ~4 min to ~15 s), `scripts/mri/sample_mu_pool.py`, `validate_wr.py`, and
the Phase-1 dataset generation (259 min → 15 min).

## Stage 0 — port from the archive (tracked-file work, fork or coordinated)

The trainers/models are **not in this repo**; they live in
`/home/dc23/projects/neural-forward-emg/training/neural_field/`:

- `train_neural_field.py` (Exp 1–3 supervised), `train_pinn.py` (Exp 4)
- `models/{neural_field,pinn}.py` (MLP+Fourier, SIREN, PINN)
- `data/pointcloud_dataset.py`, `configs/exp[1-4].yaml`
- (checkpoints 273 MB — local-only, do **not** import)

Target: `src/neural_field/{models,training}/` + `scripts/neural_field/`. This is a `git mv`-shaped
move into a package the fork may be actively reorganising → **do it as a single coordinated step,
not concurrently.**

## Possible later asks

- a `torch` optional-dependency extra in `pyproject.toml` (`[project.optional-dependencies] nf = [...]`)
- σ-tensor query API on the MRI side (see PLAN §2 option B/C) — may want a small helper in
  `emgforge.mri.core` rather than duplicating tag→σ logic here
