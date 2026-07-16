# Learned volume conductor (VC) on MRI anatomy — plan

**Status:** planning · **Opened:** 2026-07-16 · **Location:** `_investigations/learned_vc/` (gitignored scratch; promoted to `src/neural_field/` only at handoff)

## Goal

Replace / accelerate the per-electrode FEM solve with a **learned VC field** — a coordinate
network `u(x | electrode, anatomy)` — for the **MRI anatomy**, extending the closed
cylinder-only work. Downstream, φ from the learned field feeds the existing spatial engine →
MUAP, so it must be good enough *for MUAPs*, not just pointwise φ.

---

## 1. Where we already are (verified, not assumed)

**Prior work — `neural-forward-emg/training/neural_field/` — "Exp 1–4", CLOSED & complete:**

| Exp | What varies | Best result |
|---|---|---|
| 1 | electrode position, **fixed cylinder** | Rel L2 **0.029**, R² 0.999, **MUAP corr 0.942** (MLP + Fourier features) |
| 2 | pennation angle | MUAP corr 0.697 |
| 3 | fat thickness | MUAP corr 0.678 |
| 4 | **PINN, zero FEM labels** | **98.5% of Exp 1's supervised quality** |

Architectures: MLP+random-Fourier (best MUAP), SIREN (lowest pointwise error but **jaggedness
wrecked MUAPs**), SIREN+relcoords (best small-data), PINN (pure/hybrid/FiLM).

**In THIS repo (`src/neural_field/`)** — only the *data-generation* half:
- `voxel/voxelize.py` — voxelize geometry + source (needs FEM/dolfinx)
- `pointcloud/{sampling,generate,evaluate}.py` — point sampling, dataset gen, FEM+σ evaluation at points
- `pointcloud/analytical.py` — **σ at arbitrary points from tissue radii, no mesh/FEM**

⚠️ **The trainers and models are NOT in this repo** — `train_neural_field.py`, `train_pinn.py`,
`models/{neural_field,pinn}.py`, `configs/exp*.yaml` all live only in the archive. Porting them
is Stage 0.

## 2. The core problem — anatomy representation

**Why the cylinder needed no mesh** (two properties, both lost on MRI):
1. **Parametrized** by a handful of scalars (radii, pennation, fat thickness) → those scalars
   *are* the conditioning vector.
2. **Analytic σ(x)** from those scalars (`pointcloud/analytical.py`) → the PINN's `∇·(σ∇u)`
   needs no mesh at all.

**MRI has neither.** The anatomy is an arbitrary segmentation; σ(x) is only known from the label
volume / mesh tags. So a learned VC on MRI *must* pick an anatomy representation. This is the
"several ideas" fork:

| # | Representation | σ(x) source | Mesh at inference? | Cross-subject? | Notes |
|---|---|---|---|---|---|
| **A** | **Keep the mesh** | cell tags (point→cell→tag→σ) | **yes** | no | Works today (`pointcloud/evaluate.py`). Cheapest path; keeps the dolfinx dep. |
| **B** | **Voxelize the segmentation** | nifti labels → σ grid, lookup | no | (with an encoder) | We now ship `forearm_WR_segmentation.nii.gz` (**77 KB**, 320×320×60). Mesh-free at inference; natural CNN input. |
| **C** | **Implicit anatomy** (SDF/occupancy per muscle) | learned/fit implicit → smooth σ | no | maybe | Smooth + differentiable σ → best for PINN gradients. Needs a fit per subject. |
| **D** | **Conditional field** (anatomy encoder → latent z) | via B or C | no | **yes** | `u(x | z, electrode)`. The ambitious/general one. |
| **E** | **Per-subject overfit** | n/a (supervised) | no | no | `u(x | electrode)`, one net per arm. Simplest thing that's useful — directly replaces `solve_for_point`. |

**Extra complication vs the cylinder — anisotropy.** MRI muscle σ is a **tensor** oriented along
each muscle's fibre direction (`sigma_mode="centerline"`, from `forearm_WR_fibers.json`). The
cylinder's anisotropy was a single parametrized axis. A PINN on MRI needs the **σ tensor field**
at arbitrary x — that's the genuine research step, not a port.

## 3. A hard design constraint we can now state precisely

Today we proved (independently, `over_nmj_muap_discrepancy/`):

> **SFAP ∝ ∫ Vm(z)·φ″(z) dz** — the MUAP is driven by the **second derivative** of the lead field.

This *explains* the prior work's SIREN failure ("high frequencies become big artifacts after the
band-pass/differentiation"). So for the learned VC:
- **Smoothness is a first-class requirement**, not a nicety. A field with tiny pointwise error but
  rough curvature will produce garbage MUAPs.
- **Validate on the downstream MUAP**, never on φ error alone. Pointwise R²=0.999 already coexisted
  with MUAP corr 0.94 — and SIREN did *better* pointwise while doing *worse* on MUAPs.
- Consider penalising/monitoring φ″ directly during training.

## 4. Re-baseline the speed claim (honest)

The 75× figure was **~150 ms/electrode (net) vs ~13 s/electrode (FEM)**. On the WR mesh today a
solve is **~0.3 s** + ~6 s to sample φ along fibres. So the raw-speed argument is much weaker than
it was. The learned VC's case may now rest more on:
- **differentiability** (inverse problems: infer anatomy/fibre geometry from EMG),
- **mesh-free inference** (no dolfinx at deploy),
- **amortisation across many electrodes/subjects**,
than on beating a 0.3 s solve. **Measure this before investing.**

## 5. Staged plan

- **Stage 0 — port + reproduce.** Bring `train_neural_field.py`, `train_pinn.py`, `models/`,
  `configs/` from the archive into the repo. Reproduce Exp 1 (cylinder) as a regression baseline.
  *No new science; establishes the floor.* Also: re-baseline FEM vs net timing (§4).
- **Stage 1 — MRI, per-subject, supervised (option E).** `u(x | electrode)` on the WR mesh.
  Training data is cheap now (0.3 s/solve → thousands of electrodes feasible). Validate: φ error,
  φ″ smoothness, **MUAP corr vs FEM**, and **vs Neurodec** on the 105-electrode PM set.
- **Stage 2 — MRI PINN (options B/C).** Needs the σ **tensor** field at arbitrary x from the
  segmentation. The real step. Compare against Stage 1's supervised net.
- **Stage 3 — cross-subject (option D).** Anatomy encoder → latent conditioning. Only if 1–2 land.

## 6. Validation — we are unusually well placed

This session built exactly the harness this needs:
- the full spatial MUAP pipeline (`field_to_muap`, proven corr 1.0000 vs `pm_lib`),
- `emgforge.synthesis.metrics` (align_score, lobe/EOF metrics, jaggedness ← literally a smoothness metric),
- the **Neurodec ground truth** for MU-113 across **105 electrodes**,
- the committed WR segmentation + fibre config + cached mesh.

So a learned VC can be scored end-to-end: **learned φ → MUAP → vs FEM-MUAP → vs Neurodec.**

## 7. Open decisions (need a human call)

1. **Which anatomy representation** to commit to first — A (mesh, cheapest) vs B (voxel, mesh-free)?
2. **Per-subject (E) or cross-subject (D)?** Determines whether this is engineering or research.
3. **Supervised or PINN first?** Supervised is a known quantity on MRI; PINN is where the novelty is
   but needs the σ tensor field.
4. **Is the goal speed, differentiability, or generality?** §4 says speed alone is now a weak case.

## 8. Working model — two agents, one repo

**Ownership split (zero-conflict by construction):**

| | this agent (learned VC) | the fork (main refactor) |
|---|---|---|
| writes | `_investigations/learned_vc/` **only** | `src/`, `scripts/`, `tests/`, docs |
| git | **never touches the index** (folder is gitignored) | owns **all** commits |
| shared files | touches none (`.gitignore`, `pyproject.toml` off-limits) | owns them |

- `_investigations/` is **already gitignored**, so this agent's work cannot collide with the fork's
  commits — no staged-file races, no merge conflicts, nothing to rebase.
- **Need a tracked file changed?** Don't edit it — record it in `HANDOFF.md` here and let the fork
  apply it.
- **Promotion to `src/neural_field/` is a handoff, not a concurrent edit** — done once, by one agent,
  when the fork's code-move has settled.
- **If this work must commit code itself** → use a worktree instead of sharing the checkout:
  `git worktree add ../emgforge-nf -b feat/learned-vc` (own branch + checkout; merge later).
- **Compute contention is real**: FEM sweeps and training both saturate CPU. Announce long runs.
