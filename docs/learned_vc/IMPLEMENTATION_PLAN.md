# Learned VC on MRI — implementation plan (incremental)

Companion to `PLAN.md` (the design/options doc). This is the **build order**: what to
implement, what data to generate, what to run, and the gate that must pass before moving on.

**Principle:** every phase ends in a *measurable gate*. No phase starts before the previous
gate is green. Phases 0–2 are engineering with known answers; the science starts at 3–4.

---

## Phase 0 — foundation (no new science, ~1 day)

Purpose: get the closed cylinder work running *in this repo* so we have a trustworthy floor.

| # | Implement | Notes |
|---|---|---|
| 0.1 | Port `train_neural_field.py`, `train_pinn.py`, `models/{neural_field,pinn}.py`, `data/pointcloud_dataset.py`, `configs/exp[1-4].yaml` from `neural-forward-emg/training/neural_field/` → `src/neural_field/{models,training}/` + `scripts/neural_field/` | **Do NOT import checkpoints** (273 MB). Coordinate with the fork — `src/neural_field/` is theirs to reorganise (`HANDOFF.md`). |
| 0.2 | `torch` as an optional extra in `pyproject.toml` → `[project.optional-dependencies] nf = ["torch", ...]` | Keeps the core FEM/synthesis install light. Handoff item. |
| 0.3 | Re-baseline timing on WR | FEM: measure solve + φ-sample separately. Net: inference for the same query set. |

**Dataset:** none (reuse the archive's cylinder point clouds if present, else regenerate with the
existing `scripts/neural_field/06_generate_pointcloud_electrode.py`).

**Experiment E0 — reproduce cylinder Exp 1.**
**Gate:** rel L2 ≈ **0.029**, R² ≈ **0.999**, MUAP corr ≈ **0.942** (MLP + Fourier features).
If we can't reproduce it, the port is wrong — stop and fix.

**Gate 0b — the honest speed question.** Prior claim: 150 ms/electrode vs **13 s** FEM = 75×.
Today WR solves in **~0.3 s** (+ ~6 s to sample φ along 637 fibres). Re-measure and write the
number down. *If the speedup is now <5×, the justification must shift to differentiability /
mesh-free inference — decide before Phase 2.*

---

## Phase 1 — the MRI dataset (the real new work)

Purpose: the electrode-conditioned supervised dataset on **WR** anatomy.

| # | Implement | Notes |
|---|---|---|
| 1.1 | `sample_skin_electrodes(n, seed)` → n positions on the WR skin | Stratified over (θ, z_frac). Reuse `MRIFEMModel.get_skin_surface_point`. Must cover, not just cluster over FCU. |
| 1.2 | `build_mri_pointcloud(electrodes, n_points)` | Per electrode: `solve_for_point` → `evaluate_solution_at_points` at **volume** points. Reuse `neural_field.pointcloud.{sampling,generate,evaluate}` — the format already exists. |
| 1.3 | Fibre-eval companion: φ along the **FCU bed fibres** per electrode | This is what MUAPs are computed from → needed for the MUAP metric, separate from the volume training points. |
| 1.4 | Splits | Hold out electrodes (never points from a trained electrode). Plus the **PM gold set** below. |

**Datasets to generate:**

| name | size | cost (est.) | purpose |
|---|---|---|---|
| `mri_elec_dev` | **64** electrodes × ~50 k volume pts | ~3 s/elec → **~3 min** | fast iteration, shape/plumbing debug |
| `mri_elec_med` | **256** electrodes | ~15 min | data-scaling curve |
| `mri_elec_full` | **1024–2048** electrodes | ~1–2 h | the real training set (matches Exp 1's 2016) |
| `mri_fibre_eval` | 637 FCU fibres × 200 pts, per held-out electrode | reuse the solve | MUAP metric |
| `pm_gold` | **105 PM electrodes + Neurodec MUAPs** | **already exists** | the external bar |

FEM cost is no longer the bottleneck (0.3 s/solve) — 2048 electrodes is ~1–2 h, not days.

**Gate 1:** reload the dataset and confirm φ at a random sample matches a **fresh FEM solve**
bit-for-bit. If the dataset is wrong, everything downstream is noise.

---

## Phase 2 — E1-MRI: supervised, per-subject, electrode-conditioned

The direct analogue of Exp 1, on real anatomy. **Option E** in `PLAN.md` (no anatomy conditioning
— one net for the WR arm). This is the smallest thing that is actually useful: it replaces
`solve_for_point`.

**Implement:** `u(x, y, z | electrode_xyz) → φ`, MLP + random Fourier features (the prior winner).

**Experiments:**
- **E1.1** train on `mri_elec_dev` → sanity (does it fit at all?)
- **E1.2** data-scaling: dev → med → full; plot each metric vs N
- **E1.3** architecture check: MLP+Fourier vs SIREN vs SIREN+relcoords — *confirm the prior
  finding holds on MRI* (SIREN wins pointwise, loses on MUAPs)

**Metrics — all four, always:**
1. φ rel-L2 and R² (pointwise)
2. **MUAP corr vs the FEM MUAP** ← the metric that matters
3. **`jaggedness(φ)`** (`emgforge.synthesis.metrics`) ← smoothness proxy; SFAP ∝ φ″
4. inference time/electrode

**Gate 2:** MUAP corr vs FEM ≥ **0.94** on held-out electrodes (parity with the cylinder result),
*with* jaggedness no worse than the FEM field's. Pointwise R² alone does **not** pass the gate —
the prior work already showed R²=0.999 coexisting with bad MUAPs.

**Stretch gate:** learned-φ → MUAP vs **Neurodec** on `pm_gold` ≈ FEM-φ → MUAP vs Neurodec.
i.e. the surrogate loses nothing against the external ground truth.

---

## Phase 3 — σ(x) on MRI (unlocks the PINN)

The crux from `PLAN.md` §2: the cylinder had analytic σ from radii; MRI does not.

| # | Implement | Notes |
|---|---|---|
| 3.1 | `sigma_at_points(x)` from the **segmentation** (option B) | Voxel lookup in `forearm_WR_segmentation.nii.gz` (77 KB) → label → σ. Mesh-free. |
| 3.2 | **σ as a TENSOR** | Muscle σ is anisotropic along each muscle's fibre direction (`forearm_WR_fibers.json`, `sigma_mode="centerline"`). This is the step the cylinder never had. |
| 3.3 | Smoothing | Nearest-neighbour label lookup gives a *discontinuous* σ → bad PINN gradients. Options: trilinear on a smoothed σ grid, or an implicit/SDF fit (option C). |

**Gate 3:** `sigma_at_points` agrees with the FEM's own cell-tag σ (the thing `MRIFEMModel`
actually assembles) at random points — **including the tensor orientation**, not just the scalar.

---

## Phase 4 — E2-MRI: PINN on real anatomy

`∇·(σ∇u) = f`, Neumann BC, σ from Phase 3. Exp 4 got **98.5 % of supervised with zero FEM labels**
on the cylinder — the question is whether that survives a real, discontinuous, anisotropic σ.

**Experiments:** E4.1 pure PINN · E4.2 hybrid (PINN + supervised) · E4.3 vs Phase 2 supervised.
**Gate 4:** hybrid ≥ supervised at equal FEM budget, *or* pure PINN ≥ 0.9 MUAP corr with zero labels.

---

## Phase 5 — cross-subject (only if 1–4 land)

Option D: anatomy encoder → latent z → `u(x | z, electrode)`. Needs ≥2 subjects; the archive has
DH/AG/Kostia/PM/WR label volumes. This is where it stops being a surrogate and becomes a model.

---

## Order of the first increments (start here)

1. **0.3** re-baseline FEM vs net timing → decides whether the whole thing is speed-motivated. *Cheap, do first.*
2. **0.1** port the trainers (coordinate with the fork via `HANDOFF.md`).
3. **E0** reproduce cylinder Exp 1 → the port gate.
4. **1.1–1.3** + `mri_elec_dev` (64 electrodes, ~3 min) → Gate 1.
5. **E1.1** overfit the dev set → does the field learn on MRI at all?

## Open decisions that change the plan

- **Per-subject (E) or cross-subject (D)?** Phases 0–4 assume per-subject. D is a different project.
- **Is the goal speed, differentiability, or generality?** Gate 0b may kill the speed story.
- **Volume points or fibre points for training?** Plan trains on volume, evaluates on fibres.
  Training directly on fibre points is task-specific but might reach the MUAP gate with far less data.
