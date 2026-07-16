# Learned VC — implementation plan (incremental)

Companion to `PLAN.md` (design/options). This is the **build order**: what to implement, what data
to generate, what to run, and the gate that must pass before moving on.

**Principles**
1. Every phase ends in a *measurable gate*. No phase starts before the previous gate is green.
2. **Cylinder before MRI.** Prove the whole loop on the easy case (no mesh needed, analytic σ, an
   independent analytical ground truth) before taking on real anatomy.
3. **Models are judged on MUAPs, not on φ.** A frozen benchmark (Phase 1) is the scoreboard.
4. **Small here, scaled on the cluster.** See §Compute tiers.

---

## Phase 0 — foundation

| # | Implement | Status |
|---|---|---|
| 0.1 | Port `train_neural_field.py`, `train_pinn.py`, `models/`, `data/pointcloud_dataset.py`, `configs/exp[1-4].yaml` from `neural-forward-emg/training/neural_field/` | todo (handoff: `src/neural_field/` is the fork's) |
| 0.2 | `torch` as optional extra in `pyproject.toml` (`[project.optional-dependencies] nf`) | todo (handoff) |
| 0.3 | Re-baseline FEM cost | ✅ **done** |

**E0 — reproduce cylinder Exp 1. Gate:** rel L2 ≈ 0.029, R² ≈ 0.999, MUAP corr ≈ 0.942.
Can't reproduce ⇒ the port is wrong; stop and fix.

### Gate 0b — the honest speed question. ✅ RUN 2026-07-16 — and it fired.

`scripts/neural_field/00_baseline_fem_timing.py`, WR, 127k fibre query points:

| | measured |
|---|---|
| model build (one-time) | 3.90 s |
| `solve_for_point` / electrode | **0.39 s** |
| `compute_closest_entity` (point→cell) | **7.17 s** ← **99.5% of the cost** |
| `uh.eval` (interpolate) | **0.036 s** ← the only part needing the solve |

**The FEM's cost is point-location, not the solve** — and `cell_ids` depends only on
`(mesh, points)`, **not** the solution, so for a fixed query set it's cacheable.
**Verified: byte-identical φ (max|Δ| = 0.00e+00), 165× faster sampling.**

| per electrode | naive | **cached** |
|---|---|---|
| total | 7.6 s | **0.43 s** (~18×) |
| 2048-electrode dataset | 259 min | **15 min** |

**⚠️ Consequence — the speed justification is largely gone.** The prior **75×** was measured against
an FEM redundantly re-locating the same points. Against a cached FEM a 0.15 s net is only **~3×**.
The learned VC must be justified by **differentiability**, **mesh-free deployment**, or
**generalisation** — not speed. **Decide before Phase 4.** Silver lining: dataset gen is now cheap.

---

## Phase 1 — the frozen MUAP benchmark (the scoreboard) ← NEW

Purpose: one fixed, committed set of MUAP-generating configurations + their FEM-truth MUAPs **and
derived properties**, so every model/architecture is scored on the same physiological yardstick.
Without this, "which model is best" is unanswerable — φ error already proved misleading.

| # | Implement | Notes |
|---|---|---|
| 1.1 | `muap_bench` spec: a frozen list of K configs — MU fibre bed (IZ 0.305, asymmetric Lp/Ld), electrode positions, CV, spatial config | Freeze the seeds. This is a fixture, not a rerun. |
| 1.2 | Build the truth: FEM φ → `field_to_muap` → waveform **+ properties** | properties = `p2p`, `duration_ms`, `jaggedness`, `lobe_metrics` (trough, pos-before, **pos-after = EOF**), peak latency |
| 1.3 | `score_model(model) → table` | per config: `align_score` r, p2p ratio, duration err, **Δjaggedness**, EOF err |
| 1.4 | Commit as a small fixture (K × 256 floats + configs ≈ tens of KB) | mirrors `tests/synthesis/data/golden_cylindrical.npz` |

**Datasets:** `muap_bench_cyl.npz` (Phase 2 scoreboard) → later `muap_bench_mri.npz` (Phase 4+),
plus **`pm_gold`** (105 PM electrodes + Neurodec MUAPs — already exists) as the external bar.

**Gate 1: ✅ PASS (2026-07-16).** `01_build_muap_bench.py` → `_results/neural_field/muap_bench_cyl.npz`.
**27 configs** (9 electrodes × 3 MU depths), 125 s. A rerun reproduced **every array
byte-for-byte** (`t_ms, muap, cfg, p2p, duration_ms, jaggedness, eof, latency, trough`).

Frozen spec: cylinder `r_bone=10/r_muscle=35/r_fat=38/r_skin=40, L=240`; MU depths 8/15/25 mm
below skin; electrodes θ∈{0,20,40}° × z∈{90,120,150} mm; 20 fibres in a 2 mm disk (`MU_SEED=7`);
fibre 200 pts × 1.07 mm; **MU-113-like asymmetric geometry** `Lp=65 / Ld=148 / posz=−41.5`.

**Sanity checked (important):** the weak configs are **real MUAPs, not FEM noise** — 3–7
zero-crossings across all 27 (noise would be dozens). Their long durations (~35 ms) are the
*physically correct* deep-source signature: broad φ smears the MUAP, exactly as the field
cross-reference predicted. Range spans **8.7 µV (superficial, under the electrode) → 0.14 µV
(deep, off-axis)** — a real difficulty gradient, which is what a benchmark wants.

⚠️ Metric caveat: `eof` (= `lobe_metrics` pos-after, normalised) saturates at 0.00/1.00 when the
global peak lands after the trough. Fine as a *diff* target, but don't read it as a ratio.

---

## Phase 2 — cylinder, multi-electrode: the full loop on the easy case ← NEW

Purpose: dataset → train → **MUAP benchmark** end-to-end where nothing anatomical can bite. The
cylinder needs **no mesh for σ** (analytic, `neural_field/pointcloud/analytical.py`) and has an
**independent analytical ground truth** (`emgforge.analytical`) — two truths to check against.

| # | Implement | Notes |
|---|---|---|
| 2.1 | Multi-electrode array on the cylinder surface (M×M grid + a longitudinal sweep) | mirrors the HD-EMG geometry we now use on MRI (`scripts/mri/mu_electrode_grid.py`) |
| 2.2 | Dataset gen (reuse `scripts/neural_field/06_generate_pointcloud_electrode.py`) | mesh `cyl_10_35_38_40.msh` already cached |
| 2.3 | Train `u(x \| electrode)`; **score on `muap_bench_cyl`** | MLP+Fourier vs SIREN vs SIREN+relcoords |

**Datasets:** `cyl_elec_dev` (64 electrodes) → `cyl_elec_full` (~2048, matches Exp 1).

**Experiments:** E2.1 dev overfit · E2.2 data-scaling (64→256→2048) · E2.3 architecture ablation
· E2.4 **multi-electrode coherence**: does the *grid* stay spatially consistent (propagation/IZ
intact across cells), not just each electrode in isolation?

**Gate 2:** MUAP corr ≥ **0.94** on held-out electrodes *and* Δjaggedness ≈ 0 *and* the HD-EMG grid
is spatially coherent. **Only then do we touch MRI.**

---

## Phase 3 — the MRI dataset

| # | Implement |
|---|---|
| 3.1 | `sample_skin_electrodes(n, seed)` — stratified over (θ, z_frac), whole forearm not just FCU |
| 3.2 | `build_mri_pointcloud` — per electrode: solve → φ at volume points (**use cached `cell_ids`**) |
| 3.3 | φ along the FCU bed fibres per electrode (the MUAP-eval companion) |
| 3.4 | Splits: hold out electrodes; plus `pm_gold` |

**Datasets:** `mri_elec_dev` **64** (~30 s cached) · `mri_elec_med` **256** · `mri_elec_full`
**1024–2048** (~15 min cached) · `mri_fibre_eval` · `pm_gold` (exists).

**Gate 3:** reloaded φ matches a fresh FEM solve byte-for-byte.

## Phase 4 — E1-MRI: supervised, per-subject (option E)

`u(x,y,z | electrode_xyz) → φ` on WR. MLP+Fourier. Score on `muap_bench_mri` + `pm_gold`.
**Gate 4:** MUAP corr vs FEM ≥ 0.94, Δjaggedness ≈ 0. **Stretch:** learned-φ→MUAP vs Neurodec ≈
FEM-φ→MUAP vs Neurodec (the surrogate loses nothing against the external truth).

## Phase 5 — σ(x) on MRI (unlocks the PINN)

3.1 voxel lookup from `forearm_WR_segmentation.nii.gz` (77 KB) · 3.2 **σ as an anisotropic
TENSOR** (per-muscle fibre orientation — the step the cylinder never had) · 3.3 smoothing
(nearest-label σ is discontinuous → bad PINN gradients; trilinear-on-smoothed or implicit/SDF).
**Gate 5:** agrees with the FEM's own cell-tag σ **including tensor orientation**.

## Phase 6 — E2-MRI: PINN on real anatomy

`∇·(σ∇u) = f`, Neumann BC, σ from Phase 5. Exp 4 got 98.5% of supervised with **zero FEM labels**
on the cylinder — does that survive a discontinuous, anisotropic σ?
**Gate 6:** hybrid ≥ supervised at equal FEM budget, or pure PINN ≥ 0.9 MUAP corr with zero labels.

## Phase 7 — cross-subject (only if 1–6 land)

Anatomy encoder → latent z → `u(x | z, electrode)`. Archive has DH/AG/Kostia/PM/WR label volumes.

---

## Compute tiers — small here, scaled on the cluster

**The architectural constraint that makes this work:** the **trainer must never import dolfinx**.
Split the stack by dependency:

| stage | needs | runs |
|---|---|---|
| FEM dataset generation | dolfinx + mesh + segmentation | **local** (CPU; cheap now — 15 min for 2048 electrodes) |
| training | torch only + the `.npz` | **cluster** (GPU) |
| MUAP scoring | `emgforge.synthesis` — **pure NumPy, no FEM stack** | **either** ✅ |

`emgforge.synthesis` is dolfinx-free (its whole test suite is pure NumPy), so **the MUAP benchmark
can be scored on the cluster** — models are ranked by MUAP quality *in situ*, not just by val loss.
That's the key enabler; protect it (no FEM imports leaking into the trainer or the scorer).

**Interface = a self-contained artifact:** `dataset.npz` + `config.yaml` + `muap_bench_*.npz` →
ship → train → checkpoint + metrics come back. No mesh, no dolfinx, no MRI on the cluster.

**Tier A — local (here):** benchmark harness, dev datasets (64), overfit sanity, architecture
ablation at small N, all plumbing/debug. Goal: *get the loop right*.

**Tier B — cluster (scaled):** full datasets (2048+ electrodes, multi-MU, multi-subject), long
training, hyperparameter sweeps, the architecture matrix at full N, PINN collocation at scale.
Goal: *the numbers that go in the paper*.

**To set up (needs input):** scheduler (slurm?), how data gets there, GPU/env availability, whether
torch+CUDA is provisioned. Deliverable: a `scripts/neural_field/cluster/` with a submit script and
an env spec that installs **torch only** (no dolfinx).

---

## Order of the next increments

1. **1.1–1.4** the frozen MUAP benchmark (cylinder) → Gate 1. *Do first — it referees everything after.*
2. **0.1** port the trainers (handoff w/ the fork).
3. **E0** reproduce cylinder Exp 1 → the port gate.
4. **2.1–2.3** cylinder multi-electrode, dev scale → **Gate 2**.
5. Only then Phase 3 (MRI).
6. In parallel: stand up the cluster path (env spec + submit script) so Phase 2's full-scale run is the first cluster job.

## Open decisions

- **Speed is out (Gate 0b). Is the goal differentiability, mesh-free deployment, or generality?**
  This changes what "success" means and should be settled before Phase 4.
- **Per-subject (E) or cross-subject (D)?** Phases 0–6 assume per-subject; D is a different project.
- **Train on volume points or fibre points?** Plan trains on volume, evaluates on fibres; fibre-only
  is task-specific but might reach the MUAP gate with far less data.
