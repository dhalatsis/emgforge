# Scaling — and why there is (still) no cluster job here

**Status 2026-07-17: this plan's headline experiment is CLOSED, and it closed on a
workstation.** Rewritten from the original "first cluster job" framing, which the measured
costs do not support. Read `../../docs/learned_vc/RESULTS.md` first.

## What the original plan asked, and what actually happened

| the question | answer | where it was settled |
|---|---|---|
| `φ·(r+r₀)` > `raw` > `asinh` at scale? | **No — they converge.** ~6% spread on MRI, and the φ ranking that motivated the race was *noise* (36% retrain spread) | local, N=256 |
| SIREN overfits → loses? | **Was a data artifact.** SIREN PASSES on proper data (0.959), still loses to MLP+Fourier (0.996) | local, N=256 |
| `\|φ\|^α` helps monotonically? | Moot — a re-weighting patch for a starved dataset | local |
| where does N saturate? | mean at **64**; **worst case not saturated at 256** | local, free (prefix of the 256 set) |

**Every ranking the cluster was meant to adjudicate was adjudicated here, for free.** The
N-sweep cost *no new FEM at all* — the dataset's electrodes are i.i.d., so a prefix is a valid
smaller draw. The 1024-electrode extension cost **47 min** of local CPU.

## The cost case against a cluster (measured, not estimated)

| experiment | FEM (CPU) | train (GPU) | verdict |
|---|---|---|---|
| fixed anatomy, 2048 electrodes | ~25 min | ~40 min | one workstation |
| the original 48-run variant race | — | **~4 GPU-h** | **one card, overnight** |
| cross-subject, 4 subjects × 256 elec | **~19 min** | ~1 h | one workstation |

Local hardware: **12 cores, 62 GB, one RTX 2080 Ti.** Nothing above needs more.

Three independent reasons the port is the wrong investment:

1. **Nothing is cluster-scale.** The largest thing contemplated is ~1 h end-to-end.
2. **The expensive half can't go anyway.** Dataset generation needs dolfinx; the verified
   split keeps FEM local. A GPU cluster accelerates the *cheap* half.
3. **The env is the real cost.** No single env runs the pipeline: mesh generation needs
   `scifem` (pytetwild), everything else needs `fenicsx-env` (numpy<2 + torch<2.6 — the Gate
   1–4 baseline; `scifem` shifts rel-L2 ~6% on identical code). See `../../docs/learned_vc/
   ENVIRONMENTS.md`. Reproducing that remotely costs more than the compute it buys.

## What to spend the effort on instead

**Leave-one-subject-out cross-subject generalisation.** This is the 594× regime (Regime B,
varying anatomy ⇒ cache void, mesh generation dominates) *and* the actual scientific claim.
Unblocked as of today:

- **4 subjects** share one annotator's label scheme, FCU = label 8 in all: WR, DH, AG, Kostia.
- Meshing a new subject works (**68.8 s**, ASCII writer) — it was **broken** until today; the
  gmsh path fails whenever gmsh is importable and the fallback only caught `ImportError`.
- **AG's internal fat (labels 26/27) was being modelled as muscle** — 12.1% of its muscle
  volume at 32× the wrong conductivity. Fixed before any data was generated.

**Design constraints (settled — do not re-litigate):**

- Build all four on **one recorded recipe** (defaults, ~20k nodes) as a **separate set with
  its own MUAP benchmark**. Do **not** re-mesh WR in place: its recipe is unrecoverable
  (`target_z` is not the lever — 19.4k vs 21.3k nodes; the driver is `edge_length`, never
  recorded), and re-meshing would invalidate the frozen benchmark Gate 1–4 rest on. Two
  benchmarks, nothing invalidated, no subject/resolution confound.
- Score with **both** the amp-weighted mean **and the worst detectable config**. The mean
  alone is insufficient: under a "worst detectable ≥ 0.94" gate only N=256 passes, while
  N=64/128 post means of 0.989/0.994. The gate statistic decides three of four verdicts.
- **Never rank by φ.** It is non-monotonic in N, swings 36% on an identical retrain, and
  cannot separate SIREN from MLP+Fourier where the MUAP score does so decisively.

**Honest limit:** 4 subjects is a *feasibility check*, not a generalisation claim. Leave-one-out
over 4 can falsify cross-subject transfer; it cannot establish it.

## If a cluster is ever justified

It would be by **subject count**, not electrodes or variants — i.e. a real cross-subject study
needing tens of segmented forearms. That's a data-acquisition problem, not a compute one. Ship
`dataset.npz` + `muap_bench_*.npz` + `config.yaml`; get back `checkpoint.pt` + the MUAP table.
**Rule: no FEM import may leak into the trainer or the scorer** (verified: `emgforge.synthesis`
pulls no dolfinx/mpi4py/petsc4py).
