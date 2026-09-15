# Paper v2 plan — "an automated end-to-end MRI-to-EMG pipeline for general volume conductors"

Status (2026-09-15): executed. Runner `scripts/run_pipeline.py` + `pipeline_table.tex`
(Table 1); studies B and C computed (`make_fig_fibre_geometry.py`, `make_fig_crosstalk.py`,
`key_numbers_study_*.json`); manuscript restructured as below.

Reframing requested 2026-09-15: write it like NeuroDec — the object is the *pipeline*
(segmentation → mesh → lead fields → fibres → motor units → MUAPs → activation → EMG),
fully automated, for any volume conductor a segmentation describes. Explain it step by
step, justify each decision with an example, then close with studies that use it and
the conclusions they give. No comparison with NeuroDec (cite it as similar work only).
General picture over small numbers.

## Structure

1. **Introduction** — why forward models; why anatomy-general and automated; why validated
   to physics; related work (analytical, FEM, NeuroDec/NeuroMotion/BioMime/MUniverse as
   similar work); what this paper contributes (pipeline, synthesis method, validation
   suite, studies, datasets).
2. **The pipeline at a glance** — Fig 1 (MRI-first chain), one entry point, stages with
   inputs/outputs/runtimes (Table 1, from an end-to-end timed run).
3. **Step by step, with the reason for each choice**
   3.1 Anatomy from a segmentation: labels → surfaces → tetrahedral mesh → tissue tensors
       (fibre-aligned anisotropy). Example: the WR forearm mesh, cells, tissues.
   3.2 Lead fields by reciprocity: one FEM solve per electrode, sampled along every fibre.
       Example: Fig 4c/d; why the electrode source width matters (Fig 2c).
   3.3 Fibre geometry: straight Poisson beds and harmonic streamlines; example: containment
       and curvature (Fig 4b); the junction convention.
   3.4 Motor-unit pool: Henneman sizes, territories, shared fibres (Fig 5a,b).
   3.5 From lead field to action potential: direct line-source synthesis, each step with
       its example (Fig 3: monopole fit removes ripple; one-sided window; physical time Fig 8).
   3.6 Activation: pool, twitch/force, drive, montages, non-stationary (Fig 6).
4. **Validation** (condensed): first principles (Fig 7), analytical/FEM cylinder (Fig 2),
   MUAP phenomenology (Fig 9), interference statistics; scoreboard; appendix table.
5. **Studies with the pipeline**
   5.1 What the electrode sees: depth, fat, IED, electrode size, montage (tier B) →
       conclusions for HD-EMG design.
   5.2 Fibre geometry: straight vs harmonic beds in the same muscle — NEW computation.
   5.3 Crosstalk: a grid over one muscle sees the units of its neighbours — NEW computation
       (demonstrates "any muscle in the segmentation").
   5.4 Interference EMG across drive: cancellation, EMG–force, spectra (tier C, D3).
6. **Discussion and outlook** — what the pipeline is for; limitations (condensed); next.
7. Data availability, funding, references, Appendix A (checks).

## New computations (agents)
- A: end-to-end runner `scripts/run_pipeline.py` (segmentation → EMG in one command) with
  per-stage timings → Table 1; verify nothing in the chain is manual.
- B: straight vs harmonic fibre beds: same FCU pool, MUAPs on the grid column → shape,
  amplitude, EOF, CV read-off → `fig_fibre_geometry`.
- C: crosstalk: pools in the muscles neighbouring the FCU, MUAPs on the FCU grid → RMS
  maps and crosstalk ratio vs distance → `fig_crosstalk`.

## Remove
- NeuroDec reproduction (abstract, §7.4, limitations, README line); keep the citation.
