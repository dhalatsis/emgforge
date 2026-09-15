# emgforge white paper — build plan

Target: an arXiv technical report (q-bio.QM / eess.SP), ~12–16 pages, plain and clean,
describing the whole pipeline step by step with the **golden SFAP synthesis method** at
its centre, its first-principles and literature validation, the anatomical (MRI) tier,
the motor-unit/activation layer, generated datasets, and honest limitations. Compiled
with `tectonic` from `paper/main.tex`; figures are vector PDFs in `paper/figures/`
produced by scripts in the same directory (each script is re-runnable).

Repository: https://github.com/dhalatsis/emgforge (public, MIT). Branch `paper/arxiv-v1`
(= harmonic fibres + validation suite). Python: `/home/dc23/miniconda3/envs/fenicsx-env/bin/python`.

## Title (working)

**emgforge: an open, validated forward model of surface and intramuscular EMG — from
volume-conductor lead fields to motor-unit action potentials and interference signals**

## Outline

1. Introduction — why a forward model; what exists (analytical cylinders, FEM digital
   twins, generative surrogates); the gap (open, inspectable, validated to first
   principles, anatomy-capable); contributions.
2. Overview of the pipeline (Fig 1, TikZ): anatomy → volume conductor (analytical / FEM)
   → reciprocal lead field φ(z) along fibres → SFAP synthesis (golden method) → fibre bed
   & motor-unit pool → MUAP (single / HD grid) → motoneuron pool & twitch model →
   interference EMG & force (static and non-stationary).
3. Volume conductor and lead fields
   3.1 Reciprocity; the lead field as "the potential a fibre sees from a source at the electrode".
   3.2 Analytical multilayer cylinder (Farina 2004 port): geometry, conductivities (Table 1).
   3.3 FEM (FEniCSx): mesh, tissue tensors (5:1 muscle anisotropy), Gaussian electrode
       source, one solve per electrode → all fibres; the MRI forearm (segmentation →
       mesh → fibre-aligned σ). Fig 2 (cylinder lead fields), Fig 5 (MRI).
4. The golden SFAP synthesis method
   4.1 Line-source model: SFAP = ∫ i_m φ dz, i_m = σ_in π a² ∂²V_m/∂z², Rosenfalck IAP,
       the bidirectional wave V_m(v t − |z − z_NMJ|) windowed to the fibre; the dual
       form ∫ V_m·win·φ'' dz; generation + end-of-fibre terms and the monopole-free identity.
   4.2 The recipe, step by step (Fig 3): monopole denoise of φ (why: mesh ripple through
       φ''), edge taper, 2× upsampling, CSD by numerical differentiation, one-sided
       tendon window, physical time (t = 0 at the NMJ; EOF at L/v), polarity.
       Table 2: the parameters.
   4.3 What the recipe is *not*: the Fourier (Farina 2001) engine, kept for reference.
   4.4 Multi-fibre summation, NMJ / CV / tendon scatter (FibreBed), MUAPs.
5. Fibre geometry and motor units
   5.1 Straight (Poisson-disk) fibre beds; 5.2 harmonic-streamline fibres (masked Laplace)
   — single-NMJ production model; series-fibering experimental; 5.3 Henneman pool
   (exponential sizes, territories), MUAP of a MU; Fig 6.
6. Activation and interference EMG: motoneuron pool (recruitment thresholds, onion-skin
   rate coding, renewal ISIs), twitch/force (Fuglevand), compound EMG, HD grid, SD/DD,
   non-stationary (angle-dependent) EMG; Fig 7, Fig 8.
7. Validation (the tiered suite; Table 3 scoreboard; Fig 4 first principles; Fig 9 features)
   7.1 First principles (closed-form φ, infinite anisotropic medium): r = 1.0000, zero
       lag, monopole-free, invariances. 7.2 Analytical vs FEM lead fields and pipeline.
   7.3 MUAP phenomenology vs literature (IZ, EOF, depth law, fat, electrode, IED, comb
       filter, anisotropy). 7.4 Interference/pool statistics. 7.5 MRI tier: Neurodec field
       reproduction (|r| = 1.000), containment.
8. Datasets released with the paper (Table 4): cylinder SFAP atlas; forearm FCU MU pool +
   5×5 HD-EMG MUAP tensor; interference EMG trials (static levels + dynamic).
9. Limitations and open items (honest): 1/v amplitude constant; Fourier-engine
   conventions; FEM-φ amplitude fidelity ±40 %; FEM vs analytical depth law 30 %;
   fibre geometry (C-04: MUAP scale/duration) and the series-fibering experiment;
   refractory floor; no experimental confrontation yet.
10. Conclusion. Data & code availability. Author contributions (harmonic fibres: N.
    (collaborator) — author list to be finalised by the PI). References.

## Figures (owner in brackets; all via `paper/figures/style.py`; PDF + PNG)

| # | file | content | owner |
|---|---|---|---|
| 1 | `fig_pipeline` (TikZ in main.tex) | pipeline block diagram | me |
| 2 | `fig_cylinder_leadfield` | (a) cylinder cross-section schematic with layers, electrode, fibre depths; (b) φ(z) analytical vs FEM at 3 depths; (c) transverse φ across the skin; (d) FEM lead-field map on the cross-section (one solve, evaluated on an xy grid at the electrode plane, log colour) | F1 |
| 3 | `fig_golden_anatomy` | (a) Rosenfalck V_m(z), V_m', V_m'' (tripole); (b) CSD(z,t) image with the two counter-propagating tripoles, NMJ cusp, tendon ends (boxcar) and the same with one_sided; (c) the three fibre-end windows; (d) FEM φ raw vs monopole-denoised vs analytical (+ their φ''); (e) SFAPs: raw-FEM, denoised-FEM, analytical-φ, showing the ripple removal | F1 |
| 4 | `fig_first_principles` | (a) spatial engine vs closed-form oracle at NMJ 0/−20/−30 (overlay, r=1.0000); (b) Fourier engine vs oracle (anti-phase, lag); (c) amplitude ratio vs v (= 1/v); (d) monopole-free source: Σ_z i_m(t) at machine precision | F1 |
| 5 | `fig_physical_time` | golden SFAP waterfall along a 9-electrode array on FEM φ: mono (boxcar) with the propagating lobe walking at Δz/v and the EOF pinned at L/v (annotated), and the SD montage with phase reversal at the IZ | F1 |
| 6 | `fig_mri_fibres` | (a) WR forearm segmentation slice, FCU highlighted, skin electrode grid; (b) FCU fibre beds: Poisson straight vs harmonic streamlines (single NMJ), IZ marked, 2 projections; (c) reciprocal lead field of one skin electrode on the mesh (slice) and φ along 3 fibres; (d) containment / streamline stats (small inset or numbers in caption) | F2 |
| 7 | `fig_mu_pool_muaps` | (a) MU territories in the FCU cross-section, colour = size; (b) size distribution + recruitment threshold vs index; (c) golden MUAPs of a small/medium/large MU; (d) MUAP p2p vs size (log) | F2 |
| 8 | `fig_hdemg_activation` | (a) one MU on the 5×5 grid (propagation along columns, footprint across); (b) interference HD-EMG plateau snippet on the grid + SD column; (c) recruitment/rate coding; (d) trapezoid: drive, raster, EMG, force; (e) dynamic movement: angle & EMG | F2 |
| 9 | `fig_validation_features` | compact 2×3 of tier-B results in paper style: depth power law (mono/SD/Farina/FEM), fat vs Kuiken, EOF ratio vs depth, CV recovered vs set, IZ SD array, transverse spread | F1 |

## Tables

1. Tissue conductivities and cylinder geometry (analytical and FEM).
2. The golden recipe parameters (SpatialConfig) with one-line rationale each.
3. Validation scoreboard (tiers A/B/C/S) — from `docs/validation/REPORT.md`.
4. Released datasets (name, contents, shape, size).

## Datasets (written to `_results/paper/datasets/`, manifest copied to `paper/datasets/MANIFEST.md`)

- **D1 `cylinder_sfap_atlas.npz`** [F1]: radii {33,30,27,25,20,15} mm (depth below skin
  7–25) × fibre angle {0,10,20,30,45}° × NMJ offset {0,−20,−40} mm × electrode axial
  position {−40…+40 step 10}: analytical φ, FEM φ (from the validation cache), golden
  SFAP on each, plus the Farina-generator MUAP for the same case; t axis; parameters. < 25 MB.
- **D2 `forearm_fcu_mu_pool.npz`** [F2]: the 100-MU FCU pool (fibre indices, sizes,
  territory centroids/radii, recruitment order), single-channel golden MUAPs (100×256 @
  2048 Hz), the 5×5 HD grid MUAP tensor for as many MUs as feasible (≥ 42, target 100),
  electrode xyz, t axis, metadata (density, IZ_FRAC, config).
- **D3 `interference_emg_trials.npz`** [F2]: trapezoid contractions at drive 0.1/0.2/0.35/
  0.5/0.7/1.0 (2 s hold): drive, spike trains (ragged → object or padded), single-channel
  EMG, 5×5 grid EMG, force (%MVC); one dynamic (angle-modulated) trial; fs = 2048.

## Key numbers for the text (each owner writes `paper/figures/key_numbers_<owner>.json`)

F1: r/lag of spatial vs oracle; Fourier r/lag; amplitude ratio at v=2/4; monopole
residual; A1 φ r and FWHM per depth; A2.1 r per depth; CV recovered (Farina/pipeline);
EOF onset (expected/Farina/pipeline); depth-law exponents; fat retained (analytical,
FEM golden, FEM smoothed) ; EOF ratio vs depth; IZ SD null; transverse widths.
F2: number of fibres in the FCU bed (poisson/harmonic), containment %, streamline span
stats; MU sizes min/median/max; MUAP p2p range and median duration; HD-grid IED; CV
from the grid; recruitment counts vs drive; RMS vs drive; force at drives; dynamic
RMS ratio; dataset shapes and sizes; runtimes (mesh build, solve, MUAP synthesis).

## Style rules

- `style.use()`; widths W1 = 3.4 in (single column) / W2 = 7.0 in (double); ≤ 8 pt text.
- Colours: `COL["first"]` black (first principles), `analytical` blue, `fem` orange,
  `golden` green, `fourier` red, `raw` grey; montages mono/sd/dd blue/orange/purple;
  poisson purple / harmonic green.
- Panel letters via `style.letter`; no titles inside axes (captions carry them); units
  in axis labels; time axes in ms with t = 0 at the NMJ.
- Do not modify library code under `src/`; scripts live in `paper/figures/` and read
  from the validation cache (`_results/validation/fem_cache/`) and `_results/mu_pool/`.
- Each figure script prints the numbers it plots and writes them to its key_numbers JSON.
