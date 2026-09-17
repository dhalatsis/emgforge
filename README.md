# emgforge

**emgforge** is an open forward model of the electromyogram. It takes tissue geometry —
a parametric limb or a segmented MRI forearm — through a volume-conductor solve to the
reciprocal **lead field** each muscle fibre sees from an electrode, turns that lead field
into **single-fibre action potentials** with a line-source model, sums fibres into
**motor-unit action potentials** (single channel or an HD-EMG grid), and drives the motor
units with a phenomenological motoneuron pool to produce **interference EMG and force**,
static or non-stationary. Every step is checkable in isolation, and is checked: against a
closed-form line-source solution, the Farina (2004) analytical cylinder, and the
quantitative EMG literature (`docs/validation/`).

```
anatomy ──▶ volume conductor ──▶ lead field φ(z) ──▶ SFAP synthesis ──▶ MUAPs ──▶ activation ──▶ EMG & force
 cylinder /   analytical or FEM    reciprocity,       direct line-source  per MU,     pool, twitch,   static /
 MRI mask     (FEniCSx + gmsh)     one solve/electrode (CSD ⋅ φ, physical  HD grid     drive          dynamic
                                                       time)
   └──▶ fibre bed (straight / harmonic streamlines) ──▶ motor-unit pool (Henneman sizes) ──┘
```

The paper describing the whole chain, its first-principles validation and the released
datasets lives on the `paper/arxiv` branch (tag `arxiv-v1` is the source submitted to
arXiv); `main` carries the code, tests and validation suite.

## Install

The FEM stack (`fenics-dolfinx`, `gmsh`) is conda-only, so the supported setup is a conda
environment first, then an editable install:

```bash
conda env create -f environment.yml
conda activate emgforge
pip install -e ".[mri]"        # [mri] adds nibabel for the MRI forearm tier
```

Everything downstream of the volume conductor (synthesis, motor units, activation) is pure
NumPy/SciPy and works with `pip install -r requirements.txt` alone.

## Quickstart

**A single-fibre action potential from a lead field** with the production recipe (direct
line-source synthesis):

```python
import numpy as np
from emgforge.synthesis import production_config
from emgforge.synthesis.engines.spatial import compute_sfap_spatial

phi = np.load("phi_along_fibre.npy")      # lead field sampled along the fibre, centred on the electrode
cfg = production_config(fs=4096.0)        # the validated recipe (v = 4 m/s, w = 256); the ONE definition
t_ms, sfap, _ = compute_sfap_spatial(phi, dz_mm=0.977, len1_mm=60, len2_mm=60, posz_mm=-20, config=cfg)
# t = 0 is the NMJ discharge; the lobe of an electrode 20 mm away lands at 5 ms, the end-of-fibre at L/v
```

`production_config()` (monopole-fit denoising, one-sided tendon window, second-derivative
current source, 2× upsampling, physical time from −10 ms) is the default everywhere:
`field_to_muap(field, bed)` and `compute_sfap_spatial(phi, dz)` with no config use it. The
Fourier engine (`MUAPConfig`) is kept for comparison only and warns when selected — it is
anti-phase and lagged against the line-source oracle (`src/emgforge/synthesis/DIRECT_LINE_SOURCE.md`).

**A motor-unit action potential** is the sum over a fibre bed (`FibreBed` carries per-fibre
semi-lengths, NMJ position and conduction velocity):

```python
from emgforge.synthesis import FibreBed, field_to_muap
bed = FibreBed.jittered(50, dz_mm=0.977, len1_mm=60, len2_mm=60, nmj_sigma_mm=5.0, seed=1)
muap = field_to_muap(phi_matrix, bed, cfg).muap      # phi_matrix: (50, Nz), one lead-field line per fibre
```

**Interference EMG and force on an HD grid** from the MRI forearm — build the grid MUAP
tensor once with the pipeline, then drive it:

```bash
python scripts/run_pipeline.py --muscle 8 --grid 5x5 --n-mu 20    # → _results/pipeline/L8_5x5_ied10_mu20/
```

```python
from emgforge import Simulator
from emgforge.activation import drive
sim = Simulator.from_pipeline("_results/pipeline/L8_5x5_ied10_mu20/pipeline_output.npz")
# sim = Simulator.from_mri(path="muap_tensor_L8_M5.npz")     # alternative: a released (N, E, w) tensor
E = drive.add_common_drive(drive.trapezoid(0.5, 0.5, 2.0, 0.5, fs=2048.0), sigma=0.02)
rec = sim.run(E, seed=0)                              # rec.emg (25, T), rec.force (%MVC), rec.spikes
```

`Simulator.from_mri(muscle=8, m=5)` without a path looks for a `run_pipeline.py` output
for that muscle and grid first; it only falls back — with a warning — to the legacy
`_results/mu_pool/electrode_grid` tensor of `scripts/activation/build_grid_tensor.py`,
whose vertex-snapped electrode grid is known to be wrong (not for results).

## One command from segmentation to EMG

```bash
python scripts/run_pipeline.py                        # FCU, 5×5 grid at 10 mm, 20 motor units, 2 workers
python scripts/run_pipeline.py --n-mu 100 --grid 8x4 --ied 8 --muscle 11 --out _results/pipeline/brachioradialis
```

`scripts/run_pipeline.py` runs the whole chain **cold** from the committed WR forearm
segmentation (`src/emgforge/mri/data/`; needs the `[mri]` extra) — nothing is read from a
cache unless `--cache` is given — and times every stage (the stages are the functions of
`emgforge.mri.pipeline`):

1. **mesh** — resample the labels, marching cubes, smooth + decimate, fTetWild tetrahedra,
   a tissue tag per cell (47 k cells, ≈ 50 s);
2. **fibre directions + muscle geometry** — PCA direction and centreline per muscle (the
   fibre-aligned σ), ray-cast cross-sections (the morphing-disk fibre frame) (≈ 7 s);
3. **fibre bed + motor-unit pool** — Poisson-disk bed of the chosen muscle (637 FCU fibres),
   Henneman pool (≈ 1 s);
4. **volume conductor, electrode grid, lead fields** — σ tensor per cell + skin shell, a
   regular M×N grid ray-cast onto the skin over the muscle, one reciprocity solve per
   electrode with φ sampled along every fibre (25 solves, ≈ 25 s);
5. **MUAPs** — the direct line-source recipe: the 3-monopole fit of every electrode–fibre
   φ(z) once, then each unit's line-source integral on the grid (the slow part: ≈ 6 min for
   20 units, ≈ 15 min for 100 with two workers);
6. **activation** — motoneuron pool → spikes, twitches → force, spikes ⊛ MUAPs → EMG on
   the grid, one trapezoid per drive level (< 1 s).

It writes `<out>/pipeline_output.npz` (electrodes, bed, pool, lead-field bank, MUAP tensor,
drive / EMG / force / spikes — `Simulator.from_pipeline(path)` reloads it) and
`<out>/pipeline_timings.json` (per-stage wall times and sizes; `--table` renders the
paper's stage table).

## Repository structure

```
src/emgforge/
├── analytical/      Farina et al. (2004) multilayer cylinder — the analytical reference
├── meshing/         gmsh builders for layered parametric limbs
├── fem/             FEniCSx reciprocity solver, tissue tensors, lead-field sampling, geometry
├── mri/core/        MRI forearm: segmentation → mesh → fibre-aligned σ; fibre beds
│                    (Poisson / hex / harmonic streamlines); Henneman motor-unit pools
├── synthesis/       lead field → SFAP → MUAP: the spatial engine (direct line-source
│                    synthesis), the Fourier reference engine, preprocessing, FibreBed;
│                    DIRECT_LINE_SOURCE.md
├── activation/      motoneuron pool, twitch/force, drive, compound EMG, dynamic EMG
├── simulator.py     the Simulator facade
└── tissue.py        conductivity constants (single source of truth)
scripts/validation/  the tiered validation suite (run_all.py → _results/validation/VALIDATION_REPORT.md)
scripts/sanity/      chain-level sanity checks
docs/validation/     PLAN.md (the checks), BIBLIOGRAPHY.md (the literature), REPORT.md (numbers)
tests/               unit tests and byte-exact regression sets
```

## Direct line-source synthesis

The SFAP is the line-source integral `SFAP(t) = ∫ i_m(z,t) φ(z) dz` with
`i_m = σ_in π a² ∂²V_m/∂z²`, the Rosenfalck action potential launched from the NMJ in both
directions and cut at the tendons. The production recipe around that integral (the
*direct method* throughout the docs and the validation report) is: a
**3-monopole fit of the lead field** (removes FEM mesh ripple before the second
derivative), a short edge taper, **2× upsampling**, the **second-derivative current
source**, a **one-sided tendon window** (flat at the NMJ) and **physical time** (t = 0 at
the NMJ, end-of-fibre at L/v). Why each step is there, with the experiments behind it, is
in [`src/emgforge/synthesis/DIRECT_LINE_SOURCE.md`](src/emgforge/synthesis/DIRECT_LINE_SOURCE.md).

## Validation

`python scripts/validation/run_all.py` (≈4 min) runs four tiers of literature-anchored
checks; the plan is [`docs/validation/PLAN.md`](docs/validation/PLAN.md), the sources
[`docs/validation/BIBLIOGRAPHY.md`](docs/validation/BIBLIOGRAPHY.md), the numbers
[`docs/validation/REPORT.md`](docs/validation/REPORT.md).

| tier | scope | status |
|---|---|---|
| A | cylinder: first principles → analytical → FEM → pipeline | 16/20 (3 known, 1 fail) |
| B | MUAP features vs the literature | 14/14 |
| C | interference EMG & motor-unit pool (released D2 bank) | 6/8 (1 known, 1 fail) |
| S | chain-level sanity (released D2 bank) | 8/8 |

"pass/total" counts every check; "known" = documented limitations kept with their real
criteria. Headlines: the spatial engine reproduces a closed-form line-source oracle at
r = 1.0000 with zero lag, a monopole-free source and (after the 2026-09-15 correction of
a spurious 1/v) the right amplitude constant; end-of-fibre onset at L/v within 0.4 ms;
CV, innervation-zone, end-of-fibre, depth, fat, electrode-size and IED laws match the
literature. Open
items: the Fourier engine is anti-phase and lagged vs first principles; on FEM lead fields
the direct-method amplitude is erratic at ±40 % (shape and spectrum are faithful); the FEM
cylinder decays ~30 % slower with depth than the analytical one; the renewal ISI model has
no refractory floor. Tiers C/S take `EMGFORGE_MUAP_BANK=<forearm_fcu_mu_pool.npz>` to run
on the released pool (median MUAP 24 µV, 18 ms on a regular 10 mm grid).

Unit-level oracle tests — the validation ladder in `tests/validation/` (plan in
[`docs/validation/README.md`](docs/validation/README.md), principles in
[`docs/validation/PRINCIPLES.md`](docs/validation/PRINCIPLES.md)) — run with the test suite.

## Datasets

Reference datasets — a cylinder SFAP atlas, the 100-unit FCU pool with its 5×5 HD-EMG
MUAP tensor, and interference-EMG trials — are generated by the scripts on the
`paper/arxiv` branch (`paper/figures/make_dataset*.py`) and described by the manifests there.

## Citing

Halatsis, D., Ezaz-Nikpay, N., Mamidanna, P., Farina, D. (2026). *emgforge: a
first-principles-validated forward model of surface EMG, from volume-conductor lead fields
to motor-unit action potentials and interference signals.* arXiv preprint (source: branch `paper/arxiv`, tag `arxiv-v1`).

## License

MIT — © 2026 Dimitrios Halatsis.
