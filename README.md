# emgforge

**emgforge** is a biomedical simulation library for generating **synthetic
EMG / MUAP forward-model data**. It takes parameterized tissue geometry through
a **finite-element (FEM) volume-conductor** solve to obtain a reciprocal lead
field, then synthesizes **single-fibre action potentials (SFAP)** and sums them
into **motor-unit action potentials (MUAP)** and high-density surface EMG
(HD-sEMG). It is aimed at researchers who need physically-grounded, labelled EMG
data — for example to train and benchmark deep-learning models for EMG decoding.

## The pipeline

```
tissue geometry ──▶ FEM volume ──▶ lead field ──▶ SFAP ──▶ Σ over fibres ──▶ MUAP / HD-sEMG
  (gmsh mesh)       conductor      φ(z)           (per fibre)  (+ NMJ / CV /
                    solve                                        tendon scatter)
```

The lead field `φ(z)` — the potential a fibre sees from a source at the electrode
(by reciprocity) — is the hinge of the whole method. It can come from three
**field sources**, and be turned into a MUAP by either of two **integration
engines**:

| Integration engine | FEM 5-layer cylinder | Analytical Farina 4-layer | MRI-FEM forearm |
|---|---|---|---|
| **Fourier** (Farina radon) | ✅ production, array-wide | ✅ reference | ✅ anatomy |
| **Spatial** (CSD @ φ, time-domain) | cross-check¹ | ✅ r ≈ 0.997 | limited¹ |

¹ On clean cylinder/analytical φ the two engines agree at r ≈ 0.997; for
array-wide synthesis the Fourier engine is the more robust choice (see
`muap_generator/spatial_refactor/README.md`).

## Install

The FEM stack (`fenics-dolfinx`, `gmsh`) is **conda-only** — it is not on PyPI —
so the supported setup is conda first, then an editable pip install:

```bash
# 1. Create + activate the conda env (brings in dolfinx, gmsh, and the rest)
conda env create -f environment.yml
conda activate emgforge

# 2. Install emgforge itself (makes `emgop` and `muap_generator` importable)
pip install -e .
```

Pip-only users can `pip install -r requirements.txt` to get the pure-Python parts
(the SFAP/MUAP synthesis), but the FEM volume-conductor solve needs the conda env.

Scripts that read or write bulk data default their data root to `./data`; override
it with the `DATA_ROOT` environment variable.

## Quickstart

Given a matrix of lead-field lines `φ(z)` (one row per fibre) and the spatial
sampling step `dz_mm`, synthesize a MUAP:

```python
import numpy as np
from muap_generator import generate_muap_from_phi, MUAPConfig

# phi_mat: (n_fibres, n_z) array of lead-field lines phi(z), one row per fibre.
# dz_mm:   spatial sampling step along the fibre axis, in mm.
phi_mat = np.load("phi_lines.npy")          # from an FEM solve or the analytical model
result = generate_muap_from_phi(phi_mat, dz_mm=0.977, config=MUAPConfig())

print(result.t_ms.shape, result.muap.shape)  # time axis + MUAP waveform
print(result.metrics["peak_to_peak"])
result.plot()                                 # quick matplotlib view
```

Other entry points:

```python
from muap_generator import generate_muap_from_npz          # from a voxel-grid .npz
from muap_generator.spatial_refactor import compute_sfap_spatial  # time-domain engine
```

The validated production defaults and the full parameter reference live in
[`muap_generator/PIPELINE.md`](muap_generator/PIPELINE.md).

## Repository structure

```
emgforge/
├── src/emgop/            # FEM volume-conductor toolkit
│   ├── fem/              #   FEniCSx point-source solver, electrode configs, sanity checks
│   ├── meshing/          #   gmsh multi-layer cylinder mesh builders
│   ├── pointcloud/       #   point-cloud lead-field sampling + analytical model
│   ├── sampling/         #   Latin-hypercube parameter sampling
│   └── voxel/            #   voxelization of FEM fields
├── muap_generator/       # phi(z) -> SFAP -> MUAP synthesis
│   ├── api.py            #   high-level generate_muap_from_phi / _npz
│   ├── fourier.py        #   Fourier (Farina radon) integration engine
│   ├── preprocessing.py  #   smoothing / resampling / edge tapering
│   ├── spatial_refactor/ #   spatial (CSD @ phi) time-domain engine + verification
│   └── PIPELINE.md       #   the validated recipe + parameter reference
├── mri/core/             # MRI-FEM forearm anatomy pipeline
├── scripts/              # end-to-end mesh -> FEM -> voxel driver scripts
├── tests/                # unit tests + regression bench
├── docs/pipeline/        # FEM, running, geometry and workflow docs
├── pipeline.sh, run.sh   # batch drivers
└── environment.yml, pyproject.toml, requirements.txt
```

## Validation

- **analytical operator:** analytical cylinder φ passed through the public
  production pipeline reproduces the independently assembled Farina waveform to
  machine precision in signed shape and SI-scaled amplitude.
- **vs cylinder FEM:** the established five-layer FEM campaign reaches mean Pearson
  **r ≈ 0.99 (0.990–0.997)** against the four-layer cylindrical analytical model
  across muscle-region fibre depths; a like-for-like convergence tier is specified
  in the validation plan.
- **regression bench:** the 200-case snapshot bench passes its **90/90** sanity
  gate; the spatial engine matches the Fourier reference at **r = 0.997** on the
  12-case golden cylindrical set.
- The remaining analytical-vs-FEM gap (~5%) is a physical model difference
  (finite vs infinite cylinder, Gaussian vs point source, 5- vs 4-layer), not a
  pipeline artifact.

The [validation plan](docs/validation/README.md) defines the automated gates and
the [research bibliography](docs/validation/BIBLIOGRAPHY.md) maps forward-model
evidence to MUAP sanity checks.

Snapshots and meshes used by the heavier regression tests are regenerable and are
kept out of the repository; the unit tests under `tests/` run without them.

## References

- Maksymenko, K., et al. (2022). *A myoelectric digital twin for fast and
  realistic modelling in deep learning.* **Nature Communications.** — the
  digital-twin reference this work validates its forward model against.
- Farina, D., Mesin, L., Martina, S., & Merletti, R. (2004). *A surface EMG
  generation model with multilayer cylindrical description of the volume
  conductor.* **IEEE Trans. Biomed. Eng.**, 51(3), 415–426.
- Rosenfalck, P. (1969). *Intra- and extracellular potential fields of active
  nerve and muscle fibres.* **Acta Physiol. Scand.**, 321, 1–168.

## License

Released under the [MIT License](LICENSE). © 2026 Dimitrios Halatsis.
