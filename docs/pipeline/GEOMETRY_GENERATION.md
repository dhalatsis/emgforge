# Geometry generation parameters

This document describes the parameters and CLI flags used by `scripts/00_generate_meshes.py` (and the underlying sampler/mesher) to generate parametric “limb-like” meshes.

## Quick start examples

### 1) Baseline: concentric circular cylinder (original)

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape circle \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_circle
```

### 2) Elliptical limb (nested ellipses)

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape ellipse --ellipse_ratio_min 1.1 --ellipse_ratio_max 1.4 \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_ellipse
```

### 3) Off-center bone (one-bone, asymmetric inner anatomy)

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape circle \
  --bone_count 1 --bone_offset_frac_max 0.15 \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_offcenter
```

### 4) Two-bone anatomy (radius + ulna style)

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape circle \
  --bone_count 2 \
  --interbone_frac_min 0.35 --interbone_frac_max 0.60 \
  --bone2_scale_min 0.80 --bone2_scale_max 1.20 \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_two_bone
```

### 5) Longitudinal taper (z-varying cross-section)

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape circle \
  --z_profile taper --taper_scale_min 0.75 --taper_scale_max 0.95 \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_taper
```

You can combine features (e.g. ellipse + two bones + taper) as long as they’re feasible:

```bash
PYTHONPATH=src python3 scripts/00_generate_meshes.py \
  --n 10 --seed 0 \
  --shape ellipse --ellipse_ratio_min 1.2 --ellipse_ratio_max 1.6 \
  --bone_count 2 \
  --interbone_frac_min 0.35 --interbone_frac_max 0.55 \
  --bone2_scale_min 0.85 --bone2_scale_max 1.15 \
  --z_profile taper --taper_scale_min 0.80 --taper_scale_max 1.00 \
  --out_dir ${DATA_ROOT:-./data}/generated_meshes_combo
```

---

## Output layout

Running `scripts/00_generate_meshes.py` writes:
- `meshes/sample_XXXXXX.msh`: the Gmsh mesh with physical groups (tissue subdomains).
- `metadata/sample_XXXXXX.json`: geometry parameters + meshing stats.
- `manifest.csv`: one row per sample with common columns for quick indexing.

The per-sample JSON includes `geometry_params` containing everything sampled below.

---

## Geometry “families” and what parameters mean

### Cross-section shape

- **`--shape`**: `circle` or `ellipse`
  - `circle`: nested circular layers.
  - `ellipse`: nested axis-aligned ellipses, using a single aspect ratio shared by all layers.

- **`--ellipse_ratio_min`, `--ellipse_ratio_max`** (ellipse only):
  - Samples `ellipse_ratio = a/b` (x-axis radius / y-axis radius).
  - Interpretation in metadata:
    - `a_* = ellipse_ratio * r_*`
    - `b_* = r_*`

### Layer radii / thicknesses (sampled, then “derived”)

Internally, the sampler draws:
- `radius_canc_bone`
- `thickness_cort_bone`
- `thickness_muscle`
- `thickness_fat`
- `thickness_skin`

Then derives outer radii:
- `radius_cort_bone = radius_canc_bone + thickness_cort_bone`
- `radius_muscle    = radius_cort_bone + thickness_muscle`
- `radius_fat       = radius_muscle + thickness_fat`
- `radius_skin      = radius_fat + thickness_skin`

For ellipses, the same logic is applied to `a_*`/`b_*` axes.

### Length / aspect

- **`--factor`**:
  - Global scaling factor (world units). Think of it as “limb size scale”.
  - Also controls absolute mesh size behavior (larger geometries tend to produce bigger meshes given the same mesh options).

- **`--length_ratio_min`, `--length_ratio_max`**:
  - Samples `length_ratio` and sets:
    - `length = factor * length_ratio`

### Bone count / placement

- **`--bone_count`**: `1` or `2`
  - `1`: single cancellous+cortical bone core.
  - `2`: two cancellous+cortical cores (radius + ulna style).

- **`--bone_offset_frac_max`**:
  - For `bone_count=1`: shifts bone center `(bone_offset_x, bone_offset_y)` from origin.
  - For `bone_count=2`: shifts the *pair center*; bones are then placed at `±sep/2` around that.
  - Unit: fraction of outer size (approximately `r_skin`), converted to world units.
  - The sampler clamps offsets to keep the cortical region inside muscle.

Two-bone parameters:
- **`--interbone_frac_min`, `--interbone_frac_max`**:
  - Controls separation between the two bone centers.
  - Sampled “fraction” is scaled by muscle size to produce an absolute separation.
  - The sampler also enforces **non-intersection** of the two cortical shells (with a safety margin), and may shrink bone2 if needed.

- **`--bone2_scale_min`, `--bone2_scale_max`**:
  - Relative size of bone2 cancellous radius (bone1 is baseline).
  - Cortical thickness is shared (bone2 cortical radius is cancellous + thickness).

Metadata fields (two-bone):
- `bone1_center_x/y`, `bone2_center_x/y`
- `radius_canc_bone_2`, `radius_cort_bone_2`
- (ellipse) `a_canc_bone_2`, `b_canc_bone_2`, `a_cort_bone_2`, `b_cort_bone_2`
- `interbone_sep`, `bone_angle`, `bone2_scale`

### Longitudinal variation (taper)

- **`--z_profile`**: `constant` or `taper`
  - `constant`: cross-section does not change along z.
  - `taper`: linearly scales all radii/axes along z from `z=0` to `z=L`.

- **`--taper_scale_min`, `--taper_scale_max`** (taper only):
  - Samples `taper_scale_z1` (scale at z=L).
  - z=0 is fixed at `taper_scale_z0 = 1.0`.
  - Meaning: a value < 1 tapers smaller toward the distal end; > 1 flares outward.

Implementation notes:
- `circle+taper`: built with Gmsh OCC cones.
- `ellipse+taper`: loft between two ellipse cross-sections.

---

## Meshing controls (quality vs speed)

- **`--mesh_char_factor`**:
  - Gmsh `Mesh.CharacteristicLengthFactor`.
  - Larger → fewer elements (faster), smaller → more elements (slower, more accurate).

- **`--refine_on`**: `skin` or `all_interfaces`
  - `skin`: refinement guided by outer boundary.
  - `all_interfaces`: refinement guided by every tissue boundary (usually much slower).

---

## Practical tips

- **Debugging geometry**:
  - The mesher has a `debug_gui` param in metadata (not wired to CLI yet). If enabled, it can pop the Gmsh GUI; keep this off for batch generation.

- **Performance**:
  - Two-bone + taper can substantially increase meshing time; start with a higher `--mesh_char_factor` (e.g. `0.4`) when iterating.

- **Physical groups**:
  - Labels remain compatible with the FEM solver:
    - `Cancellous Bone` (id=1)
    - `Cortical Bone` (id=2)
    - `Muscle` (id=3)
    - `Fat` (id=4)
    - `Skin` (id=5)

---

## Reference: CLI flags in `scripts/00_generate_meshes.py`

- **Sampling / geometry**
  - `--n`, `--seed`
  - `--factor`
  - `--length_ratio_min`, `--length_ratio_max`
  - `--shape` (`circle|ellipse`)
  - `--ellipse_ratio_min`, `--ellipse_ratio_max`
  - `--bone_count` (`1|2`)
  - `--bone_offset_frac_max`
  - `--interbone_frac_min`, `--interbone_frac_max`
  - `--bone2_scale_min`, `--bone2_scale_max`
  - `--z_profile` (`constant|taper`)
  - `--taper_scale_min`, `--taper_scale_max`

- **Meshing**
  - `--mesh_char_factor`
  - `--refine_on`

- **I/O**
  - `--out_dir`
  - `--limit`
  - `--verbose`

