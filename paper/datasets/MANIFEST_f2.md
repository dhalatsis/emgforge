# F2 datasets — D2 `forearm_fcu_mu_pool.npz`, D3 `interference_emg_trials.npz`

Written by `paper/figures/make_datasets_d2_d3.py` to `_results/paper/datasets/`; each `.npz`
has a `.json` manifest beside it with the full generation record. All floats are float32.

**MUAP source: `new`** — regular 5×5 grid, 10 mm IED along (z) and across (arc length), ray-cast onto the mesh skin surface over FCU (f2_common.grid_electrodes).
`muap_grid` / `emg_grid` cover **all 100 MUs**.

## D2 — forearm FCU motor-unit pool + 5×5 HD-EMG MUAP tensor (2.4 MB)

100-MU Henneman pool of the flexor carpi ulnaris (WR forearm MRI) with golden single-channel and 5×5 HD-grid MUAPs.

* Anatomy: WR forearm segmentation (`forearm_WR_segmentation.nii.gz`, label 8 = FCU), FEM mesh `forearm_WR.msh`
  with fibre-aligned muscle anisotropy (`forearm_WR_fibers.json`), skin shell 1.5 mm.
* Fibre bed: Poisson-disk straight (morphing-disk) fibres, density 4.0 /mm², 637 fibres,
  fibre length 204 mm, dz 1 mm; mask containment 96.6 %.
  (Harmonic single-NMJ streamline bed for reference: {'n_fibres': 827, 'seed_grid_mm': 0.5, 'containment': 0.9996205958105022, 'length_mean_mm': 164.85812401991794}.)
* Pool: `sample_henneman_pool(n_mu=100, size_min=5, size_max=400, seed=0)` —
  exponential sizes 5–395 (median 66), territories grown around a random
  anchor fibre; index = size rank = recruitment order. `mu_fibre_idx` (−1-padded) indexes `bed_paths`.
* Electrodes: regular 5×5 grid, 10 mm IED along (z) and across (arc length), ray-cast onto the mesh skin surface over FCU (f2_common.grid_electrodes). Median IED along 10.1 mm (range 10.0–10.6),
  across 10.0 mm (range 9.7–10.0); centre z = 167.3 mm,
  FCU direction 75.6°. Electrode `e = row*5 + col` (rows along the arm, columns around it);
  `muap_single` is electrode 12 (centre).
* Lead fields: one FEniCSx reciprocity solve per electrode (Gaussian source σ = 5 mm), φ sampled along all
  637 fibres (FEM build 3.5 s; 0.28 s solve + 6.1 s sampling per electrode).
* MUAPs: golden spatial recipe (`field_to_muap` with `SpatialConfig(denoise=monopole(3), one_sided window,
  csd_derivative=2, upsample 2, fs 2048, w 256, t_start −10 ms, v 4 m/s)`), IZ at 0.305 ± 0.02 of the fibre;
  physical time, t = 0 at NMJ firing. Tensor build 1254 s wall (12530 s CPU).
* Activation-layer per-MU parameters (`MotoneuronPool` / `TwitchPool` defaults) are included for convenience.

| array | shape | dtype | unit |
|---|---|---|---|
| `mu_sizes` | (100,) | int32 |  |
| `mu_fibre_idx` | (100, 395) | int32 |  |
| `mu_centre_xy` | (100, 2) | float32 | mm at the muscle mid-z |
| `mu_territory_radius_mm` | (100,) | float32 |  |
| `mu_recruitment_order` | (100,) | int32 |  |
| `mu_recruitment_threshold_excitation` | (100,) | float32 |  |
| `mu_peak_rate_hz` | (100,) | float32 |  |
| `mu_min_rate_hz` | (100,) | float32 |  |
| `mu_twitch_peak` | (100,) | float32 |  |
| `mu_twitch_contraction_time_ms` | (100,) | float32 |  |
| `muap_single` | (100, 256) | float32 | µV |
| `muap_grid` | (100, 25, 256) | float32 | µV |
| `muap_grid_mu_index` | (100,) | int32 |  |
| `t_ms` | (256,) | float32 | ms (t = 0 at NMJ firing) |
| `elec_xyz` | (25, 3) | float32 | mm (segmentation voxel frame: index × voxel size) |
| `elec_grid_xyz` | (5, 5, 3) | float32 |  |
| `fs` | () | float32 | Hz |
| `bed_xy_mid` | (637, 2) | float32 |  |
| `bed_paths` | (637, 200, 3) | float32 | mm |
| `bed_arc_dz_mm` | (637,) | float32 |  |
| `bed_length_mm` | (637,) | float32 |  |
| `fcu_centroid_xy` | (2,) | float32 |  |

## D3 — interference-EMG trials (4.14 MB)

Interference EMG (single channel + 5×5 grid), spike trains and force for trapezoid contractions at six drive levels, plus one angle-modulated dynamic trial.

* Trapezoid: lead 0.2 s, rise 0.5 s, hold 2.0 s, fall 0.5 s
  (6554 samples = 3.20 s at 2048 Hz), drive levels [0.1, 0.2, 0.35, 0.5, 0.7, 1.0], plus a shared low-pass common drive
  (σ = 0.015, 2.0 Hz, seed 0). Spike trains: `MotoneuronPool.spike_trains(seed=0)`.
* EMG = D2 MUAPs convolved with the spike trains (`compound_emg` / `compound_emg_multi`); force from `TwitchPool` in %MVC.
  `emg_grid` covers all 100 MUs.
* Dynamic trial: 6.0 s, 2 flexion–extension cycles (`angle_track`), drive from the angle
  (+ common drive σ 0.012), MUAP amplitude gain 0.8 and time-warp gain 0.22
  (`dynamic_compound_emg`), single channel.
* Plateau summary (window 1.0–2.4 s): active MUs [53, 68, 81, 90, 98, 100] (on the grid [53, 68, 81, 90, 98, 100]);
  RMS (µV) [13.537, 27.185, 46.759, 73.459, 101.022, 132.762]; force (%MVC) [7.4, 15.6, 29.8, 45.8, 68.6, 99.6]; dynamic RMS flexed/extended 8.66.

| array | shape | dtype | unit |
|---|---|---|---|
| `levels` | (6,) | float32 |  |
| `drive` | (6, 6554) | float32 | fraction of maximal drive (0–1) |
| `emg_single` | (6, 6554) | float32 | µV |
| `emg_grid` | (6, 25, 6554) | float32 | µV (electrode e = row*5 + col of D2 elec_xyz) |
| `emg_grid_mu_index` | (100,) | int32 |  |
| `force` | (6, 6554) | float32 | %MVC |
| `spikes` | (6, 100, 117) | int32 | sample index at fs (−1 = padding) |
| `spike_counts` | (6, 100) | int32 |  |
| `t_s` | (6554,) | float32 |  |
| `dyn_angle` | (12288,) | float32 | normalised joint angle (0 extended, 1 flexed) |
| `dyn_drive` | (12288,) | float32 |  |
| `dyn_emg` | (12288,) | float32 |  |
| `dyn_amp` | (12288,) | float32 |  |
| `dyn_warp` | (12288,) | float32 |  |
| `dyn_spikes` | (100, 150) | int32 |  |
| `dyn_spike_counts` | (100,) | int32 |  |
| `dyn_t_s` | (12288,) | float32 |  |
| `fs` | () | float32 |  |
