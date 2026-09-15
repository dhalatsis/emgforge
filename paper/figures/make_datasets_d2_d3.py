"""Datasets D2 (forearm FCU MU pool + HD-grid MUAP tensor) and D3 (interference-EMG trials).

D2 ``_results/paper/datasets/forearm_fcu_mu_pool.npz`` (+ ``.json`` manifest)
D3 ``_results/paper/datasets/interference_emg_trials.npz`` (+ ``.json`` manifest)
plus the human-readable ``paper/datasets/MANIFEST_f2.md``.

MUAP source (env ``F2_TENSOR``): ``new`` = the regular 10 mm 5×5 grid tensor for all 100 MUs
(f2_common caches); ``legacy42`` = the pre-existing 42-unit tensor on the old grid plus the
100 single-channel MUAPs of ``mu_pool.npz``; ``auto`` (default) = new if complete, else legacy.
The manifests state which one was used.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_datasets_d2_d3.py
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "paper/figures")
import numpy as np

import f2_common as C

T0 = time.time()
KN = {}
C.DATASETS.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = C.ROOT / "paper/datasets"; MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
f32 = lambda a: np.asarray(a, dtype=np.float32)

# =========================================================================== D2
fm = C.load_fibre_model(); bed = C.poisson_bed(fm); pool = C.henneman_pool(bed)
D = C.load_muaps()
W, t_ms, sizes, n_grid = D["W"], D["t_ms"], D["sizes"], D["n_grid_mus"]
assert np.array_equal(sizes, [m.size for m in pool]), "pool/tensor size mismatch"
elec = D["elec_xyz"]
arc_dz, L_fib = C.bed_arc_geometry(bed)
cont_p, _ = C.mask_containment(bed.paths, fm)
try:
    hb = C.harmonic_bed(fm)
    hstats = dict(n_fibres=int(len(hb.r_norms)), seed_grid_mm=C.HARM_GRID_MM,
                  containment=float(C.mask_containment(hb.paths, fm)[0]),
                  length_mean_mm=float((hb.half1_mm + hb.half2_mm).mean()))
except Exception as exc:                                          # pragma: no cover
    hstats = dict(error=str(exc))
print(f"MUAP source: {D['source']} ({n_grid} MUs on the grid) — {D['grid_note']}")

fibre_idx = np.full((C.N_MU, int(sizes.max())), -1, dtype=np.int32)
for k, m in enumerate(pool):
    fibre_idx[k, :m.size] = m.fiber_idxs
mn, tw = C.activation_models()
D2 = dict(
    mu_sizes=sizes.astype(np.int32),
    mu_fibre_idx=fibre_idx,                                        # (100, max_size) −1-padded indices into bed_paths
    mu_centre_xy=f32([m.centre_xy for m in pool]),
    mu_territory_radius_mm=f32([m.territory_radius_mm for m in pool]),
    mu_recruitment_order=np.arange(C.N_MU, dtype=np.int32),        # pool index = size rank = recruitment order
    mu_recruitment_threshold_excitation=f32(mn.rte),               # activation-layer threshold (fraction of max drive)
    mu_peak_rate_hz=f32(mn.peak_fr), mu_min_rate_hz=f32(mn.min_fr),
    mu_twitch_peak=f32(tw.P), mu_twitch_contraction_time_ms=f32(tw.T / C.FS * 1000),
    muap_single=f32(D["muap_single"] * 1e6),                       # (100, 256) µV, centre electrode
    muap_grid=f32(W * 1e6),                                        # (n_grid, 25, 256) µV
    muap_grid_mu_index=np.arange(n_grid, dtype=np.int32),          # which pool MUs muap_grid covers
    t_ms=f32(t_ms),
    elec_xyz=f32(elec.reshape(-1, 3)),                             # (25, 3) mm, e = row*5 + col
    elec_grid_xyz=f32(elec),                                       # (5, 5, 3) mm
    fs=np.float32(C.FS),
    bed_xy_mid=f32(bed.xy_mid), bed_paths=f32(bed.paths),          # (637, 2), (637, 200, 3) mm
    bed_arc_dz_mm=f32(arc_dz), bed_length_mm=f32(L_fib),
    fcu_centroid_xy=f32(bed.centroid_xy),
)
d2_path = C.DATASETS / "forearm_fcu_mu_pool.npz"
np.savez_compressed(d2_path, **D2)
grid_cov = (f"all {n_grid} MUs" if n_grid == C.N_MU else
            f"ONLY the {n_grid} smallest of the 100 MUs (muap_grid_mu_index); muap_single covers all 100")
d2_manifest = dict(
    name="forearm_fcu_mu_pool", file=d2_path.name, size_MB=round(d2_path.stat().st_size / 1e6, 2),
    description="100-MU Henneman pool of the flexor carpi ulnaris (WR forearm MRI) with golden single-channel and 5×5 HD-grid MUAPs.",
    muap_source=D["source"], grid=D["grid_note"], muap_grid_coverage=grid_cov,
    arrays={k: dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype)) for k, v in D2.items()},
    units=dict(muap_single="µV", muap_grid="µV", t_ms="ms (t = 0 at NMJ firing)", elec_xyz="mm (segmentation voxel frame: index × voxel size)",
               bed_paths="mm", mu_centre_xy="mm at the muscle mid-z", fs="Hz"),
    generation=dict(
        segmentation=C.SEG.name, mesh=C.MESH.name, fibre_config=C.FIBER_CFG.name, muscle_label=C.FCU, muscle="FlexorCarpiUlnaris",
        fibre_bed=dict(method="poisson", density_per_mm2=C.DENSITY, n_fibres=int(len(bed.r_norms)), dz_mm=float(bed.z_vals[1] - bed.z_vals[0]),
                       half_mm=float(bed.half_mm), area_mm2=float(bed.cross_section_area_mm2), mask_containment=float(cont_p),
                       harmonic_single_nmj_bed_for_reference=hstats),
        pool=dict(n_mu=C.N_MU, size_min=5, size_max=int(min(400, len(bed.r_norms))), distribution="exponential", seed=C.SEED,
                  sizes_min_median_max=[int(sizes.min()), float(np.median(sizes)), int(sizes.max())]),
        innervation=dict(IZ_FRAC=C.IZ_FRAC, IZ_jitter_frac=C.IZ_JITTER, per_MU_rng_seed="mu.idx"),
        electrodes=dict(M=C.M, ied_along_mm_median=D["ied_along"], ied_across_mm_median=D["ied_across"],
                        ied_along_mm_range=D["ied_along_range"], ied_across_mm_range=D["ied_across_range"],
                        centre_z_mm=D["zc_mm"], fcu_angle_deg=D["fcu_ang"], placement=D["grid_note"]),
        fem=dict(solver="MRIFEMModel (FEniCSx), sigma_mode=centerline, skin_shell_mm=1.5, Gaussian source sigma 5 mm, one reciprocity solve per electrode",
                 fem_build_s=D["fem_build_s"], solve_s_per_electrode=D["solve_s"], phi_sampling_s_per_electrode=D["sample_s"]),
        synthesis=dict(engine="spatial (golden recipe): field_to_muap per MU", SpatialConfig=C.spcfg_dict(),
                       tensor_wall_s=D["tensor_wall_s"], tensor_cpu_s=D["tensor_cpu_s"]),
        activation_defaults="MotoneuronPool(n_mu=100, fs=2048) / TwitchPool defaults (NeuroMotion/Fuglevand parametrisation)",
    ),
)
with open(d2_path.with_suffix(".json"), "w") as fh:
    json.dump(d2_manifest, fh, indent=2)
print(f"D2 → {d2_path} ({d2_manifest['size_MB']} MB); muap_grid covers {grid_cov}")

# =========================================================================== D3
t0 = time.time()
W1 = D["muap_single"]
trials = [C.trapezoid_trial(lv, mn, tw, W1, Wg=W) for lv in C.LEVELS]
T = len(trials[0]["drive"])
sp_pad = [C.pad_spikes(tr["spikes"]) for tr in trials]
max_sp = max(p[0].shape[1] for p in sp_pad)
spikes = np.full((len(C.LEVELS), C.N_MU, max_sp), -1, dtype=np.int32)
counts = np.zeros((len(C.LEVELS), C.N_MU), dtype=np.int32)
for i, (p, c) in enumerate(sp_pad):
    spikes[i, :, :p.shape[1]] = p; counts[i] = c
dyn = C.dynamic_trial(mn, W1)
dsp, dcnt = C.pad_spikes(dyn["spikes"])
pl = C.plateau_slice(T)
D3 = dict(
    levels=f32(C.LEVELS),
    drive=f32([tr["drive"] for tr in trials]),                      # (6, T)
    emg_single=f32([tr["emg_single"] * 1e6 for tr in trials]),      # (6, T) µV, centre electrode, all 100 MUs
    emg_grid=f32([tr["emg_grid"] * 1e6 for tr in trials]),          # (6, 25, T) µV, the n_grid MUs of D2 muap_grid
    emg_grid_mu_index=np.arange(n_grid, dtype=np.int32),
    force=f32([tr["force"] * 100 for tr in trials]),                # (6, T) %MVC
    spikes=spikes, spike_counts=counts,                             # (6, 100, max) sample indices, −1 padded
    t_s=f32(np.arange(T) / C.FS),
    dyn_angle=f32(dyn["angle"]), dyn_drive=f32(dyn["drive"]), dyn_emg=f32(dyn["emg"] * 1e6),
    dyn_amp=f32(dyn["amp"]), dyn_warp=f32(dyn["warp"]),
    dyn_spikes=dsp, dyn_spike_counts=dcnt,
    dyn_t_s=f32(np.arange(len(dyn["angle"])) / C.FS),
    fs=np.float32(C.FS),
)
d3_path = C.DATASETS / "interference_emg_trials.npz"
np.savez_compressed(d3_path, **D3)
rms = [float(np.sqrt((tr["emg_single"][pl] ** 2).mean()) * 1e6) for tr in trials]
rms_grid = [float(np.sqrt((tr["emg_grid"][:, pl] ** 2).mean()) * 1e6) for tr in trials]
force_pl = [float(tr["force"][pl].mean() * 100) for tr in trials]
n_active = [int((c > 0).sum()) for c in counts]
n_active_grid = [int((c[:n_grid] > 0).sum()) for c in counts]
ang = dyn["angle"]; rf = float(np.sqrt((dyn["emg"][ang > 0.75] ** 2).mean())); re = float(np.sqrt((dyn["emg"][ang < 0.25] ** 2).mean()))
grid_cov3 = (f"all 100 MUs" if n_grid == C.N_MU else
             f"ONLY the {n_grid} smallest MUs (emg_grid_mu_index) — the larger units are absent from emg_grid; emg_single/force/spikes use all 100")
d3_manifest = dict(
    name="interference_emg_trials", file=d3_path.name, size_MB=round(d3_path.stat().st_size / 1e6, 2),
    description="Interference EMG (single channel + 5×5 grid), spike trains and force for trapezoid contractions at six drive levels, plus one angle-modulated dynamic trial.",
    muap_source=D["source"], grid=D["grid_note"], emg_grid_coverage=grid_cov3,
    arrays={k: dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype)) for k, v in D3.items()},
    units=dict(drive="fraction of maximal drive (0–1)", emg_single="µV", emg_grid="µV (electrode e = row*5 + col of D2 elec_xyz)", force="%MVC",
               spikes="sample index at fs (−1 = padding)", dyn_angle="normalised joint angle (0 extended, 1 flexed)"),
    generation=dict(
        muaps="D2 muap_grid / muap_single (same pool, same electrode geometry)",
        trapezoid=dict(**C.TRAP, levels=C.LEVELS, common_drive=C.COMMON_DRIVE, seed=C.SEED, n_samples=int(T), duration_s=T / C.FS),
        pool="MotoneuronPool(n_mu=100, fs=2048) defaults: rr=50, rm=0.75, pfr 40→30 Hz, mfr 10→5 Hz, ISI CV 1/6",
        force="TwitchPool defaults (rp=100, tmax 90 ms, tr=3), normalised to the mean force at full drive (%MVC)",
        dynamic=dict(**C.DYN, spike_seed=C.SEED, n_samples=int(len(ang))),
        plateau_window_s=[pl.start / C.FS, pl.stop / C.FS],
    ),
    summary=dict(n_active_per_level=dict(zip(map(str, C.LEVELS), n_active)),
                 n_active_on_grid_per_level=dict(zip(map(str, C.LEVELS), n_active_grid)),
                 plateau_rms_single_uV=dict(zip(map(str, C.LEVELS), rms)),
                 plateau_rms_grid_mean_uV=dict(zip(map(str, C.LEVELS), rms_grid)),
                 plateau_force_pct_mvc=dict(zip(map(str, C.LEVELS), force_pl)),
                 max_spikes_per_mu=int(max_sp),
                 dynamic_rms_flexed_over_extended=rf / max(re, 1e-12), dynamic_n_active=int((dcnt > 0).sum())),
)
with open(d3_path.with_suffix(".json"), "w") as fh:
    json.dump(d3_manifest, fh, indent=2)
print(f"D3 → {d3_path} ({d3_manifest['size_MB']} MB) in {time.time()-t0:.0f}s; emg_grid covers {grid_cov3}")
print("  active MUs", n_active, "| plateau RMS µV", np.round(rms, 3).tolist(), "| force %MVC", np.round(force_pl, 1).tolist())

# =========================================================================== manifest (markdown)
def arr_table(arrays, units):
    rows = ["| array | shape | dtype | unit |", "|---|---|---|---|"]
    for k, v in arrays.items():
        rows.append(f"| `{k}` | {tuple(v['shape'])} | {v['dtype']} | {units.get(k, '')} |")
    return "\n".join(rows)


fmt = lambda v, s="{:.2f}": ("n/a" if v is None else s.format(v))
md = f"""# F2 datasets — D2 `forearm_fcu_mu_pool.npz`, D3 `interference_emg_trials.npz`

Written by `paper/figures/make_datasets_d2_d3.py` to `_results/paper/datasets/`; each `.npz`
has a `.json` manifest beside it with the full generation record. All floats are float32.

**MUAP source: `{D['source']}`** — {D['grid_note']}.
`muap_grid` / `emg_grid` cover **{grid_cov}**.

## D2 — forearm FCU motor-unit pool + 5×5 HD-EMG MUAP tensor ({d2_manifest['size_MB']} MB)

{d2_manifest['description']}

* Anatomy: WR forearm segmentation (`{C.SEG.name}`, label {C.FCU} = FCU), FEM mesh `{C.MESH.name}`
  with fibre-aligned muscle anisotropy (`{C.FIBER_CFG.name}`), skin shell 1.5 mm.
* Fibre bed: Poisson-disk straight (morphing-disk) fibres, density {C.DENSITY} /mm², {len(bed.r_norms)} fibres,
  fibre length {L_fib.mean():.0f} mm, dz 1 mm; mask containment {cont_p*100:.1f} %.
  (Harmonic single-NMJ streamline bed for reference: {hstats}.)
* Pool: `sample_henneman_pool(n_mu=100, size_min=5, size_max={int(min(400, len(bed.r_norms)))}, seed=0)` —
  exponential sizes {int(sizes.min())}–{int(sizes.max())} (median {np.median(sizes):.0f}), territories grown around a random
  anchor fibre; index = size rank = recruitment order. `mu_fibre_idx` (−1-padded) indexes `bed_paths`.
* Electrodes: {D['grid_note']}. Median IED along {D['ied_along']:.1f} mm (range {D['ied_along_range'][0]:.1f}–{D['ied_along_range'][1]:.1f}),
  across {D['ied_across']:.1f} mm (range {D['ied_across_range'][0]:.1f}–{D['ied_across_range'][1]:.1f}); centre z = {D['zc_mm']:.1f} mm,
  FCU direction {D['fcu_ang']:.1f}°. Electrode `e = row*5 + col` (rows along the arm, columns around it);
  `muap_single` is electrode 12 (centre).
* Lead fields: one FEniCSx reciprocity solve per electrode (Gaussian source σ = 5 mm), φ sampled along all
  {len(bed.r_norms)} fibres (FEM build {fmt(D['fem_build_s'], '{:.1f}')} s; {fmt(D['solve_s'])} s solve + {fmt(D['sample_s'], '{:.1f}')} s sampling per electrode).
* MUAPs: golden spatial recipe (`field_to_muap` with `SpatialConfig(denoise=monopole(3), one_sided window,
  csd_derivative=2, upsample 2, fs 2048, w 256, t_start −10 ms, v 4 m/s)`), IZ at 0.305 ± 0.02 of the fibre;
  physical time, t = 0 at NMJ firing. Tensor build {fmt(D['tensor_wall_s'], '{:.0f}')} s wall ({fmt(D['tensor_cpu_s'], '{:.0f}')} s CPU).
* Activation-layer per-MU parameters (`MotoneuronPool` / `TwitchPool` defaults) are included for convenience.

{arr_table(d2_manifest['arrays'], d2_manifest['units'])}

## D3 — interference-EMG trials ({d3_manifest['size_MB']} MB)

{d3_manifest['description']}

* Trapezoid: lead {C.TRAP['lead_s']} s, rise {C.TRAP['rise_s']} s, hold {C.TRAP['hold_s']} s, fall {C.TRAP['fall_s']} s
  ({T} samples = {T/C.FS:.2f} s at {C.FS:.0f} Hz), drive levels {C.LEVELS}, plus a shared low-pass common drive
  (σ = {C.COMMON_DRIVE['sigma']}, {C.COMMON_DRIVE['cutoff_hz']} Hz, seed 0). Spike trains: `MotoneuronPool.spike_trains(seed=0)`.
* EMG = D2 MUAPs convolved with the spike trains (`compound_emg` / `compound_emg_multi`); force from `TwitchPool` in %MVC.
  `emg_grid` covers {grid_cov3}.
* Dynamic trial: {C.DYN['duration_s']} s, {C.DYN['cycles']} flexion–extension cycles (`angle_track`), drive from the angle
  (+ common drive σ {C.DYN['drive_sigma']}), MUAP amplitude gain {C.DYN['amp_gain']} and time-warp gain {C.DYN['warp_gain']}
  (`dynamic_compound_emg`), single channel.
* Plateau summary (window {pl.start/C.FS:.1f}–{pl.stop/C.FS:.1f} s): active MUs {n_active} (on the grid {n_active_grid});
  RMS (µV) {np.round(rms, 3).tolist()}; force (%MVC) {np.round(force_pl, 1).tolist()}; dynamic RMS flexed/extended {rf/re:.2f}.

{arr_table(d3_manifest['arrays'], d3_manifest['units'])}
"""
with open(MANIFEST_DIR / "MANIFEST_f2.md", "w") as fh:
    fh.write(md)
print("manifest →", MANIFEST_DIR / "MANIFEST_f2.md")

KN.update(tensor_source=D["source"], grid_note=D["grid_note"], n_grid_mus=n_grid,
          D2=dict(file=str(d2_path), size_MB=d2_manifest["size_MB"], muap_grid_coverage=grid_cov,
                  shapes={k: v["shape"] for k, v in d2_manifest["arrays"].items()}),
          D3=dict(file=str(d3_path), size_MB=d3_manifest["size_MB"], emg_grid_coverage=grid_cov3,
                  shapes={k: v["shape"] for k, v in d3_manifest["arrays"].items()}, **d3_manifest["summary"]),
          runtimes=dict(fem_build_s=D["fem_build_s"], solve_s_per_electrode=D["solve_s"],
                        phi_sampling_s_per_electrode=D["sample_s"], tensor_wall_s=D["tensor_wall_s"],
                        tensor_cpu_s=D["tensor_cpu_s"], tensor_s_per_mu_min_max=D["tensor_s_per_mu"],
                        datasets_total_s=time.time() - T0))
C.update_key_numbers("datasets_d2_d3", KN)
