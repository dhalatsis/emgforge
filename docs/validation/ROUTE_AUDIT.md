# Route audit — which pipeline to use, measured

Generated 2026-09-17 04:37:02 by `python scripts/sanity/route_audit.py` in 5 s. Live forks run on the analytical cylinder (fibre 10 mm below the skin, NMJ 20 mm proximal of the electrode, tendons at -60/+60 mm from the electrode, v = 4 m/s, fs = 4096 Hz, dz = 0.977 mm) and on the closed-form line-source oracle (electrode 10 mm from the fibre). Recorded forks quote `_results/validation/*.json` and the paper key numbers (`git show paper/arxiv:paper/figures/…`). The recipe under test is `emgforge.synthesis.production_config()`.

## Summary

| # | fork | use | current default | status |
|---|---|---|---|---|
| 1 | Synthesis engine | direct line-source (spatial) engine | production_config() → SpatialConfig; field_to_muap(config=None) → spatial (direct, physical time) | PASS — both the recipe object and the no-config default must be the direct engine |
| 2 | Lead-field conditioning (φ denoise before the CSD) | monopole(3) — production_config().denoise = 'monopole', denoise_n_poles = 3 | denoise = 'monopole', denoise_n_poles = 3 | PASS |
| 3 | Tendon (fibre-end) window | one_sided (α = 0.25) — production_config().fiber_window | fiber_window = 'one_sided', tukey_alpha = 0.25 | PASS |
| 4 | Sampling (fs, dz, upsample, w) | fs 4096 / dz = v/fs on the cylinder tiers, fs 2048 / arc dz ≈ 1 mm on MRI, upsample_factor = 2, w = 256 | upsample_factor = 2, fsamp = 2048 (MRI regime; harness uses 4096), w = 256 | PASS |
| 5 | Time base | physical time: center_time = False, t_start_ms = −10 (t = 0 is the NMJ discharge) | center_time = False, t_start_ms = -10.0 | PASS |
| 6 | Amplitude constant (no 1/v prefactor) | the corrected constant (CV-independent integral) | compute_sfap_spatial has no '/ cfg.v' on the sfap line | PASS |
| 7 | Volume conductor and electrode source width | analytical cylinder for laws; FEM/MRI for anatomy; σ_s = 1 mm for NEW solves, 5 mm only to stay consistent with the released datasets | solve_grid_leadfields(source_sigma = 5.0); cyl_fem.SOURCE_SIGMA = 5.0 | PASS — deliberate: 5 mm = released-dataset compatibility; pass source_sigma=1.0 for new solves |
| 8 | Fibre bed (straight Poisson vs harmonic streamlines) | straight Poisson bed (run_pipeline.py / mri.pipeline.poisson_bed) | run_pipeline.py stage 4 = poisson_bed (straight, density 4/mm²) | PASS |
| 9 | Electrode grid | the regular ray-cast grid built by emgforge.mri.pipeline.grid_electrodes (run_pipeline.py) | run_pipeline.py → grid_electrodes (regular); Simulator.from_mri → legacy cache (only as a warned fallback behind run_pipeline.py outputs) | PASS — Simulator.from_mri prefers the largest run_pipeline.py output; the legacy tensor loads only with a UserWarning |
| 10 | Entry point | python scripts/run_pipeline.py (cold, reproducible) → Simulator.from_pipeline(npz) | scripts/run_pipeline.py present; Simulator.from_pipeline present; verify identical in both recorded runs | PASS |
| 11 | Activation defaults (pool, rate coding, twitch/force) | the shipped activation defaults (MotoneuronPool / TwitchPool) | defaults unchanged | PASS — open item: C2 refractory floor (FAIL in C.json, not a known-flag) |

## Recipe guard

`emgforge.synthesis.production_config()` vs the documented recipe (DIRECT_LINE_SOURCE.md §0): **no mismatch**. `field_to_muap(field, bed)` with no config → **spatial (direct, physical time)**.

## Fork 1 — Synthesis engine  *(live)*

**Use:** direct line-source (spatial) engine.  
**Alternative is legitimate when:** Fourier only as the shape oracle against the Farina MATLAB reference set (tests/synthesis/test_golden.py); its polarity is anti-phase and its time base window-centred, so never for physical-time / MRI synthesis.  
**Why:** the direct engine reproduces the closed-form line-source integral to r ≥ 0.999 with zero lag, a monopole-free source and the EOF at L/v; the Fourier engine does not.

| option | key numbers | verdict |
|---|---|---|
| direct (spatial line-source), boxcar / no denoise | r = 0.9999, lag = -0.05 ms, amp ratio = 0.998 vs the closed-form oracle; monopole-free residual 7.7e-17 (boxcar) / 7.7e-17 (one_sided); EOF onset 10.26 ms (L/v = 10) | USE — reproduces the integral |
| direct, full production recipe (monopole 3 + one_sided) | r = 0.9964, lag = -0.10 ms, amp ratio = 0.998 vs the sharp-tendon oracle (the one-sided taper softens the tendon term — fork 3); EOF onset 8.55 ms | USE (production) — same integral, tendon softened by design |
| Fourier (Farina 2001 radon/pare) | signed r = -0.684 at lag +2.55 ms, amp ratio = 0.376; vs a spatially mirrored IAP r = -0.684; EOF onset 7.57 ms; no CSD (monopole test n/a); window-centred time | shape oracle only (r = 0.997 vs the Farina MATLAB reference); not for physical time or MRI |

**Current default:** production_config() → SpatialConfig; field_to_muap(config=None) → spatial (direct, physical time) → **PASS** (both the recipe object and the no-config default must be the direct engine)  
**Thresholds:** r_min = 0.999, lag_max_ms = 0.1, amp_tol = 0.02, monopole_free_max = 1e-12, eof_tol_ms = 0.5  
**Evidence:** live: closed-form oracle harness.sfap_first_principles (Rosenfalck 1969 / Dimitrov & Dimitrova 1998); recorded: _results/validation/A.json A0.1–A0.3, A0.5; key_numbers_f1.json fig4_first_principles

> The production recipe's earlier EOF 'onset' is the one-sided taper starting 25 % of the 40 mm semi-fibre before the tendon (30 mm / v = 7.5 ms) — by design, see fork 3.

> The Fourier amplitude ratio (0.38) is low here because the closed-form φ does not decay at the FFT window edges — the reason analytical φ is excluded from the Fourier reference set (DIRECT_LINE_SOURCE.md §6); on decaying Gaussian φ the two engines agree to 0.84–1.01.

## Fork 2 — Lead-field conditioning (φ denoise before the CSD)  *(live)*

**Use:** monopole(3) — production_config().denoise = 'monopole', denoise_n_poles = 3.  
**Alternative is legitimate when:** Butterworth(0.03) when the question is an amplitude law across FEM depths (it halves the spectral content, so never for shape/spectrum); 'none' only on clean analytical φ, where monopole(3) is a near no-op anyway.  
**Why:** the 3-monopole fit is the analytic form the field has: it removes the mesh ripple from φ'' (the kernel the SFAP integrates) without rounding the peak. CAVEAT (recorded A1.2): on FEM φ the SFAP amplitude stays erratic at ±40 % across depth — not high-frequency, and 5/7 poles do not remove it; shape, timing and spectrum are faithful.

| option | key numbers | verdict |
|---|---|---|
| none | SFAP jaggedness 0.0114; r(φ'') vs the analytical φ'' 0.726; vs the in-band raw FEM SFAP: MNF ×1.22, p2p ×1.16, r 0.985; vs the analytical SFAP: MNF ×1.69, p2p ×2.38, r 0.914; on the analytical φ itself: r 1.0000, p2p ×1.000; 8 ms | diagnostic only (raw ripple reaches φ'') |
| butterworth(0.03) | SFAP jaggedness 0.0009; r(φ'') vs the analytical φ'' 0.930; vs the in-band raw FEM SFAP: MNF ×0.46, p2p ×0.23, r 0.674; vs the analytical SFAP: MNF ×0.63, p2p ×0.48, r 0.879; on the analytical φ itself: r 0.7668, p2p ×0.433; 8 ms | legitimate ONLY for relative amplitude-vs-depth laws on FEM φ (A1.2: 1.22× spread vs 1.69× direct); it destroys the absolute amplitude (×0.23 in-band) and spectrum (MNF ×0.46) and is not a no-op even on the clean analytical φ (r 0.77) |
| monopole(3) | SFAP jaggedness 0.0022; r(φ'') vs the analytical φ'' 0.913; vs the in-band raw FEM SFAP: MNF ×0.99, p2p ×0.98, r 0.996; vs the analytical SFAP: MNF ×1.37, p2p ×2.01, r 0.933; on the analytical φ itself: r 0.9999, p2p ×0.991; 35 ms | USE — removes the ripple, keeps φ'' and the spectrum |
| monopole(5) | SFAP jaggedness 0.0022; r(φ'') vs the analytical φ'' 0.915; vs the in-band raw FEM SFAP: MNF ×0.99, p2p ×0.98, r 0.998; vs the analytical SFAP: MNF ×1.37, p2p ×2.01, r 0.934; on the analytical φ itself: r 0.9999, p2p ×0.995; 166 ms | no gain over 3 poles (A1.2: 5/7 poles 1.49×/1.51× amplitude spread) |
| monopole(7) | SFAP jaggedness 0.0022; r(φ'') vs the analytical φ'' 0.911; vs the in-band raw FEM SFAP: MNF ×1.00, p2p ×0.99, r 0.997; vs the analytical SFAP: MNF ×1.39, p2p ×2.02, r 0.930; on the analytical φ itself: r 1.0000, p2p ×0.999; 199 ms | no gain over 3 poles (A1.2: 5/7 poles 1.49×/1.51× amplitude spread) |

**Current default:** denoise = 'monopole', denoise_n_poles = 3 → **PASS**  
**Thresholds:** jaggedness_max = 0.005, r_phidd_min = 0.9, mnf_retained = [0.85, 1.15], p2p_retained = [0.8, 1.2], noop_r_min = 0.99  
**Evidence:** live φ: FEM cylinder lead field, _results/validation/fem_cache/cyl_lines.npz (r = 30 mm, θ = 0, σ_s = 5 mm electrode source, mesh cl 0.3); analytical φ: harness.ana_phi(30) (Farina 2004, Ø10 mm disk electrode); r(φ FEM, φ analytical) on |z| ≤ 60 mm = 0.9948; FEM φ peak scaled ×0.121 to the analytical peak; reference for 'retained': the raw-φ SFAP low-passed at 500 Hz (A1.2's in-band content). The FEM φ at this depth is narrower than the analytical one (FWHM 45 vs 53 mm, A1.1: σ_s = 5 mm blob vs Ø10 mm disk), so every variant's MNF/p2p vs the ANALYTICAL SFAP exceeds 1 — fork 7's volume-conductor difference, not the conditioning; recorded: _results/validation/A.json A1.1b, A1.2 (amplitude ±40 %), A2.2; key_numbers_f1.json fig3_recipe_anatomy.phi_r30; A1.2 measured: p2p ratio FEM/ana, r=33→20 mm, rel. to r=33 — direct method: [1.0, 1.46, 1.35, 1.2, 0.86] (spread 1.69×); no denoise: [1.0, 1.59, 1.3, 1.72, 2.03] (2.03×); 5/7 poles: 1.49×/1.51×; Fourier route: [1.0, 1.1, 1.16, 1.19, 1.31] (1.31×). After a 500 Hz low-pass — direct method: [1.0, 1.47, 1.35, 1.2, 0.87] (1.70×), no denoise: [1.0, 1.41, 1.25, 1.2, 1.44] (1.44×).  With a σ=1 mm electrode source instead of σ=5 mm — Fourier route: [1.0, 1.06, 1.14, 1.21, 1.41] (1.41×), direct method in-band: [1.0, 0.7, 0.59, 0.64, 0.95] (1.70×).  Spatial engine with Butterworth φ-smoothing (the Fourier route's, ≈32 mm cutoff): [1.0, 1.2, 1.22, 1.18, 1.12] (1.22×)

> A1.2 recorded: TWO findings. (i) A smooth 1.3–1.4× drift survives every preprocessing and a σ=1 mm source: the FEM cylinder decays ~30 % slower with depth than the analytical one over 7→20 mm — a volume-conductor difference still to be attributed (mesh/skin-layer resolution vs the analytical k-grid). (ii) The direct recipe's SFAP amplitude on FEM φ is erratic at the ±40 % level (not high-frequency: survives a 500 Hz low-pass; differs between σ=5 and σ=1 sources) because it integrates φ'' faithfully down to ~8 mm wavelengths where FEM φ carries mesh structure; the 3-monopole fit does not remove it (5/7 poles neither) and Butterworth φ-smoothing does — at the cost of the shallow-fibre detail

## Fork 3 — Tendon (fibre-end) window  *(live)*

**Use:** one_sided (α = 0.25) — production_config().fiber_window.  
**Alternative is legitimate when:** boxcar when comparing against a sharp-tendon oracle (first principles, Farina) or to isolate the end-of-fibre potential; never tukey/hann (junction notch).  
**Why:** a symmetric window notches the junction where the travelling wave is born (window = 0 at the NMJ); one_sided keeps the generation lobe at boxcar strength, keeps the EOF/propagating ratio inside the literature range and stays within r ≥ 0.95 of the sharp-tendon oracle.

| option | key numbers | verdict |
|---|---|---|
| boxcar | r vs sharp-tendon oracle 0.9999 (lag -0.05 ms, amp 0.998); EOF/propagating 0.053 (lit. 0.02–0.2 at 7–20 mm), EOF onset 10.26 ms; window at the NMJ 1.00/1.00; generation lobe 1.000× boxcar | sharp-tendon oracle comparisons and 'see the EOF' diagnostics |
| tukey | r vs sharp-tendon oracle 0.9953 (lag -0.10 ms, amp 0.998); EOF/propagating 0.098 (lit. 0.02–0.2 at 7–20 mm), EOF onset 0.99 ms; window at the NMJ 0.00/0.00; generation lobe 0.632× boxcar | ARTEFACT — notches the junction (window 0 at the NMJ); legacy default, do not use |
| hann | r vs sharp-tendon oracle 0.9634 (lag -0.90 ms, amp 0.539); EOF/propagating 0.149 (lit. 0.02–0.2 at 7–20 mm), EOF onset 1.47 ms; window at the NMJ 0.00/0.00; generation lobe 0.093× boxcar | ARTEFACT — notches the junction; reference only |
| one_sided | r vs sharp-tendon oracle 0.9964 (lag -0.10 ms, amp 0.998); EOF/propagating 0.061 (lit. 0.02–0.2 at 7–20 mm), EOF onset 8.55 ms; window at the NMJ 1.00/1.00; generation lobe 1.000× boxcar | USE — flat through the NMJ, tendon softened |

**Current default:** fiber_window = 'one_sided', tukey_alpha = 0.25 → **PASS**  
**Thresholds:** one_sided_r_min = 0.95, boxcar_r_min = 0.999, eof_ratio_lit = [0.02, 0.2], junction_flat = 1.0, gen_lobe_min = 0.95  
**Evidence:** live: r on the closed-form φ vs harness.sfap_first_principles (boxcar oracle); EOF ratio on the analytical cylinder φ at 10 mm depth by tier B4's subtraction (near tendon moved +60 mm); recorded: _results/validation/B.json B4b (0.021–0.204 at 7–20 mm); key_numbers_f1.json fig3_recipe_anatomy.windows; DIRECT_LINE_SOURCE.md §3

> 'EOF onset' for the tapered windows is where the two fibres first differ: one_sided's 8.55 ms is the start of its 10 mm tendon ramp (30 mm / v = 7.5 ms), tukey's and hann's ≈1 ms is the junction notch itself (its width changes with the semi-fibre length) — the notch is the artefact.

## Fork 4 — Sampling (fs, dz, upsample, w)  *(live)*

**Use:** fs 4096 / dz = v/fs on the cylinder tiers, fs 2048 / arc dz ≈ 1 mm on MRI, upsample_factor = 2, w = 256.  
**Alternative is legitimate when:** upsample 4 only as a convergence reference; fs is the recording rate, not a physics knob.  
**Why:** the SFAP is discretisation-converged: fs and w leave it unchanged to < 2 %, and ×2 ≡ ×4 to r ≥ 0.999 in every regime. Measured here, ×1 is already converged once φ is monopole-fitted (jaggedness ×1/×2 ≈ 1 even at dz ≠ v·dt), so upsample 2 is kept as a cheap guard for rougher φ (the recorded 0.08→0.01 was on the pre-monopole PM FEM route); 4× buys nothing.

| option | key numbers | verdict |
|---|---|---|
| fs 4096 (dz = v/fs = 0.98 mm) vs fs 2048 | fs 2048 with dz = v/fs = 1.95 mm: r = 0.99935, p2p ratio 0.9909; fs 2048 with dz = 0.98 mm (the MRI regime, arc dz ≈ 1 mm): r = 1.00000, p2p ratio 1.0000 — all vs fs 4096 | invariant: fs 4096 for the cylinder tiers, 2048 for MRI/HD-EMG (the recording rate) |
| upsample 1 / 2 / 4 | analytical φ, dz 0.98 mm / fs 4096 (cylinder tiers; v·dt = dz): jaggedness ×1 0.0018 → ×2 0.0018 → ×4 0.0018, r(×2, ×4) 0.99986, p2p(×2)/p2p(×4) 0.9938, 4/8/15 ms; analytical φ, dz 1.02 mm / fs 2048 (MRI regime; v·dt = 1.95 mm ≠ dz): jaggedness ×1 0.0035 → ×2 0.0034 → ×4 0.0034, r(×2, ×4) 0.99986, p2p(×2)/p2p(×4) 0.9980, 3/5/10 ms; FEM φ, dz 1.02 mm / fs 2048 (MRI regime, monopole 3): jaggedness ×1 0.0044 → ×2 0.0043 → ×4 0.0043, r(×2, ×4) 0.99973, p2p(×2)/p2p(×4) 1.0023, 33/35/41 ms; FEM φ RAW (no denoise), dz 1.02 mm / fs 2048: jaggedness ×1 0.0098 → ×2 0.0136 → ×4 0.0135, r(×2, ×4) 0.99920, p2p(×2)/p2p(×4) 0.9726, 3/5/10 ms; analytical φ, dz 1.95 mm / fs 2048 (v·dt = dz): jaggedness ×1 0.0034 → ×2 0.0035 → ×4 0.0035, r(×2, ×4) 0.99938, p2p(×2)/p2p(×4) 0.9810, 4/7/13 ms | KEEP 2 — converged (≡ ×4 to r ≥ 0.999) in every regime; jaggedness ×1/×2 = 1.00, 1.01, 1.01, 0.73, 0.98: with the monopole-fitted φ, ×1 is already converged, so 2 is a cheap guard (+10–50 % per SFAP), not load-bearing; the recorded 0.08→0.01 came from the pre-monopole PM FEM route. On RAW FEM φ upsampling does not remove the ripple (it propagates it into φ''): the monopole fit does |
| w 256 vs 512 | first 256 samples identical: r = 1.000000, p2p ratio 1.000000 | w only sets the window length; 256 (62.5 ms at 4096, 125 ms at 2048) holds the far EOF |

**Current default:** upsample_factor = 2, fsamp = 2048 (MRI regime; harness uses 4096), w = 256 → **PASS**  
**Thresholds:** r_min = 0.995, amp_tol = 0.02, jaggedness_max_x2 = 0.01, r_x2_vs_x4_min = 0.999  
**Evidence:** live on the analytical cylinder φ at 10 mm depth; recorded: _results/validation/A.json A0.4a; engines/spatial.py SpatialConfig.upsample_factor

## Fork 5 — Time base  *(live)*

**Use:** physical time: center_time = False, t_start_ms = −10 (t = 0 is the NMJ discharge).  
**Alternative is legitimate when:** center_time = True only to overlay a Fourier (window-centred) trace.  
**Why:** in physical time the EOF lands at L/v and the propagating lobe walks at Δz/v — the two timings HD-EMG decomposition and CV estimation rely on; a centred axis breaks both.

| option | key numbers | verdict |
|---|---|---|
| center_time = False | EOF onset 10.26 ms (L/v = 10); propagating lobe at Δz = 20/40 mm: 6.11/11.00 ms (Δz/v = 5/10; residual +1.11/+1.00 ms), CV from the lobes 4.10 m/s; window starts at -10.00 ms | USE — physical time, t = 0 at the NMJ discharge |
| center_time = True | EOF onset -20.99 ms (L/v = 10); propagating lobe at Δz = 20/40 mm: -25.14/-20.25 ms (Δz/v = 5/10; residual -30.14/-30.25 ms), CV from the lobes 4.10 m/s; window starts at -41.25 ms | only to overlay a window-centred Fourier output; every timing is shifted by −w/(2·fs) |

**Current default:** center_time = False, t_start_ms = -10.0 → **PASS**  
**Thresholds:** eof_tol_ms = 0.5, lobe_tol_ms = 1.5, cv_tol = 0.1  
**Evidence:** live on the analytical cylinder φ at 10 mm depth (boxcar so the EOF is sharp); recorded: _results/validation/A.json A0.4b, A2.3; key_numbers_f1.json fig5_physical_time

## Fork 6 — Amplitude constant (no 1/v prefactor)  *(live)*

**Use:** the corrected constant (CV-independent integral).  
**Alternative is legitimate when:** none; if a CV-dependent amplitude is ever wanted it must come from stretching the spatial IAP.  
**Why:** the engine/oracle ratio is flat at ≈0.994 across v = 2–5; the old /v made it exactly 1/v.

| option | key numbers | verdict |
|---|---|---|
| current constant: sfap = (CSD @ φ)·dz·polarity | engine/oracle amplitude ratio at v = 2/3/4/5 m/s: 0.9976, 0.9963, 0.9977, 0.9977 (mean 0.9973, spread 0.13 %) | USE — flat in v, equal to the closed-form constant |
| the 1/v trap: sfap = (CSD @ φ)·dz·polarity / v (before 2026-09-15) | would give ratio/v = 0.499, 0.332, 0.249, 0.200 — exactly 1/v (recorded 0.497 / 0.332 / 0.249 / 0.199), i.e. every MUAP 4× too small at v = 4 | NEVER — the IAP is defined in space, so the line-source integral is CV-independent; Nandedkar–Stålberg's amplitude ∝ 1/CV is for an IAP fixed in time and would have to come from stretching the IAP, not a prefactor |

**Current default:** compute_sfap_spatial has no '/ cfg.v' on the sfap line → **PASS**  
**Thresholds:** spread_max = 0.02, mean_tol = 0.05  
**Evidence:** live vs harness.sfap_first_principles at four CVs; recorded: A.json A0.2; key_numbers_f1.json fig4_first_principles.c_*

## Fork 7 — Volume conductor and electrode source width  *(recorded)*

**Use:** analytical cylinder for laws; FEM/MRI for anatomy; σ_s = 1 mm for NEW solves, 5 mm only to stay consistent with the released datasets.  
**Alternative is legitimate when:** σ_s = 5 mm when a result must be comparable to D1–D3 / key_numbers_pipeline.  
**Why:** FEM reproduces the analytical φ to r ≥ 0.995 but with a 30 % slower depth decay that no preprocessing or source width removes (open); the σ_s = 1 mm source removes the transverse over-smoothing of the 5 mm blob (25 vs 40 mm FWHM).

| option | key numbers | verdict |
|---|---|---|
| analytical 4-layer cylinder (Farina 2004) | the reference for every law: depth power-law exponent n = 2.940 (mono) / 3.477 (SD), R² > 0.99 (B5); transverse FWHM at r = 30: 39.7 mm; anisotropy elongation 1.48 (B12) | USE for laws, thresholds and oracles (closed form, no mesh) |
| FEM cylinder, σ_s = 5 mm Gaussian electrode source | φ(z) vs analytical r = 0.995–1.000 at five depths, FWHM within 20 % (A1.1); φ'' r 0.92–0.96 at the best-matched disk (A1.1b); FEM depth exponent 3.169 vs 2.940 analytical → decays ~30 % slower over 7–20 mm (A1.2 (i)); transverse FWHM 25.4 mm vs 39.7 analytical; max \|profile − analytical\| = 0.171 | USE for anatomy (MRI); keep σ_s = 5 mm ONLY for consistency with the released datasets (D2/D3, key_numbers_pipeline args.source_sigma = 5.0) |
| FEM cylinder, σ_s = 1 mm source | transverse peak profile tracks the analytical one: max \|profile − analytical\| = 0.058 (vs 0.171 at 5 mm); the 30 % slower depth decay SURVIVES σ_s = 1 mm (A1.2: Fourier-route spread 1.41× vs 1.31×) — it is a volume-conductor difference, not the source | RECOMMENDED for new solves (the 5 mm blob reaches through 2 mm skin + 3 mm fat into the muscle) |

**Current default:** solve_grid_leadfields(source_sigma = 5.0); cyl_fem.SOURCE_SIGMA = 5.0 → **PASS** (deliberate: 5 mm = released-dataset compatibility; pass source_sigma=1.0 for new solves)  
**Evidence:** `git show paper/arxiv:paper/figures/key_numbers_f1.json`; _results/validation/A.json A1.1, A1.1b, A1.2, A1.3; B.json B5, B12; key_numbers_f1.json fig2_cylinder_leadfield.transverse_r30 (σ_s 5 vs 1 mm profiles), fig9_validation_features.depth_power_law

> A1.1: r = [0.9962, 0.9948, 0.9969, 0.9984, 0.9998]; FWHM_z ana/FEM (mm) = ['41.0/35.1', '53.1/45.3', '63.1/57.7', '68.9/65.8', '81.2/82.1'] at r = 33/30/27/25/20 mm

> A1.3: r = [0.9978, 0.9997, 0.9985, 0.997]

> B5: exponent n (p2p ∝ d^−n), R²: pipeline mono 2.94 (0.998), SD 3.48 (0.998); Farina generator mono 2.04 (0.986); FEM pipeline mono 3.17 (0.987) (depth-below-skin 7–25 mm; the fibre at 7 mm sits 2 mm inside the muscle)

> B12: FWHM_z(aniso)/FWHM_z(iso): analytical cylinder 1.48, infinite medium 2.236 (theory √5 = 2.236); FEM along/across FWHM at r=30: 2.29 (layered, bounded → not exactly √5)

## Fork 8 — Fibre bed (straight Poisson vs harmonic streamlines)  *(recorded)*

**Use:** straight Poisson bed (run_pipeline.py / mri.pipeline.poisson_bed).  
**Alternative is legitimate when:** harmonic streamlines for anatomy, with the ~16 % truncated streamlines pruned or re-innervated (the study's full-span variant); series-fibering is experimental.  
**Why:** on full-span units the two beds give the same MUAPs (p2p ratio 1.01, r 0.98, CV within 3 %); the harmonic bed's only large differences come from streamlines that never reach the IZ.

| option | key numbers | verdict |
|---|---|---|
| straight Poisson bed (4 fibres/mm², 637 fibres, 204 mm) | = the released tensor bit-for-bit (max \|Δ\| = 0.000 V, identical = True); containment 0.966; CV 4.30 m/s (set 4.0); onset fraction 0.180; EOF fraction 0.237; duration 18.1 ms | USE (default) — reproducible, contained, the released numbers |
| harmonic single-NMJ bed, full-span units (18/24) | p2p harmonic/straight median 1.006 (q25–q75 0.958–1.095); r(zero lag) median 0.985 (min 0.880); CV 4.28 m/s (ratio 0.973); containment 0.999; duration 14.2 ms | legitimate for anatomy (curved, contained fibres) — same MUAPs within ~5 % |
| harmonic bed, units with truncated streamlines (6/24) | 16.3 % of streamlines (135) never reach the IZ plane → NMJ at the fibre end: p2p ratio median 2.607 (max 18.9), r median 0.160, onset fraction 5.8 (straight 0.18), CV unreadable (0 of 6 with R² > 0.9) | ARTEFACT — prune or re-innervate: the full-span variant restores r median 0.984, onset 0.156 |
| series-fibered sampling | no recorded numbers on this branch (C-04 fibre-length work) | EXPERIMENTAL |

**Current default:** run_pipeline.py stage 4 = poisson_bed (straight, density 4/mm²) → **PASS**  
**Evidence:** `git show paper/arxiv:paper/figures/key_numbers_study_fibres.json`; `git show paper/arxiv:paper/figures/key_numbers_f2.json` fig6_mri_fibres.harmonic_nmj; scripts/paper (study_fibres) on branch paper/arxiv

## Fork 9 — Electrode grid  *(live+recorded)*

**Use:** the regular ray-cast grid built by emgforge.mri.pipeline.grid_electrodes (run_pipeline.py).  
**Alternative is legitimate when:** none — the legacy cache is kept only so old scripts still load; set EMGFORGE_MUAP_BANK to D2 for the chain checks.  
**Why:** a grid snapped to mesh vertices had duplicated columns and irregular IED and sat off the muscle, which is where the physiologically wrong 0.6 µV / 33 ms MUAPs came from; the regular grid gives 10.1 × 10.0 mm spacing and 24 µV / 18 ms MUAPs (S: 'MUAP physiological scale').

| option | key numbers | verdict |
|---|---|---|
| legacy vertex-snapped grid (_results/mu_pool/electrode_grid, Simulator.from_mri) | 23 unique positions of 25 (2 duplicated); IED along 25.4 mm (range 20.7–30.9), across 5.2 mm (range 0.0–20.4); grid centre 7.9 mm from the FCU centroid (xy); MUAPs: single-channel bank median 0.57 µV / 33.2 ms; grid tensor (42 MUs) median 2.22 µV | DEPRECATED — irregular spacing, duplicated electrodes, off the muscle; the source of the 0.6 µV / 33 ms scale |
| regular ray-cast grid (mri.pipeline.grid_electrodes / run_pipeline.py / f2_common) | 5×5, IED along 10.15 mm (range 10.03–10.57), across 9.96 mm (range 9.67–10.00); centre electrode 6.5 mm from the FCU centroid (xy, on the skin above it); MUAP p2p median 23.6 µV (range 0.67–159.6), duration median 18.1 ms; 100 MUs | USE |

**Current default:** run_pipeline.py → grid_electrodes (regular); Simulator.from_mri → legacy cache (only as a warned fallback behind run_pipeline.py outputs) → **PASS** (Simulator.from_mri prefers the largest run_pipeline.py output; the legacy tensor loads only with a UserWarning)  
**Thresholds:** ied_tol_mm = 1.0, duplicates_allowed = 0  
**Evidence:** live: _results/mu_pool/electrode_grid/_phigrid_M5_dt15_z0.30-0.70_N637.npz (elec_xyz), muap_tensor_L8_M5.npz, _results/mu_pool/spatial/mu_pool.npz; scripts/validation/muap_bank.py docstring; `git show paper/arxiv:paper/figures/key_numbers_f2.json` fig6_mri_fibres.grid, fig7_mu_pool_muaps; src/emgforge/simulator.py Simulator.from_mri

## Fork 10 — Entry point  *(recorded)*

**Use:** python scripts/run_pipeline.py (cold, reproducible) → Simulator.from_pipeline(npz).  
**Alternative is legitimate when:** from_mri only to load the released legacy tensor; f2_common only to regenerate the paper figures.  
**Why:** the runner's verify step shows the factored conditioning is bit-identical to the full recipe, so the one-command cold run IS the recipe, with timings.

| option | key numbers | verdict |
|---|---|---|
| scripts/run_pipeline.py (cold) + Simulator.from_pipeline | 20 MUs: 415 s wall with 2 workers (mesh 39, lead fields 12, conditioning 221, MUAPs 128, activation 0.4 s; 673 CPU-s), verify identical = True (max \|Δ\| 0.000 V); 100 MUs: 993 s wall with 2 workers (mesh 42, lead fields 18, conditioning 253, MUAPs 659, activation 1.8 s; 1796 CPU-s), verify identical = True (max \|Δ\| 0.000 V) | USE — cold, reproducible, every stage timed and the factored conditioning verified bit-identical to the full recipe |
| paper f2_common tensor build (per-unit refits, 100 MUs) | tensor wall 1254 s, 12530 CPU-s (no conditioning factoring: each shared fibre refitted per unit); same recipe, same grid → the released D2/D3 numbers | legitimate for reproducing the paper figures only |
| Simulator.from_mri (cached legacy tensor) | instant load of _results/mu_pool/electrode_grid/muap_tensor_L8_M5.npz — 42 MUs on the legacy vertex-snapped grid (fork 9) | only for the released legacy tensor; deprecated for new work |

**Current default:** scripts/run_pipeline.py present; Simulator.from_pipeline present; verify identical in both recorded runs → **PASS**  
**Evidence:** `git show paper/arxiv:paper/figures/key_numbers_pipeline.json` default_n_mu_20 / n_mu_100 (stages, totals, muaps.notes.recipe_check); `git show paper/arxiv:paper/figures/key_numbers_f2.json` datasets_d2_d3.runtimes

## Fork 11 — Activation defaults (pool, rate coding, twitch/force)  *(recorded)*

**Use:** the shipped activation defaults (MotoneuronPool / TwitchPool).  
**Alternative is legitimate when:** none needed; when sub-20 ms intervals matter, clip the ISI draw (open item C2).  
**Why:** onion skin, threshold skew, twitch range and MVC calibration all pass their literature checks; the one failure is the missing refractory floor of the Gaussian renewal ISI.

| option | key numbers | verdict |
|---|---|---|
| onion-skin rate coding (5–40 pps) | model: min rate 5, peak 40 pps; measured @drive 0.5: 89 active, 7.6–36.5 pps, Spearman(threshold, rate) = -0.98; @drive 1.0 max 42.9 pps (renewal-process estimate runs ~5 % above the nominal peak); sanity: @drive 0.7: first-recruited 40.0 Hz vs last 5.6 Hz (98 active) | USE (passes) |
| ISI variability: Gaussian renewal, CoV 0.167 | median per-MU ISI CoV 0.16 (model ISI_CV = 0.167); ISIs < 20 ms: 1.58 %; min ISI 9.8 ms | CoV fine; OPEN ITEM — no refractory floor (1.58 % of ISIs < 20 ms, min 9.8 ms): clip the ISI or draw gamma/lognormal |
| recruitment thresholds / twitch range | threshold skewness +1.25 (median/max = 0.12); twitch P_max/P_min = 95 | USE (passes) |
| force calibration (twitch pool) | plateau force(%MVC) @ [0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0] = [7.4, 16.0, 30.3, 46.8, 69.8, 92.2, 102.5]; C6: RMS(50 % MVC)/RMS(100 % MVC) = 0.56; force plateau %MVC = [7, 16, 30, 46, 69, 85, 100] at drive [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0] | USE — ≈100 %MVC at full drive, monotone |

**Current default:** defaults unchanged → **PASS** (open item: C2 refractory floor (FAIL in C.json, not a known-flag))  
**Evidence:** _results/validation/C.json C1–C3, C6; S.json (simulator sanity)
