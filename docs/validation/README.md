# Forward-model validation plan

The validation strategy is a ladder. Each rung answers a different question, and a
failure should identify which part of the model is wrong. A realistic-looking MUAP is
not enough: many combinations of source, conductor, electrode and sign convention can
produce a plausible trace.

## Validation contract

| Gate | Question | Oracle | Status | Acceptance |
|---|---|---|---|---|
| A0: numerical contracts | Is the PDE problem well formed? | Source balance, residual, nullspace and reciprocity checks | Source balance, nullspace and convergence guards exist in `tests/fem`; explicit reciprocity remains | Existing solver tolerances |
| A1: analytical operator | Does analytical cylinder φ passed through the public synthesis pipeline reproduce the direct analytical MUAP? | Independent Farina four-layer `SignalGenerator` | Implemented in `tests/validation/test_cylinder_oracle.py` | Signed r ≥ 1 − 1e-13 and peak ratio 1 ± 2e-12 |
| A2: FEM cylinder | Does a like-for-like FEM cylinder converge to the analytical cylinder? | Same four layers, conductivities, source, electrode filter and fibre geometry | Existing heavy bench is useful but not yet like-for-like | Paper-matched normalized MSE ≤ 5% initially, plus a convergence trend with mesh refinement |
| B: physics response | Does changing one cause move the MUAP in the expected direction? | Metamorphic relations | Implemented for five high-value relations | Every relation passes without fitting to a snapshot |
| C: morphology | Is each waveform finite, transient, spectrally plausible and free of edge artifacts? | Protocol-aware feature envelopes | Feature extraction implemented; empirical envelopes remain | No universal hard range until montage-specific reference data exist |
| D: population realism | Do simulated MUAP populations match experimental joint distributions? | Decomposed HD-sEMG templates plus acquisition metadata | Planned | Pre-registered distributional tolerances |

The distinction between A1 and A2 matters. A1 isolates the φ-to-MUAP operator and can
be exact. A2 includes discretisation, finite-domain boundaries, source regularisation,
and any mismatch between the analytical and FEM conductors. Those effects should be
measured rather than hidden by peak normalisation or time alignment.

## A1: strict analytical-cylinder gate

The analytical model directly constructs the multilayer-cylinder transfer spectrum
`C(kz)` and synthesises a waveform. The validation adapter independently transforms
that spectrum into the real reciprocal line field `φ(z)`. The public
`field_to_muap(φ, bed, config)` path must reconstruct the same waveform.

The cases cover a symmetric central fibre, an off-axis asymmetric fibre, and a more
superficial fibre with lower conduction velocity and a smaller electrode. The gate
checks the time axis, signed shape and absolute amplitude. The analytical translation
retains its original millivolt output, so `analytical_muap` now converts it to volts at
the adapter boundary. Random 0.1 mm endplate and tendon offsets were removed from the
single-fibre oracle; scatter belongs in an explicit motor-unit test.

This gate catches FFT normalisation, IAP amplitude, polarity, fibre-end signs, time
orientation, reciprocal-field extraction and unit mistakes. It does not validate the
FEM solve because its input field is analytical.

## A2: like-for-like FEM-cylinder gate

The existing regression bench compares a five-layer finite cylinder and regularised
FEM source with a four-layer infinite analytical cylinder. Its high correlations are a
valuable characterization, but the residual cannot be assigned uniquely to FEM error.
A decisive A2 experiment should use the following sequence:

1. Build a four-layer cylinder matching the analytical radii and conductivity tensors.
2. Match the monopolar or differential electrode transfer function and reference
   convention. Record the sign convention explicitly.
3. Compare reciprocal `φ(z)` before synthesising a MUAP. This separates field error
   from synthesis error.
4. Compare the unnormalised MUAP in volts using signed correlation, NRMSE, peak ratio,
   DC-area ratio and tail-energy ratio. Report any alignment lag; do not silently shift.
5. Repeat at three mesh sizes and at two cylinder lengths. Error should decrease with
   mesh refinement and become insensitive to further domain extension.
6. Sweep source-to-skin distance, angular offset and electrode size. Keep cases away
   from tissue interfaces until the basic convergence gate passes.

Maksymenko et al. reported 3–5% normalized mean-square error for a numerical cylinder
against its analytical counterpart despite finite/infinite-domain differences. Their 5%
value is a defensible initial ceiling only when the same error definition is reproduced;
the report should state every formula and also include NRMSE. Once the models are truly
matched, the refinement curve should determine a tighter project-specific tolerance.

The Slurm job should write a small JSON report with parameters, package versions,
mesh statistics, solver residuals and all waveform metrics. Large meshes and fields may
remain external, but the case definitions and summary report should be committed.

## B: implemented physics-response checks

`tests/validation/test_physiology_invariants.py` implements relations that remain valid
across a wide range of waveform shapes:

| Relation | Expected response | Failure points toward |
|---|---|---|
| Linearity and polarity | Scaling `φ` by `a` scales the MUAP by `a`, including `a < 0` | Hidden normalization, nonlinear preprocessing or sign loss |
| Superposition | A motor-unit waveform equals the sum of its fibre waveforms | Fibre indexing, bed geometry or accumulation |
| Source distance | A broader, more distant monopole field attenuates peak-to-peak amplitude and median frequency | Volume-conductor filtering or spatial sampling |
| Conduction velocity | Higher velocity shortens duration and raises median frequency for fixed spatial geometry | Space/time conversion or coupled Fourier grids |
| Propagation | A 20 mm detector displacement along a 4 mm/ms fibre produces a 5 ms delay | Physical-time convention, fibre direction or sampling |
| Endplate scatter | Distributed NMJs broaden the MUAP and reduce coherent peak amplitude | Per-fibre offsets or motor-unit summation |

These are controlled model experiments, not claims that amplitude or median frequency
always identifies one physiological parameter in real recordings. Geometry, fat,
electrode montage and filtering confound those marginal associations.

## C: waveform feature checks

`emgforge.synthesis.waveform_features(t_ms, waveform)` provides a shared feature
definition for validation reports. `MUAPResult.metrics` uses the same implementation.
It returns peak and RMS amplitude, active duration, signed and absolute area,
`dc_area_ratio`, `tail_energy_ratio`, median/mean/dominant frequency, energy above a
configurable high-frequency cutoff, roughness and active phase count.

The most useful immediate diagnostics are:

- `dc_area_ratio`: catches a large one-signed residual or baseline problem while being
  invariant to amplitude and polarity.
- `tail_energy_ratio`: catches truncated, wrapped or badly centred waveforms.
- duration and median frequency together: expose time/space scaling errors better than
  either alone.
- phase count and end-of-fibre lobe metrics: useful for stratified reports, but too
  montage-sensitive for a global pass/fail threshold.
- peak-to-peak amplitude in volts: always retain it even when shape is also shown after
  normalisation. A normalized correlation cannot detect a 1000-fold unit error.

The next implementation step is to derive envelopes from experimental MUAP templates
stratified by muscle, electrode montage, source distance and acquisition bandpass.
Record the median and 1st/99th percentiles for each feature, plus pairwise relations such
as duration versus conduction velocity. Do not pool protocols with different filters.

## Prioritized next checks

1. **Like-for-like FEM cylinder convergence.** This closes the remaining gap in A and
   should run as a scheduled Slurm validation job.
2. **Reciprocity check.** Exchange a localized source and detector in the FEM model and
   verify the transfer coefficient within solver tolerance.
3. **HD array topology.** Verify two outward-propagating branches from the innervation
   zone, delay `Δz/v` in the bulk, cancellation for a differential pair straddling the
   NMJ, and a simultaneous non-propagating end-of-fibre component near the tendon.
4. **Fat and electrode sweeps.** Increasing source distance or subcutaneous thickness
   should attenuate and spatially widen the surface field; increasing electrode area
   should suppress fine spatial structure. Compare trends rather than one waveform.
5. **Geometry perturbations.** Rotate fibres, add pennation, move bone, shorten muscle
   and extend the domain. Require continuity for small perturbations and convergence
   with discretisation.
6. **Experimental population validation.** Compare multichannel templates in their
   original montage using waveform and spatial features, then validate interference
   EMG statistics separately from MUAP morphology.

## How to run

The fast validation tier needs NumPy, SciPy and pytest but no mesh:

```bash
pytest -q tests/validation tests/synthesis/test_metrics.py
```

Run the full pure-Python synthesis suite before merging:

```bash
pytest -q tests/synthesis tests/validation
```

FEM tests and A2 must run in an environment containing DOLFINx, Gmsh, PETSc and MPI,
inside an allocated compute node. Save the Slurm job ID and software versions in the
validation report. See [the bibliography](BIBLIOGRAPHY.md) for the evidence behind the
relations and the limits on interpreting them.

The first execution record for this validation layer is
[RUN_2026-09-11.md](RUN_2026-09-11.md).
