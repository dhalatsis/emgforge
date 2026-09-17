# EMG forward models and MUAP validation evidence

## Research conclusion

The literature supports a hierarchy of sanity checks rather than a universal “healthy
MUAP shape.” The strongest checks are mathematical identities and controlled responses:
linearity, source balance, analytical-cylinder reproduction, mesh convergence,
propagation delay, attenuation with source distance, and predictable changes under
electrode or fibre perturbations. Absolute duration, amplitude, phase count and spectral
limits depend on the muscle, recording montage, electrode area, tissue thickness,
filtering and how the waveform was extracted. They should be validated against a matched
experimental reference population, not used as context-free constants.

This conclusion agrees with the structure-based modelling literature, which separates
source description, motor-unit organization, volume conduction, recording configuration,
and recruitment/firing behaviour.[^1] It also matches modern numerical pipelines, where
the conductor solve is separated from fibre physiology and the final EMG is a linear
superposition of fibre or motor-unit contributions.[^2]

## Forward-model families

| Family | Representative work | What it contributes | Best use in emgforge validation |
|---|---|---|---|
| Single-fibre source models | Rosenfalck; Gydikov and Trayanova; Dimitrov/Dimitrova lineage[^3][^4] | IAP/current-source shape, onset, propagation and extinction | IAP units, derivative convention, finite-fibre end components |
| Layered planar analytical | Farina and Merletti; Farina and Rainoldi[^5][^6] | Separation of source, conductor and detection filters; fast controlled sweeps | Linearity, electrode filter, fat/depth low-pass trends |
| Layered cylindrical analytical | Blok et al.; Farina et al.[^7][^8] | Eccentric fibres, anisotropic muscle, bone/fat/skin and finite fibres | Primary analytical oracle and parameter sweeps |
| Idealized FEM | Lowery et al.; Stoykov et al.[^9][^10] | Nonsymmetric multilayer conductors and dispersive/capacitive variants | Mesh/domain/source convergence and analytical-cylinder comparison |
| MRI/anatomical FEM | Lowery et al.; Mesin et al.; Pereira Botelho et al.[^11][^12][^13] | Real boundaries, curved fibres, spatially varying anisotropy and deformation | Geometry canaries and subject-specific array patterns |
| Multiscale biophysical FEM | Mordhorst et al.[^14] | Membrane excitation, contraction, fatigue and deformation in one framework | Future checks for dynamic geometry and changing membrane properties |
| Fast adjoint digital twin | Maksymenko et al.[^2] | Electrode-wise adjoint solves, basis sources and large fibre populations | Architectural reference; cylinder error and array-feature benchmarks |
| Learned surrogate | BioMime and NeuroMotion[^15][^16] | Conditional generation of dynamic MUAP fields from a numerical teacher | Future surrogate-versus-teacher and out-of-distribution tests |
| Whole-pool analytical model | Petersen and Rostalski[^17] | Recruitment, rate coding, force, EMG, and explicit current conservation | Activation and interference-EMG validation beyond single MUAPs |

No single family is a complete oracle. Analytical models offer exact controlled cases but
simplify anatomy. Anatomical FEM handles geometry but introduces discretisation,
conductivity uncertainty and boundary choices. Learned generators inherit their
teacher’s assumptions. Experimental templates contain the desired physiology but also
unknown sources, noise, filtering and decomposition bias.

## Principles that can become strong sanity checks

### 1. The conductor and synthesis operators are linear

In the quasi-static formulation, tissue potential is obtained from a Poisson problem,
and detected EMG is the conductor response to a distributed membrane-current source.
Farina and Merletti explicitly separated temporal source properties from the spatial
volume-conductor and detection filters.[^5] The modern digital-twin formulation likewise
reuses linear basis solutions and forms arbitrary fibre contributions afterward.[^2]

Therefore source scaling, polarity inversion, fibre superposition and motor-unit
superposition are exact tests. They should pass to numerical precision before any
physiological trend is considered. A failure is a software or formulation error, not
biological variability.

### 2. A matched cylinder is the primary end-to-end numerical oracle

Farina et al. derived a multilayer cylindrical surface-EMG model with anisotropic muscle
and concentric bone, muscle, fat and skin layers.[^8] Blok et al. independently developed
a finite three-layer eccentric-source cylinder and found that adding skin improved
agreement with measured potential distributions.[^7] These models offer a rare case in
which complex volume conduction has an analytical reference.

Maksymenko et al. used this strategy to validate a numerical four-layer cylinder and
reported normalized mean-square errors of 3% for a fibre 1 mm from the muscle surface
and 5% at 11 mm.[^2] They attributed part of the residual to the analytical cylinder being
infinite and the numerical one finite. Their 5% normalized-MSE value is a useful initial
reference only if the same formula is reproduced and all remaining model differences are
documented. The validation report should state each error formula explicitly and also
include NRMSE. A more useful result is a convergence curve: error should fall with mesh
refinement and stabilize as the cylinder is extended.

Shape-only correlation is insufficient. The comparison must preserve sign and volts,
report NRMSE and amplitude ratio, and show any lag without silently optimizing it away.
The analytical-φ-to-synthesis subproblem should be much stricter because it excludes FEM
error; emgforge now gates it at machine precision.

### 3. Propagating and non-propagating components have different spatial signatures

Surface-array measurements show that MUAPs propagate away from the motor endplate in
both directions. Masuda and Sadoyama found that most extracted surface MUAPs were
triphasic and propagated symmetrically toward the tendons; more complex and asymmetric
waveforms also occurred and were associated with endplate scatter and excitation
delays.[^18] Conduction velocity is inferred from the delay between channels along the
fibre.[^19]

For two electrodes separated by `Δz` in a locally space-invariant region, the expected
delay is `Δt = Δz/v`. This is a strong local test of the physical-time engine and fibre
orientation. It becomes only approximate near boundaries, curved fibres, tissue
inhomogeneity or large electrodes.

The innervation zone and tendons add components that do not simply translate. The
digital-twin paper shows differential cancellation for electrodes straddling the NMJ,
propagating components in the bulk, and non-propagating components from AP generation
at the NMJ and extinction at the tendon.[^2] Roeleveld et al. reported a mainly negative
propagating wave followed by a positive wave simultaneously present across electrode
positions.[^20] Consequently an HD-array test should separately score propagation slope,
NMJ cancellation and simultaneous end-of-fibre activity.

Absolute polarity is montage- and reference-dependent. The robust rule is consistency
with the declared electrode order and sign convention.

### 4. Finite fibres must generate onset and termination effects

Gydikov and Trayanova showed that finite-fibre onset and termination alter the
extracellular potential, with distinct biphasic components near the endplate and fibre
end.[^4] Farina and Merletti’s Radon formulation incorporates generation and extinction
without approximating the source shape.[^5] More recent work confirms that muscle
shortening changes final phases in a way that depends strongly on electrode distance to
the myotendinous zone.[^21]

A useful test therefore changes tendon distance while holding the conductor fixed. The
late component should move continuously and change in relative amplitude; removing the
finite-end operator should cause a large, intentional test failure. A universal sign for
the last lobe is unsafe because it changes with electrode placement and convention.

### 5. Source distance and subcutaneous tissue act as spatial low-pass filters

Fuglevand et al. found that surface MUAP frequency content decreases steeply as the
electrode-to-motor-unit distance increases, and that the dominant detected contribution
came from fibres within roughly 10–12 mm for their model and montage.[^22] That distance
is not a universal detection boundary, but the attenuation and loss of high spatial
frequency are robust directions.

Farina and Rainoldi found that subcutaneous layers attenuate and widen the potential
distribution at the muscle surface.[^6] Lowery et al.’s multilayer FEM also showed that
fat, skin and bone alter amplitude, frequency content and circumferential decay, with
increasing fat thickness lowering amplitude/frequency and changing spread.[^9] These
findings justify monotonic controlled sweeps of source distance and fat thickness using
peak-to-peak amplitude, median frequency and spatial width.

The relation should not be inverted into a claim that amplitude uniquely determines
depth. Motor-unit size, fibre count, cancellation, electrode montage and conductivity
also affect amplitude.

### 6. Electrode geometry is part of the forward model

Farina, Cescon and Merletti systematically varied electrode size/shape, spatial filter,
interelectrode distance, fibre inclination/depth/length and subcutaneous thickness, and
showed that all affect surface SFAP amplitude and spectral content.[^23] Finite electrodes
approximately average the potential beneath their surface under practical EMG
conditions.[^24] Larger electrodes should therefore suppress fine spatial structure, and
differential montages should reject shared far-field components while introducing their
own spatial transfer function.

Validation inputs must record electrode area, shape, interelectrode distance, filter
order and channel polarity. Comparing a point-electrode monopolar simulation with a
large bipolar experimental template as if they were the same observable is not a valid
morphology test.

### 7. Conduction velocity links spatial and temporal scales

For a fixed spatial source and conductor, higher conduction velocity compresses the
waveform in time and shifts power upward in frequency. This relation underpins array
methods that estimate velocity from interchannel delay.[^19] Experimental low-threshold
MU data provide a concrete scale: Farina et al. reported a pre-fatigue conduction
velocity of 3.9 ± 0.2 m/s and action-potential duration of 11.1 ± 0.8 ms; after endurance,
velocity fell by 6.3% while duration increased by 9.8%.[^25]

Those numbers are informative, not global gates. The direction of change is the stronger
test. The model should also be tested at fixed fibre geometry so that changing `v` does
not silently change the sampled spatial window.

### 8. Motor-unit morphology requires explicit dispersion

A surface MUAP is a sum of SFAPs, and the fibre population is not perfectly synchronous.
Masuda and Sadoyama observed endplate spread up to 14 mm along the fibre direction and
associated some complex asymmetric MUAPs with junction scatter and excitation delay.[^18]
The expected controlled response is broader duration, lower coherent peak and often
lower median frequency as NMJ or conduction-velocity dispersion grows.

This is stronger than requiring every MUAP to be triphasic. Most MUAPs in one classic
array study were triphasic, but some had more than five phases.[^18] Phase count is useful
as a population statistic after matching the acquisition filter; it is not a universal
per-waveform rejection rule.

## Morphology features and how to use them

| Feature | Strong use | Main confounders |
|---|---|---|
| Signed waveform correlation | Matched implementation/oracle comparison | Time convention and electrode polarity |
| NRMSE in volts | Matched end-to-end comparison | Calibration and reference definition |
| Peak-to-peak / RMS | Controlled depth, size or recruitment sweep | Fibre count, cancellation, montage, conductivity |
| Active duration | Velocity and dispersion sweeps | Bandpass, threshold, tendon distance |
| Median/mean frequency | Depth/fat/velocity trends | Window, sampling, electrode filter, noise |
| DC-area ratio | Baseline or unbalanced transient diagnostic | Truncated acquisition window |
| Tail-energy ratio | Cropping, wraparound and centring diagnostic | A real event located at the window edge |
| Phase/turn count | Protocol-matched population comparison | Noise and bandwidth |
| End-of-fibre lobe ratio/timing | Tendon and fibre-length sweep | Electrode location and sign convention |
| Array propagation slope | Fibre direction and conduction velocity | Curvature, pennation, inhomogeneity |
| Spatial amplitude width | Source distance/fat/electrode area sweep | Muscle boundaries and anisotropy |

Clinical needle-EMG duration values should not be transferred directly to surface MUAPs.
Even within invasive recordings, Dumitru et al. showed that wider recording bandwidth
and improved signal-to-noise could extend measured MUAP duration from the conventional
roughly 10 ms toward 30 ms.[^26] The operational definition and acquisition chain are part
of the metric.

## Numerical and anatomical checks beyond morphology

Lowery et al. demonstrated that idealized cylinders can approximate amplitude-decay
trends when tissue thickness is chosen well, while subject-specific geometry can still
substantially alter waveform shape.[^11] Their work also examined capacitance and
dispersion; Stoykov et al. showed that plausible low-conductivity/high-permittivity
choices could materially reduce surface potential at 100 Hz, while emphasizing the
uncertainty in in-vivo properties.[^10] This argues for treating resistive quasi-static
physics as a declared model assumption and adding a sensitivity study before claiming
absolute spectral fidelity.

Pereira Botelho et al. built an MRI/DTI-informed forearm FEM with spatially varying
anisotropy and used reciprocity to solve once per electrode rather than once per fibre.[^13]
Mesin et al. showed that shortening-induced geometry and conductivity-tensor changes can
substantially alter amplitude and frequency content.[^12] Small-perturbation continuity,
coordinate-frame rotation, reciprocity and mesh refinement are therefore as important as
single-waveform resemblance for emgforge’s MRI path.

At the population level, Petersen and Rostalski explicitly formulate fibres so they are
never a net current source or sink.[^17] Source balance is a strong numerical invariant.
Their integration of MU recruitment, rate coding, force and surface EMG also provides a
reference for validating the activation layer separately from the MUAP forward model.

## Recommended evidence program

The immediate release gate should combine the exact analytical-operator test, the
metamorphic physics suite, existing solver contracts and tolerant regression snapshots.
The next Slurm campaign should produce a like-for-like FEM-cylinder convergence report.
After that, build array-level canaries for propagation/NMJ/tendon topology and controlled
fat, electrode and geometry sweeps.

Experimental validation should use decomposed multichannel MUAP templates with the raw
montage geometry, sampling rate, analogue/digital filters, muscle identity and subject
anatomy where possible. Split subjects between calibration and validation. Compare joint
feature distributions and spatial maps, retain absolute volts, and report how much each
simulator parameter was tuned to the validation data. A fit obtained after selecting
parameters on the same waveform is a calibration result, not an independent validation.

## Sources

[^1]: Stegeman, D. F., Blok, J. H., Hermens, H. J., and Roeleveld, K. “[Surface EMG models: properties and applications](https://doi.org/10.1016/S1050-6411(00)00023-7).” *Journal of Electromyography and Kinesiology* 10(5), 313–326 (2000).
[^2]: Maksymenko, K., Clarke, A. K., Mendez Guerra, I., Deslauriers-Gauthier, S., and Farina, D. “[A myoelectric digital twin for fast and realistic modelling in deep learning](https://doi.org/10.1038/s41467-023-37238-w).” *Nature Communications* 14, 1600 (2023).
[^3]: Rosenfalck, P. “[Intra- and extracellular potential fields of active nerve and muscle fibres](https://pubmed.ncbi.nlm.nih.gov/5383732/).” *Acta Physiologica Scandinavica Supplementum* 321, 1–168 (1969).
[^4]: Gydikov, A. A., and Trayanova, N. A. “[Extracellular potentials of single active muscle fibres: effects of finite fibre length](https://doi.org/10.1007/BF00318202).” *Biological Cybernetics* 53, 363–372 (1986).
[^5]: Farina, D., and Merletti, R. “[A novel approach for precise simulation of the EMG signal detected by surface electrodes](https://doi.org/10.1109/10.923782).” *IEEE Transactions on Biomedical Engineering* 48(6), 637–646 (2001).
[^6]: Farina, D., and Rainoldi, A. “[Compensation of the effect of sub-cutaneous tissue layers on surface EMG: a simulation study](https://doi.org/10.1016/S1350-4533(99)00075-2).” *Medical Engineering & Physics* 21(6–7), 487–497 (1999).
[^7]: Blok, J. H., Stegeman, D. F., and van Oosterom, A. “[Three-layer volume conductor model and software package for applications in surface electromyography](https://doi.org/10.1114/1.1475345).” *Annals of Biomedical Engineering* 30, 566–577 (2002).
[^8]: Farina, D., Mesin, L., Martina, S., and Merletti, R. “[A surface EMG generation model with multilayer cylindrical description of the volume conductor](https://doi.org/10.1109/TBME.2003.820998).” *IEEE Transactions on Biomedical Engineering* 51(3), 415–426 (2004).
[^9]: Lowery, M. M., Stoykov, N. S., Taflove, A., and Kuiken, T. A. “[A multiple-layer finite-element model of the surface EMG signal](https://doi.org/10.1109/10.995683).” *IEEE Transactions on Biomedical Engineering* 49(5), 446–454 (2002).
[^10]: Stoykov, N. S., Lowery, M. M., Taflove, A., and Kuiken, T. A. “[Frequency- and time-domain FEM models of EMG: capacitive effects and aspects of dispersion](https://doi.org/10.1109/TBME.2002.800754).” *IEEE Transactions on Biomedical Engineering* 49(8), 763–772 (2002).
[^11]: Lowery, M. M., Stoykov, N. S., Dewald, J. P. A., and Kuiken, T. A. “[Volume conduction in an anatomically based surface EMG model](https://doi.org/10.1109/TBME.2004.836494).” *IEEE Transactions on Biomedical Engineering* 51(12), 2138–2147 (2004).
[^12]: Mesin, L., Joubert, M., Hanekom, T., Merletti, R., and Farina, D. “[A finite element model for describing the effect of muscle shortening on surface EMG](https://doi.org/10.1109/TBME.2006.870256).” *IEEE Transactions on Biomedical Engineering* 53(4), 593–600 (2006).
[^13]: Pereira Botelho, D., Curran, K., and Lowery, M. M. “[Anatomically accurate model of EMG during index finger flexion and abduction derived from diffusion tensor imaging](https://doi.org/10.1371/journal.pcbi.1007267).” *PLOS Computational Biology* 15(8), e1007267 (2019).
[^14]: Mordhorst, M., Heidlauf, T., and Röhrle, O. “[Predicting electromyographic signals under realistic conditions using a multiscale chemo-electro-mechanical finite element model](https://doi.org/10.1098/rsfs.2014.0076).” *Interface Focus* 5, 20140076 (2015).
[^15]: Ma, S., Clarke, A. K., Maksymenko, K., Deslauriers-Gauthier, S., Sheng, X., Zhu, X., and Farina, D. “[Conditional generative models for simulation of EMG during naturalistic movements](https://doi.org/10.1109/TNNLS.2024.3438368).” *IEEE Transactions on Neural Networks and Learning Systems* 36(5), 9224–9237 (2025).
[^16]: Ma, S., Mendez Guerra, I., Caillet, A. H., et al. “[NeuroMotion: open-source platform with neuromechanical and deep network modules to generate surface EMG signals during voluntary movement](https://doi.org/10.1371/journal.pcbi.1012257).” *PLOS Computational Biology* 20(7), e1012257 (2024).
[^17]: Petersen, E., and Rostalski, P. “[A comprehensive mathematical model of motor unit pool organization, surface electromyography, and force generation](https://doi.org/10.3389/fphys.2019.00176).” *Frontiers in Physiology* 10, 176 (2019).
[^18]: Masuda, T., and Sadoyama, T. “[The propagation of single motor unit action potentials detected by a surface electrode array](https://doi.org/10.1016/0013-4694(86)90146-X).” *Electroencephalography and Clinical Neurophysiology* 63(6), 590–598 (1986).
[^19]: Soares, F. A., Carvalho, J. L. A., Miosso, C. J., de Andrade, M. M., and da Rocha, A. F. “[Motor unit action potential conduction velocity estimated from surface electromyographic signals using image processing techniques](https://doi.org/10.1186/s12938-015-0079-4).” *BioMedical Engineering OnLine* 14, 84 (2015).
[^20]: Roeleveld, K., Blok, J. H., Stegeman, D. F., and van Oosterom, A. “[Volume conduction models for surface EMG; confrontation with measurements](https://doi.org/10.1016/S1050-6411(97)00009-6).” *Journal of Electromyography and Kinesiology* 7(4), 221–232 (1997).
[^21]: Rodríguez-Falces, J., Malanda, A., and Navallas, J. “[Effects of muscle shortening on single-fiber, motor unit, and compound muscle action potentials](https://doi.org/10.1007/s11517-021-02482-z).” *Medical & Biological Engineering & Computing* 60, 349–364 (2022).
[^22]: Fuglevand, A. J., Winter, D. A., Patla, A. E., and Stashuk, D. “[Detection of motor unit action potentials with surface electrodes: influence of electrode size and spacing](https://doi.org/10.1007/BF00201021).” *Biological Cybernetics* 67, 143–153 (1992).
[^23]: Farina, D., Cescon, C., and Merletti, R. “[Influence of anatomical, physical, and detection-system parameters on surface EMG](https://doi.org/10.1007/s00422-002-0309-2).” *Biological Cybernetics* 86, 445–456 (2002).
[^24]: van Dijk, J. P., Lowery, M. M., Lapatki, B. G., and Stegeman, D. F. “[Evidence of potential averaging over the finite surface of a bioelectric surface electrode](https://doi.org/10.1007/s10439-009-9680-7).” *Annals of Biomedical Engineering* 37, 1141–1151 (2009).
[^25]: Farina, D., Gazzoni, M., and Merletti, R. “[Spike-triggered average torque and muscle fiber conduction velocity of low-threshold motor units following submaximal endurance contractions](https://doi.org/10.1152/japplphysiol.01127.2004).” *Journal of Applied Physiology* 98, 1495–1502 (2005).
[^26]: Dumitru, D., King, J. C., and Nandedkar, S. D. “[Comparison of single-fiber and macro electrode recordings: relationship to motor unit action potential duration](https://doi.org/10.1002/(SICI)1097-4598(199711)20:11%3C1381::AID-MUS5%3E3.0.CO;2-6).” *Muscle & Nerve* 20(11), 1381–1388 (1997).
