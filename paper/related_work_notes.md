# Related-work notes

One or two plain sentences per key reference: what the work did, and how it was
validated. BibTeX keys refer to `paper/refs.bib`. These feed the related-work
paragraphs of the introduction and the validation section; they are not meant to be
pasted verbatim.

## Source models (transmembrane potential and current)

- **rosenfalck1969intra** — Core-conductor analysis of the intra- and extracellular
  fields of active nerve and muscle fibres; the transmembrane current is the second
  spatial derivative of the intracellular action potential (IAP), and the analytical IAP
  `Vm(z) = 96 z^3 e^{-z} - 90 mV` is introduced here. Model fields were compared with
  potentials recorded from single fibres; this IAP is the source used by emgforge,
  Petersen & Rostalski 2019 and Maksymenko et al. 2023.
- **andreassen1981relationship** — Review that states the IAP-to-extracellular relation
  as a convolution of the transmembrane current with a volume-conductor weighting
  function, and confronts the 1969 model with recorded single-fibre potentials.
- **nandedkar1983simulation** — Line-source simulation of single-fibre action
  potentials as a function of fibre diameter, conduction velocity and radial distance;
  derives the amplitude/duration scaling laws (amplitude inversely proportional to
  velocity when the IAP is fixed in time) and checks them against single-fibre EMG
  recordings.
- **dimitrov1998precise** — Treats the motor-unit potential as a linear
  time-shift-invariant system whose input is the first temporal derivative of the IAP
  and whose impulse response is a dipole travelling along a finite fibre, with explicit
  generation (end-plate) and extinction (tendon) terms; validated by agreement with
  earlier analytical solutions at a fraction of the computing cost.
- **kleinpenning1990equivalent** — Derives the equivalent source that represents the
  extinction of an action potential at a fibre ending; these end-of-fibre terms are what
  make the finite-fibre source monopole-free (the identity emgforge checks at machine
  precision).
- **vanveen1993bioelectrical** — Computes single-fibre action potentials from measured
  transmembrane currents and from analytical IAPs and compares both with recorded SFAPs;
  the measured current fits best, but analytical and measured IAPs give comparable
  results.
- **griep1982calculation** — The reference design for validation: the same motor-unit
  action potential was both calculated and recorded, with fibre positions taken from
  post-hoc histology.

## Analytical cylinders and their confrontation with data

- **gootzen1991finite** — Bounded anisotropic cylinder with fibres of finite length;
  finite limb dimensions enhance the end-of-fibre component. Simulated surface MUAPs
  were compared with recordings and judged a "very good resemblance".
- **roeleveld1997motor**, **roeleveld1997distribution**, **roeleveld1997volume** — 52
  biceps motor units recorded with a 36-channel surface array plus scanning EMG: MUP
  amplitude decays with depth as an inverse power law (bipolar steeper than monopolar),
  and depth is about 0.2 times the surface width over which the MUP exceeds half its
  maximum. The 1997 J. Electromyogr. Kinesiol. paper confronts several volume-conductor
  models with these data: all decayed faster than the measurements.
- **blok2002three** — Three-layer (muscle, fat, skin) cylindrical model and software
  package; a distinct thin skin layer is needed to reproduce the lateral decay measured
  by Roeleveld et al.
- **merletti1999modeling1**, **merletti1999modeling2** — Tripole source in an
  anisotropic infinite medium with realistic electrode systems (Part I); Part II fits
  biceps array recordings at 10-30 % MVC and uses the model to interpret spectral and
  conduction-velocity estimates.
- **farina2001novel** — The layered planar conductor as a two-dimensional spatial
  filter, with the Radon transform handling generation and extinction; verified against
  the closed-form solution and shown to reproduce the spectral dips of single- and
  double-differential detection.
- **farina2004multilayer** — Multilayer cylindrical volume conductor (bone, muscle,
  fat, skin) solved in the Fourier domain with finite-length fibres and arbitrary
  electrodes; the model emgforge's analytical tier ports. Verified against limiting
  cases in the paper and, independently, against a finite-element solution by
  Maksymenko et al. 2023 (normalised MSE 3-5 %).
- **mesin2005analytical** — General analytical method for multilayer conductors (fat
  over a bipinnate muscle); the new solver is checked "by comparison with the known
  solution" for homogeneous planar layers, and the paper lists the analytic solutions
  available for checking numerical models.
- **ma2022analytical** — Analytical model for curvilinear fibres with an approximate
  conductivity tensor, compared with two finite-element models on the same anatomy:
  cross-correlation 0.98, normalised RMSE at most 0.04, median-frequency error about
  3 %.
- **carriou2016fast** — Fast high-density surface EMG generator in a cylindrical
  conductor; the numerical electrode integration is checked against the analytic
  electrode transfer function.

## Finite-element models and digital twins

- **lowery2002multiple** — Multilayer finite-element model of the surface EMG; shows
  how fat, skin and bone change amplitude, frequency content and lateral decay relative
  to a homogeneous muscle, after verifying the FEM against an analytical solution for
  the simplest geometry.
- **lowery2004volume** — MRI-based finite-element model of the upper arm compared with
  measured surface potentials: normalised RMS error 18-27 %.
- **kuiken2003effect** — Same modelling framework applied to fat thickness (3, 9,
  18 mm): RMS falls by 31, 80 and 90 % and cross-talk rises; used as a quantitative
  target in emgforge's fat-layer check.
- **botelho2019anatomically** — DTI-based model of the first dorsal interosseous with
  reciprocity lead fields and a Rosenfalck source with compensating end sources;
  simulated EMG RMS falls within the experimental interquartile range and the
  skewness/kurtosis of the signal match measured values.
- **maksymenko2023myoelectric** — "Myoelectric digital twin": finite-element volume
  conductor with hierarchical basis sources and adjoint (reciprocal) lead fields, fast
  enough to train deep networks. Validated against the Farina 2004 cylinder (NMSE
  3-5 %), against experimental RMS during wrist tasks, and downstream through motor-unit
  decomposition accuracy.
- **teklemariam2016finite** — Finite-element study of electrode design and muscle
  architecture; documents mesh-refinement convergence over six levels using signal RMS,
  the discretisation practice emgforge follows.

## Generative surrogates

- **ma2025conditional** (BioMime) — Conditional generative network trained
  adversarially on digital-twin MUAPs, producing MUAP waveforms for arbitrary
  volume-conductor parameters; held-out normalised RMSE 1.8 % against the physical
  model.
- **ma2024neuromotion** — Open-source platform combining a musculoskeletal model with
  BioMime to generate surface EMG during voluntary movement; validated in silico against
  the underlying physical model and by comparison with experimental movement EMG.

## Motor-unit pools and interference EMG

- **fuglevand1993models** — The canonical motoneuron-pool model: exponentially
  distributed recruitment thresholds, linear rate coding, renewal inter-spike intervals
  and a twitch model with a 100-fold range; plausibility judged through the simulated
  EMG-force relation.
- **fuglevand1992detection** — Dipole-based surface MUAP model showing that only motor
  units within about 10-12 mm of the electrode contribute significant energy, and that
  inter-electrode distance, not electrode area, sets the detection depth.
- **keenan2007experimentally** — Monte-Carlo screening of pool-model parameters
  against two experimental constraints (EMG amplitude vs. force and force variability
  vs. force); only 3 of 439 parameter sets passed, and the outcome was most sensitive to
  neural properties.
- **keenan2005influence** — Quantifies amplitude cancellation in the simulated surface
  EMG: about 33 % at 20 % excitation rising to 62-65 % at maximum.
- **petersen2019comprehensive** — Comprehensive open model of pool organisation,
  surface EMG (planar multilayer conductor) and force; the interference EMG amplitude
  distribution is near-Laplacian, consistent with experimental reports.
- **hamiltonwright2005physiologically** — Physiologically based simulator of clinical
  (needle) EMG validated against clinical MUP statistics and jitter.

## Validation practice and reference values

- **merletti2019tutorial** — Tutorial on surface EMG detection in space and time; the
  source of most of the phenomenological targets used in emgforge's tier-B checks
  (innervation-zone behaviour, end-of-fibre components, depth attenuation, electrode
  size and spacing).
- **stegeman2000surface** — Review of surface EMG models and what they can and cannot
  be used for; lists confrontation with measurements as the missing step for most.
