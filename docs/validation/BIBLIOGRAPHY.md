# Validation bibliography — forward models and MUAP phenomenology

Compiled 2026-09-10 from two literature surveys (forward models and their validation;
quantitative MUAP/EMG phenomenology). Purpose: every check in `scripts/validation/`
cites a principle from here, with the number it is held against. Where a source could
only be read in abstract, that is said. Check IDs (A0.1, B4b, …) refer to
`docs/validation/PLAN.md`.

---

## Part I — Forward models, and how each was validated

### Source models (transmembrane potential / current)

- **Rosenfalck 1969**, *Acta Physiol Scand Suppl* 321:1–168. Core-conductor theory: the
  transmembrane current is the second spatial derivative of the IAP, scaled by fibre
  radius/intracellular conductivity. Default IAP `Vm(z) = 96·z³·e^(−z) − 90 mV` (z in mm) —
  the source used by our engines (`emgforge.synthesis.iap`), by Petersen & Rostalski 2019
  and by Maksymenko 2023. → **A0.1, A0.5**.
- **Andreassen & Rosenfalck 1981**, *CRC Crit Rev Bioeng* 6:267 — the IAP→SFAP relation as
  a convolution of the transmembrane current with the volume conductor's weighting
  function; the experimental confrontation of the 1969 model. → **A0.1**.
- **Nandedkar & Stålberg 1983**, *Med Biol Eng Comput* 21:158 (doi 10.1007/BF02441531).
  Line-source SFAP; "for a given fibre the amplitude is inversely proportional to the
  conduction velocity" — *when the IAP is fixed in time*. Our engines fix the IAP in space,
  so the potential is CV-independent; see the note under A0.2.
- **Dimitrov & Dimitrova 1998**, *Med Eng Phys* 20:374 (doi 10.1016/S1350-4533(98)00014-9).
  MUP as a linear time-shift-invariant system: input = first time derivative of the IAP,
  impulse response = a dipole moving along the finite fibre, with generation at the
  end-plate and extinction at the tendons. Companion spectral papers *Med Eng Phys* 20:580
  and 20:702. → **A2.3, B4**.
- **Kleinpenning, Gootzen, van Oosterom & Stegeman 1990**, *Math Biosci* 101:41 and
  **Petersen 2016** (Lübeck workshop paper): the Farina–Merletti source
  `i(z,t) = d/dz[ψ(z−z_i−vt)p₁(z) − ψ(−z+z_i−vt)p₂(z)]` equals propagating terms +
  `GEN(t)δ(z−z_i)` + `EOF_k(t)δ(z−z_i∓L_k)`, and these are the **unique** terms that make
  `∫ i(z,t) dz = 0` for all t. → **A0.5** (monopole-free source).
- **van Veen et al. 1993**, *Biophys J* 64:1492 (PMC1262474): measured transmembrane
  current fits experimental SFAPs best; analytical and measured IAPs give comparable
  results. **Wallinga-de Jonge et al. 1985**, *EEG Clin Neurophysiol* 60:539: rat fast-fibre
  IAP amplitude ≈ 91 mV, rise 0.14 ms, 690 V/s. **Lateva & McGill 1998** (PMID 9851304):
  the slow afterwave comes from the IAP's negative afterpotential.
- **Griep et al. 1982**, *EEG Clin Neurophysiol* 53:388: the gold-standard design —
  calculated and recorded the *same* MUAP with post-hoc histological fibre positions.

### Analytical volume conductors

- **Gootzen, Stegeman & van Oosterom 1991**, *EEG Clin Neurophysiol* 81:152 (PMID 1708717).
  Bounded anisotropic cylinder + finite fibre; verified against surface MUAPs ("very good
  resemblance"); finite limb dimensions *enhance* the end-of-fibre effect. → **B4**.
- **Roeleveld, Stegeman et al. 1997a/b**, *Acta Physiol Scand* 160:175 and 161:465;
  **Roeleveld, Blok, Stegeman & van Oosterom 1997**, *J Electromyogr Kinesiol* 7:221 (PMID 11369265); **Blok, Stegeman
  & van Oosterom 2002**, *Ann Biomed Eng* 30:566. 52 biceps MUs, 36-channel sEMG + scanning
  EMG: MUP amplitude vs depth is an inverse power law (bipolar steeper than monopolar);
  MU depth ≈ 0.2 × the surface width over which the MUP exceeds 50 % of its maximum; *all*
  models decayed faster than the measurements and a distinct thin skin layer (3-layer
  model) fits best. → **B5, B6**.
- **Merletti, Lo Conte, Avignone & Guglielminotti 1999** (Parts I/II), *IEEE TBME*
  46:810/821. Tripole model (I = 24.6, −35.4, 10.8; a = 2.1, b = 4.8 mm) in an anisotropic
  infinite medium; Part II fits biceps array recordings at 10–30 % MVC ("closely
  approximated"). → **A0.5, B3**.
- **Farina & Rainoldi 1999**, *Med Eng Phys* 21:487: planar 4-layer transfer function
  (closed form reproduced in Petersen 2016) — fat attenuates *and widens* the surface
  distribution. **Farina & Merletti 2001**, *IEEE TBME* 48:637: the layered conductor as a
  2-D spatial filter; Radon transform for generation/extinction; explains the SD spectral
  dips. **Farina, Cescon & Merletti 2002**, *Biol Cybern* 86:445: the canonical sensitivity
  study (fat, inclination, depth, electrode size, IED, fibre length → amplitude, spectrum,
  CV estimate); tables paywalled. **Farina et al. 2002**, *Muscle Nerve* 26:681: crosstalk is
  mostly the non-propagating component; SD > DD; grows with IED. → **B4c, B8, B9, B10, B11**.
- **Farina, Mesin, Martina & Merletti 2004**, *IEEE TBME* 51:415 — the multilayer
  cylinder our `emgforge.analytical` ports. Parameters (Merletti & Muceli 2019 Fig. 7):
  σ_bone 0.02, σ_fat 0.05, σ_skin 1, σ_muscle 0.1/0.5 S/m. Validated *by others*:
  **Maksymenko et al. 2023** FEM vs this cylinder NMSE 3 % (1 mm deep) – 5 % (11 mm).
  → **A1, A2**.
- **Mesin & Farina 2004–2008**; **Mesin 2005** (WIT): a new solver is verified "by
  comparison with the known solution" for homogeneous planar multilayer tissue; lists the
  analytic solutions available for checking numerics (Clark & Plonsey 1968 infinite
  isotropic; Farina & Merletti 2001 planar; Gootzen 1991 / Blok 2002 / Farina 2004
  cylindrical). End-of-fibre components "have approximately constant amplitude on
  different channels". **Mesin, Damiano & Farina 2007** (J Neurosci Methods 160:327; PMID 17070925): pennation biases surface CV
  estimates (15° → 4.7–4.9 m/s for a true 4.0). **Mesin 2013** *Comput Biol Med* 43:942/953
  reviews. → **A0, B4a, B2**.
- **Carriou et al. 2016**, *Comput Biol Med* 74:54: fast HD-sEMG cylinder model;
  numerical electrode integration checked against the analytic electrode transfer
  function. → **B9**.
- **Ma et al. 2022**, *IEEE TBME* (PMID 34529557): curvilinear-fibre analytical model vs
  two FEMs — cross-correlation 0.98, nRMSE ≤ 0.04, MDF error ≈ 3 % — the cleanest template
  for "analytical vs FEM on one anatomy". → **A2.1**.

### Numerical (FEM) volume conductors

- **Lowery, Stoykov, Taflove & Kuiken 2002**, *IEEE TBME* 49:446: multilayer FEM; replacing
  outer muscle by resistive fat/skin at fixed distance *raises* amplitude and frequency
  content and steepens the decay; adding fat thickness *lowers* amplitude, frequency
  content and circumferential decay; bone near the surface raises the potential between
  bone and source. **Kuiken, Lowery & Stoykov 2003**, *Prosthet Orthot Int* 27:48: fat
  3/9/18 mm → RMS −31.3/−80.2/−90.0 %. **Lowery et al. 2004**, *IEEE TBME* 51:2138: MRI-based
  arm vs measured surface potentials, normalised rms error 18–27 %. **Stoykov et al. 2002**:
  capacitive effects up to 50 % at 100 Hz for extreme permittivities. → **B11, B12**.
- **Botelho, Curran & Lowery 2019**, *PLoS Comput Biol* 15:e1007267: DTI-based FDI model,
  reciprocity lead fields, Rosenfalck source with compensatory end sources; RMS within the
  experimental IQR; skewness/kurtosis of the EMG 0.21/5.41 simulated vs 0.49/5.44 measured.
  → **C4**.
- **Teklemariam et al. 2016**, *PLoS One* (PMC4757537): COMSOL h-refinement over 6 mesh
  levels with signal RMS as the convergence metric. → discretisation practice (A0.4a).
- **Maksymenko, Clarke, Mendez Guerra, Deslauriers-Gauthier & Farina 2023**, *Nat Commun*
  14:1600 (NeuroDec): FEM with hierarchical basis sources and adjoint/reciprocity; validated
  against the Farina 2004 cylinder (NMSE 3–5 %), against experimental wrist-task RMS, and
  downstream (decomposition RoA 93.8 % vs 82.4 %). → **A1, A2**.
- **Klotz et al. 2020** multi-domain (bidomain-type) model; **OpenDiHu 2024** — in-silico
  scalability rather than experimental validation.

### Motor-unit pools and whole-signal models

- **Fuglevand, Winter, Patla & Stashuk 1992**, *Biol Cybern* 67:143: dipole model; only MUs
  within 10–12 mm contribute significant energy; electrode area barely changes detection
  depth, IED does. **Fuglevand, Winter & Patla 1993**, *J Neurophysiol* 70:2470: the pool
  model (exponential thresholds, linear rate coding, renewal ISIs); plausibility judged by
  EMG–force. → **B5, B9, C3**.
- **Keenan & Valero-Cuevas 2007**, *J Neurophysiol* 98:1581: a Monte-Carlo fitness rubric —
  EMG amplitude vs force and force CoV vs force must match experiment; 3/439 parameter
  sets passed. **Keenan et al. 2005**, *J Appl Physiol* 98:120: amplitude cancellation 33 %
  at 20 % excitation → 62–65 % at maximum. → **C5, C6**.
- **Hamilton-Wright & Stashuk 2005**, *IEEE TBME* 52:171: validated by clinical MUP
  statistics and jitter. **Dimitrov et al. 2008**, *J Electromyogr Kinesiol* 18:35: fatigue
  simulation. **Petersen & Rostalski 2019**, *Front Physiol* 10:176: comprehensive model on
  the planar conductor; near-Laplacian amplitude distribution. **Arjunan et al. 2020**:
  equivalence tests on PSD peak, RMS and force.
- **Ma et al. 2024** BioMime (*IEEE TNNLS*; arXiv 2211.01856): conditional generative
  surrogate of NeuroDec MUAPs, held-out nRMSE 1.8 %. **NeuroMotion 2024**, *PLoS Comput
  Biol* 20:e1012257. **MUniverse 2025** (NeurIPS D&B): synthetic / hybrid / experimental
  decomposition benchmarks.

### Conductivity sets in circulation

| set | σ_muscle,z | σ_muscle,r | fat | skin | bone |
|---|---|---|---|---|---|
| Torino / Farina 2004 (ours) | 0.5 | 0.1 | 0.04–0.05 | **1.0** | 0.02 S/m |
| Gabriel-1996-based, 100 Hz | ≈1.33 | 0.267 | 0.021 | **4.6×10⁻⁴** | separate cortical/cancellous |

Anisotropy ratio: 5 (Gielen 1984, macroscopic) to 16 (Rush 1963). The skin value is the
largest disagreement in the field; Roeleveld et al. 1997 (JEK) and Blok 2002 show a distinct
thin skin layer changes the lateral decay enough to matter against data.

---

## Part II — Quantitative phenomenology (what a correct SFAP / MUAP / EMG looks like)

| feature | expected | source | check |
|---|---|---|---|
| Fibre conduction velocity | ≈4 m/s, range 3–5; populations 4.55 ± 0.33 (Zwarts 1988), 2.6–5.3 (Andreassen & Arendt-Nielsen 1987); CV(m/s) = 0.043·D(µm) + 0.83 (Blijham 2006); 3.4 %/°C (Troni 1991) | Merletti & Muceli 2019 | A2.4, B2, S5 |
| CV from arrays | cross-correlation of SD/DD channels on one side of the IZ; SD 0.1–0.2 m/s; IED > 10 mm unsuitable | Farina & Merletti 2004 | A2.4, B2 |
| Monopolar SFAP between IZ and tendon | triphasic + − +, dominant negative phase | Merletti & Muceli 2019 Fig. 2; Arabadzhiev 2013 | B1 |
| Surface MUAP duration | ≈15 ms for an SD MUAP of a 60 mm fibre at 4 m/s; ~20 ms typical | Merletti & Muceli 2019; Farina 2014 | B7, S8 |
| Intramuscular MUAP / SFAP | needle MUAP 8–15 ms, ~0.5 mV; SFAP > 200 µV, rise < 300 µs from fibres within 0.3 mm | Dumitru 1999; SFEMG guidelines 2019 | — |
| End-of-fibre component | same latency on all channels; onset at L/CV; present in monopolar, reduced by SD, further by DD; EOF/propagating grows with depth, comparable at ~22 mm | Merletti & Muceli 2019 §2.2.2; Gootzen 1991; Roeleveld 1998; Rodriguez-Falces & Place 2018 | A2.3, B4a–c |
| Crosstalk | mostly non-propagating; 20–30 mm lateral to a muscle; SD > DD; grows with IED | Farina 2002 (Muscle Nerve) | B4c |
| Innervation zone | bidirectional propagation; monopolar potentials mirror about the IZ; SD channel on the IZ ≈ zero and reverses phase | Masuda 1983/1985; Merletti & Muceli 2019 §2.2.1 | B3 |
| Spatial filters | SD gain \|2 sin(π e f_s)\|; zeros at f = n·v/IED (400 Hz at 10 mm, 4 m/s); mono > SD > DD detection depth | Lindström & Magnusson 1977; Lynn 1978; Disselhorst-Klug 1997 | B8, B4c |
| Point source in anisotropic medium | φ ∝ 1/√(σ_z ρ² + σ_r z²); isopotentials elongated by √(σ_z/σ_r) | Plonsey & Barr; Rush 1963; Malmivuo & Plonsey ch. 11 | A0, B12 |
| Amplitude vs depth | inverse power law, log-log linear; bipolar steeper; MU depth ≈ 0.2 × 50 %-width; only MUs within 10–12 mm contribute significant energy; SD single fibre at 1 % of the superficial max by ≈8 mm | Roeleveld 1997a/b; Fuglevand 1992; Merletti & Muceli 2019 Fig. 7 | B5, B6 |
| Fat | RMS −31/−80/−90 % at 3/9/18 mm; attenuation and widening; MNF down | Kuiken 2003; Farina & Rainoldi 1999; Lowery 2002 | B11 |
| Transverse extent | bipolar 24–32 mm, monopolar 72–96 mm (biceps MUs 15–25 mm deep) | Roeleveld/Stegeman 2013 | B6 |
| MU territory / scale | 5–10 mm diameter; 15–1500 fibres; largest monopolar MUAPs 1–2 mV; MVC RMS 0.2–1.5 mV | Buchthal 1957/59; Stålberg & Antoni 1980; Merletti & Muceli 2019 | S8, C8 |
| Amplitude ∝ fibre count | linear (MUAP = Σ SFAP) | Merletti & Muceli 2019; Roeleveld 1998 | A0.4c, B7 |
| Electrode size | Ø5 mm −3 dB at 100 c/m; Ø10 mm at 50 c/m; > 5 mm alters spectra | Merletti & Muceli 2019 Table 1 | B9 |
| IED | SD amplitude ∝ IED for small IED; saturates near λ/2 (≈20 mm) | De Luca 2002; Hermens 2000 | B10 |
| Spectrum | 95 % of power < 400–500 Hz; MDF 70–130 Hz at moderate force (biceps 90 ± 18, TA 116 ± 20); MDF, MNF ∝ CV | Stulen & De Luca 1981; Arendt-Nielsen & Mills 1985; J Clin Neurophysiol 1998 norms | A0.4d, C7 |
| EMG–force | between linear (FDI, soleus) and quadratic (biceps, deltoid) | Lawrence & De Luca 1983; Woods & Bigland-Ritchie 1983 | C6, S3–4 |
| Amplitude cancellation | 33 % at 20 % → 62–65 % at maximum | Keenan 2005 | C5 |
| Amplitude PDF | between Laplacian and Gaussian; super-Gaussian at ≤ 10 % MVC; ≈ Gaussian above 40–50 %; ARV/RMS 0.71–0.80 | Clancy & Hogan 1999; Nazarpour 2013 | C4 |
| Firing rates / ISI | 5–40 pps; onion skin; ISI CoV 0.1–0.3; refractory ~20 ms | De Luca & Hostage 2010; Dideriksen 2012 | C1, C2, S1–2 |
| Recruitment / twitch | thresholds right-skewed; twitch range ≈100× | Fuglevand 1993 | C3 |
| Pennation | 15° inclination → CV overestimated 15–25 %; DD least biased | Mesin et al. 2007 | (future) |
| Reciprocity / superposition / translation | Helmholtz reciprocity (Malmivuo & Plonsey eq. 11.30); linearity; z-invariance of layered conductors | Plonsey 1963; Farina & Merletti 2001 | A0.4b–c, A3 |

---

## Part III — Validation practices in the field, ranked, and where they land in our suite

1. Monopole-free source (∫ i dz = 0 ∀t) — Petersen 2016, Merletti 1999 → **A0.5** ✓ (7.7e-17).
2. FEM vs analytical multilayer cylinder on identical geometry — Maksymenko 2023 (NMSE 3–5 %), Ma 2022 (xcorr 0.98) → **A1.1–1.3, A2.1**.
3. Analytical limits (infinite anisotropic medium, planar multilayer) — Mesin 2005 → **A0.1–0.3**.
4. Discretisation convergence — Mesin 2005, Teklemariam 2016 → **A0.4a**.
5. Amplitude-vs-depth law and detection volume — Roeleveld 1997, Fuglevand 1992 → **B5, B6**.
6. Propagating vs non-propagating behaviour — Gootzen 1991, Dimitrov 1998, Mesin 2005 → **A2.3, B4**.
7. Layer material vs distance sign checks — Lowery 2002, Kuiken 2003 → **B11**.
8. Electrode / spatial-filter transfer functions — Farina & Merletti 2001, Lynn 1978 → **B8–B10**.
9. CV recovery — Merletti 1999 II, Farina 2002, Mesin et al. 2007 → **A2.4, B2, S5**.
10. Single-fibre scaling laws — Nandedkar & Stålberg 1983 → **A0.2, A0.4d**.
11. Source-level plausibility — Rosenfalck, Wallinga 1985 → **A0.1** (IAP fixed by the engines).
12. Surface MUAP shape vs recordings with known geometry — Griep 1982, Merletti 1999 II, Lowery 2004, Botelho 2019 → **not yet**: needs the WR HD-sEMG units (see PLAN.md, next steps).
13. Interference statistics and pool relations — Keenan 2007, Fuglevand 1993, Clancy & Hogan 1999 → **C1–C8, S1–S4**.
14. Quasi-static / capacitance — Stoykov 2002 → assumption stated; not tested.
15. Downstream-task validation — Maksymenko 2023, MUniverse → **not yet**.

---

## Part IV — Sources (with links)

Merletti & Muceli 2019 *J Electromyogr Kinesiol* 49:102363 (doi 10.1016/j.jelekin.2019.102363) ·
Campanini et al. 2022 *Sensors* 22:4150 (PMC9185290) · Fuglevand et al. 1992 (PMID 1627684) ·
Fuglevand, Winter & Patla 1993 *J Neurophysiol* 70:2470 · Roeleveld et al. 1997a (PMID 9208044),
1997b (PMID 9429653), 2001 (PMID 11369265), 1998 (PMID 9626247) · Roeleveld/Stegeman 2013
*J Electromyogr Kinesiol* (S1050641113000734) · Blok et al. 2002 (PMID 12086007) · Gootzen et al.
1991 (PMID 1708717) · Lowery et al. 2002 (doi 10.1109/10.995683), 2004 (PMID 15605861) · Kuiken,
Lowery & Stoykov 2003 *Prosthet Orthot Int* 27:48 · Stoykov et al. 2002 (PMID 12148814) · Farina &
Rainoldi 1999 (doi 10.1016/S1350-4533(99)00075-2) · Farina & Merletti 2001 (PMID 11396594) · Farina,
Cescon & Merletti 2002 (doi 10.1007/s00422-002-0309-2) · Farina et al. 2002 *Muscle Nerve* (doi
10.1002/mus.10256) · Farina et al. 2003 (doi 10.1109/TBME.2003.808830) · Farina, Mesin, Martina &
Merletti 2004 (doi 10.1109/TBME.2003.820998) · Farina & Merletti 2004 *J Neurosci Methods* (PMID
15003386) · Farina, Negro, Gazzoni & Enoka 2008 (PMC2544462) · Farina, Merletti & Enoka 2014
(PMC4254845) · Mesin & Farina 2004 (PMID 15376500), 2006 (PMID 16686399); Mesin et al. 2006 (PMID
16602565, 17070925, 17073322); Mesin 2005 WIT (BIO05010FU); Mesin, Merletti & Vieira 2011 *J
Biomech* 44:1096; Mesin 2013 (PMID 23489655; S0010482513000784) · Merletti et al. 1999 I/II (PMID
10396899/10396900) · Merletti, Farina & Gazzoni 2003 (doi 10.1016/S1050-6411(02)00082-2) · Carriou
et al. 2016 (PMID 27183535) · Ma et al. 2022 (PMID 34529557); Ma et al. 2024 BioMime (arXiv
2211.01856); NeuroMotion 2024 *PLoS Comput Biol* 20:e1012257 · Maksymenko et al. 2023 *Nat Commun*
14:1600 (doi 10.1038/s41467-023-37238-w) · Botelho, Curran & Lowery 2019 *PLoS Comput Biol*
15:e1007267 · Teklemariam et al. 2016 (PMC4757537) · Klotz et al. 2020 (PMID 31529291); OpenDiHu
2024 (S187775032400084X) · Keenan et al. 2005 (PMID 15377649); Keenan & Valero-Cuevas 2007 (doi
10.1152/jn.00577.2007) · Hamilton-Wright & Stashuk 2005 (PMID 15709654) · Dimitrov & Dimitrova
1998 (PMID 9773690; 10098616); Arabadzhiev 2013 (doi 10.1007/s11517-013-1037-6) ·
Dimitrov et al. 2008 (PMID 16963280) · Petersen & Rostalski 2019 *Front Physiol* 10:176; Petersen
2016 (Lübeck IME) · Arjunan et al. 2020 (PMID 31774372) · Rosenfalck 1969; Andreassen &
Rosenfalck 1981 (PMID 7044677) · Nandedkar & Stålberg 1983 (doi 10.1007/BF02441531) · Griep et
al. 1982 (PMID 6175501) · van Veen et al. 1993 (PMC1262474) · Wallinga-de Jonge et al. 1985 ·
Rodriguez-Falces et al. 2012 (doi 10.1007/s11517-012-0879-7); Rodriguez-Falces & Place 2018
(PMC5852100) · Lateva & McGill 1998 (PMID 9851304) · Kleinpenning et al. 1990 *Math Biosci* 101:41
· Clancy & Hogan 1999 (PMID 10356879) · Nazarpour et al. 2013 (PMC3878385) · Lawrence & De Luca
1983 (PMID 6874489) · Woods & Bigland-Ritchie 1983 (PMID 6650674) · Beck et al. 2005 (PMID 15935960)
· De Luca & Hostage 2010 (doi 10.1152/jn.01018.2009); De Luca & Contessa 2015 (PMC4295621) ·
Dideriksen et al. 2012 (PMC3378401) · Stulen & De Luca 1981 (PMID 7275132) · Lindström & Magnusson
1977 *Proc IEEE* 65:653 · Arendt-Nielsen & Mills 1985 (PMID 2578364) · Sinderby et al. 1996 (PMID
8606692) · Lynn et al. 1978 (doi 10.1007/BF02442444) · initial-MDF norms *J Clin Neurophysiol* 1998
(PMID 9563580) · De Luca 2002 Delsys tutorial · Hermens et al. 2000 SENIAM (doi
10.1016/S1050-6411(00)00027-4) · Rush, Abildskov & McFee 1963 *Circ Res* 12:40 · Gielen,
Wallinga-de Jonge & Boon 1984 (PMID 6503387) · Gabriel, Lau & Gabriel 1996 *Phys Med Biol* 41 ·
Malmivuo & Plonsey 1995 ch. 11 (bem.fi/book/11) · Plonsey & Barr, *Bioelectricity* · Masuda et al.
1983/1985 (doi 10.1109/TBME.1985.325614) · Beretta Piccoli et al. 2014 (doi 10.1002/mus.23934) ·
Disselhorst-Klug, Silny & Rau 1997 (PMID 9210816) · Buchthal et al. 1957/1959 · Stålberg & Antoni
1980 (PMC490585) · Sanders et al. 2019 SFEMG guidelines · Dumitru, King & Rogers 1999 (PMID
10366227) · Andreassen & Arendt-Nielsen 1987 (PMC1192232) · Del Vecchio et al. 2017 (PMID
28751374), 2018 (doi 10.1111/apha.12930) · Zwarts et al. 1988; Arendt-Nielsen & Zwarts 1989 · Troni
et al. 1991 (PMID 20870519) · Blijham et al. 2006 (PMID 16424073) · Håkansson 1956 (PMID 13339449)
· Nordander et al. 2003 *Eur J Appl Physiol* 89:514 · Lundsberg et al. 2024 *Sci Rep* (PMC10869353).
