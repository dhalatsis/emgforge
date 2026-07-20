import base64, pathlib
def b64(p): return base64.b64encode(pathlib.Path(p).read_bytes()).decode()
cmp_img=b64("compare_methods.png"); show_img=b64("showcase.png"); sch_img=b64("schematic.png"); fcu_img=b64("fcu_compare.png"); diag_img=b64("diag_fill.png"); her_img=b64("herring.png"); muap_img=b64("muap_compare.png"); score_img=b64("muap_scorecard.png"); lit_img=b64("muap_literature.png"); fixed_img=b64("muap_fixed.png"); flen_img=b64("fibre_length_check.png"); ica_img=b64("ica_wr2.png"); overlay_img=b64("overlay_real_model.png"); fa_img=b64("field_architecture.png"); nd_img=b64("validate_field_neurodec.png")
HTML=f"""<title>Muscle-fibre geometry for EMG simulation</title>
<style>
:root{{
  --bg:#f6f7f8; --panel:#ffffff; --ink:#171d21; --ink-soft:#4a565d; --line:#dde2e5;
  --accent:#1f7a68; --accent-soft:#e4f0ec; --red:#c0392b; --grey:#7a7f83; --green:#1f8a54;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#101619; --panel:#171f23; --ink:#e7edf0; --ink-soft:#9fb0b8; --line:#26333a;
  --accent:#4fbfa6; --accent-soft:#123029; --red:#e07a6f; --grey:#9aa1a6; --green:#5cc98a;
}}}}
:root[data-theme="dark"]{{
  --bg:#101619; --panel:#171f23; --ink:#e7edf0; --ink-soft:#9fb0b8; --line:#26333a;
  --accent:#4fbfa6; --accent-soft:#123029; --red:#e07a6f; --grey:#9aa1a6; --green:#5cc98a;
}}
:root[data-theme="light"]{{
  --bg:#f6f7f8; --panel:#ffffff; --ink:#171d21; --ink-soft:#4a565d; --line:#dde2e5;
  --accent:#1f7a68; --accent-soft:#e4f0ec; --red:#c0392b; --grey:#7a7f83; --green:#1f8a54;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.6;font-size:17px;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:820px;margin:0 auto;padding:56px 24px 96px}}
header{{border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:8px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 14px}}
h1{{font-family:var(--serif);font-weight:600;font-size:38px;line-height:1.12;margin:0 0 14px;
  text-wrap:balance;letter-spacing:-.01em}}
.lede{{font-size:19px;color:var(--ink-soft);margin:0;max-width:64ch}}
h2{{font-family:var(--serif);font-weight:600;font-size:26px;margin:52px 0 6px;letter-spacing:-.01em;
  text-wrap:balance}}
h2 .n{{font-family:var(--mono);font-size:14px;color:var(--accent);margin-right:12px;font-weight:400}}
h3{{font-size:16px;margin:28px 0 8px;letter-spacing:.01em}}
p{{margin:14px 0;max-width:66ch}}
strong{{color:var(--ink);font-weight:640}}
em{{color:var(--ink)}}
a{{color:var(--accent)}}
figure{{margin:26px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px;overflow-x:auto}}
figure img{{display:block;width:100%;min-width:680px;height:auto;border-radius:4px}}
figure.wide{{width:min(1180px,96vw);position:relative;left:50%;transform:translateX(-50%)}}
figure.wide img{{min-width:0}}
figcaption{{font-size:13.5px;color:var(--ink-soft);margin-top:12px;padding:0 4px;max-width:none}}
figcaption b{{color:var(--ink)}}
.callout{{background:var(--accent-soft);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:8px;padding:18px 22px;margin:24px 0}}
.callout .q{{font-family:var(--serif);font-size:18px;font-weight:600;margin:0 0 8px}}
.callout p{{margin:8px 0}}
table{{width:100%;border-collapse:collapse;margin:22px 0;font-size:14.5px}}
th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
thead th{{font-family:var(--mono);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-soft);font-weight:600;border-bottom:2px solid var(--line)}}
tbody tr td:first-child{{font-weight:600}}
.tag{{font-family:var(--mono);font-size:12px;padding:1px 7px;border-radius:4px;white-space:nowrap}}
.tag.red{{color:var(--red);background:color-mix(in srgb,var(--red) 14%,transparent)}}
.tag.grey{{color:var(--grey);background:color-mix(in srgb,var(--grey) 16%,transparent)}}
.tag.green{{color:var(--green);background:color-mix(in srgb,var(--green) 16%,transparent)}}
ul{{max-width:66ch;padding-left:22px}}
li{{margin:9px 0}}
.wrong li strong{{color:var(--red)}}
:root[data-theme="dark"] .wrong li strong{{color:var(--red)}}
.mono{{font-family:var(--mono);font-size:.92em}}
.refs{{margin-top:56px;padding-top:22px;border-top:1px solid var(--line);font-size:13.5px;color:var(--ink-soft)}}
.refs h2{{font-size:15px;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:10px}}
.refs p{{margin:6px 0;max-width:none}}
.divider{{height:1px;background:var(--line);border:0;margin:8px 0}}
code{{font-family:var(--mono);font-size:.88em;background:color-mix(in srgb,var(--ink) 7%,transparent);
  padding:1px 5px;border-radius:4px}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">MRI → volume conductor · fibre geometry</p>
  <h1>Placing muscle fibres from MRI for EMG simulation</h1>
  <p class="lede">Where each fibre begins, ends and is innervated sets the source model for simulated EMG.
  <span class="mono">emgforge</span> currently offers two ways to derive this from an MRI segmentation; both
  misrepresent the anatomy in ways that change the signal. This describes a third method, shows how it aligns
  with the muscle-architecture and EMG literature, and validates it against a real subject's HD-sEMG.</p>
</header>

<div class="callout" style="border-left-width:4px;margin-top:26px">
  <p class="q">Executive summary</p>
  <p><strong>The problem.</strong> <span class="mono">emgforge</span> derives each muscle's EMG source model from an
  MRI segmentation in two ways, and both place a <em>single fibre spanning the whole muscle</em> with its innervation
  zone pinned at mid-length — anatomically impossible: real forearm fibres are ~40–60&nbsp;mm (about a fifth of the
  muscle), innervated in the proximal third.</p>
  <p><strong>The method.</strong> A third approach — short, physiological fibres tiled on a harmonic direction field
  (contained and non-crossing by construction), with innervation zones drawn from the measured atlas distribution.</p>
  <p><strong>The evidence.</strong> The subject's <em>own</em> HD-sEMG settles it: 15 blindly-decomposed motor units
  have a propagation extent of <b>40–50&nbsp;mm</b> — several-fold shorter than the 226&nbsp;mm full-muscle fibre, and
  irreconcilable with it at <em>any</em> electrode pitch. On the two quantities HD-sEMG can measure — propagation
  extent and innervation-zone scatter — the new method matches the subject and the existing methods do not.</p>
  <p><strong>Scope, stated plainly.</strong> This validates the <em>source model</em> (fibre length + IZ). The
  direction field's longitudinal axis is separately confirmed on a second subject (PM: median 0.9° vs an independent
  hand annotation); its pennation and curvature remain literature-validated (would need diffusion imaging), and the
  EMG is a single subject.</p>
</div>

<h2><span class="n">01</span>The two existing methods, and what is wrong with them</h2>
<p>Both current methods live in <span class="mono">fiber_directions.py</span> and share one defining property:
a <strong>single fibre spanning the entire muscle</strong>, innervated once at its geometric middle.</p>
<ul>
  <li><span class="tag red">RED</span> — <strong>morphing disk</strong> (<span class="mono">fiber_path_morphing</span>,
  constant offset): one fibre follows the centre-line from end to end.</li>
  <li><span class="tag grey">GREY</span> — <strong>fusiform taper</strong> (taper parameter): the same single
  fibre, tapering toward the tendon ends. This imitates NeuroDec, the state-of-the-art decomposition simulator.</li>
</ul>
<figure>
  <img alt="Three-row comparison of RED, GREY and NEW fibre methods on muscle L13" src="data:image/png;base64,{cmp_img}">
  <figcaption><b>Figure 1 — the three methods on one muscle (FDS, a 222&nbsp;mm belly).</b> Every fibre carries an
  <span style="color:#e08214;font-weight:600">orange ring</span> (begin) and a <b>red square</b> (end). Top two rows
  <span class="tag red">RED</span> and <span class="tag grey">GREY</span>: one fibre spans the whole muscle, so its
  <b>single innervation zone (NMJ) is stuck at mid-length</b> (x≈0). Third and fourth rows
  <span class="tag green">NEW</span>: short fibres <b>tiled in series</b> whose NMJs are placed at the muscle's
  measured innervation zones (<b>black</b> = atlas IZ fractions, FDS 0.42/0.72; <b>grey</b> = geometric fill bands
  where L<sub>f</sub>&nbsp;≪&nbsp;L<sub>m</sub> requires more in-series junctions than surface EMG resolves). The
  bottom row is the whole belly at full density (438 fibres); the black NMJs line up into vertical bands at the atlas
  fractions.</figcaption>
</figure>
<p>For the <em>conductivity-tensor direction</em> alone the two are not badly wrong — the fibre axis is roughly
along the muscle either way. The problems are all in the <strong>source model</strong>, which is exactly what this
stage feeds downstream:</p>
<ul class="wrong">
  <li><strong>Fibre length = muscle length.</strong> This implies <span class="mono">L<sub>f</sub>/L<sub>m</sub>=1</span>;
  measured forearm ratios are <span class="mono">0.18–0.41</span> (Lieber &amp; Fridén 2000). A single 222&nbsp;mm
  fibre does not exist in a forearm muscle.</li>
  <li><strong>Innervation zone pinned at the mid-point.</strong> Real forearm IZs sit in the <em>proximal third</em>
  and are frequently <em>multi-band</em> (Barbero–Merletti atlas; Safwat 2007; Lateva 2010). The electrode signal
  depends strongly on IZ position.</li>
  <li><strong>End-of-fibre effect at the wrong distance.</strong> The largest non-propagating part of the surface
  MUAP is generated where the action potential reaches the tendon. A full-length fibre places that ~111&nbsp;mm from
  the IZ; the real distance is ~23&nbsp;mm (L<sub>f</sub>/2) — a 5× error in a dominant waveform feature (Gootzen
  1991; Rodriguez-Falces 2018).</li>
  <li><strong>No pennation.</strong> Fibres run parallel to the centre-line; forearm muscles run at ~2–12° to their
  tendon (Lieber 1992; this atlas), which shifts both the σ tensor and the source location.</li>
  <li><strong>Grey's taper has no anatomical basis.</strong> The reference it imitates (NeuroDec, Maksymenko 2023)
  does <em>not</em> taper fibres to a point — it runs them tendon-to-tendon with a Tukey end-window.</li>
</ul>

<h2><span class="n">02</span>What the literature says should be happening</h2>
<p>Three well-established facts define what a correct fibre model must reproduce.</p>
<h3>1 · Fibres are much shorter than the muscle</h3>
<p>A forearm muscle fibre is ~40–60&nbsp;mm (Lieber 1990: FCU ~42&nbsp;mm; Fridén, Lovering &amp; Lieber 2004:
FCU ~63&nbsp;mm, with ~2× regional variation), i.e. <span class="mono">L<sub>f</sub>/L<sub>m</sub>&nbsp;≈&nbsp;0.2</span>.
In a 222&nbsp;mm belly (FDS, Fig&nbsp;1, whose own fibres run ~75&nbsp;mm) that is <strong>three or more fibres
end-to-end</strong>, each with its own two ends and its own innervation zone. Only the body's very longest fibres (in the ~50&nbsp;cm sartorius) reach hundreds of mm.</p>
<h3>2 · Muscles have a definite fibre architecture</h3>
<figure>
  <img alt="Schematic of fusiform, unipennate and bipennate muscle fibre architecture" src="data:image/png;base64,{sch_img}">
  <figcaption><b>Figure 2 — the three architecture classes.</b> <b>Fusiform</b> (brachioradialis): long fibres
  roughly spanning the belly, one innervation band. <b>Unipennate</b> (FPL, EPL): short fibres tilting to a single
  aponeurosis. <b>Bipennate / herringbone</b> (FCU, ECU, pronator teres): a <em>central tendon</em> with two fibre
  populations attaching from either side. A fibre-placement method must take the muscle's architecture class as an
  input — the fibres attach to these aponeuroses.</figcaption>
</figure>
<h3>3 · Innervation is one zone per motor unit, scattered across the muscle</h3>
<p>Every muscle fibre has a single endplate; the fibres of one motor unit share <strong>one</strong> localised
innervation band (NeuroDec models exactly one NMJ per unit). Across the population of units the innervation zones
<strong>scatter</strong> along the muscle (Saitou 2000: FCU IZs "scattered around the belly"), and the IZ location
varies enormously <em>between</em> muscles — it is the one routinely-measured MUAP characteristic that encodes
architecture (Barbero, Merletti &amp; Rainoldi 2012, atlas of 43 muscles).</p>
<figure>
  <img alt="Innervation-zone location across 29 forearm muscles vs the fixed-0.5 assumption of red/grey" src="data:image/png;base64,{lit_img}">
  <figcaption><b>Figure 3 — innervation-zone location across 29 forearm muscles.</b> Each row is a muscle; green
  dots are its innervation-zone fraction(s) (proximal&nbsp;0 → distal&nbsp;1); the red line marks the fixed
  mid-muscle (0.5) that red/grey assume. Real IZs <b>span roughly 0.13–0.87</b> and cluster in the <b>proximal
  third</b> — so a fixed-0.5 IZ typically sits <b>~0.2 of muscle length</b> away from the nearest real zone, and
  <b>cannot represent the many muscles with multiple IZ bands</b> (FDS, FDP, pronator teres, brachioradialis, FCU…).
  Provenance: these forearm IZ fractions come from the openly-readable primary sources (Safwat 2007; Saitou 2000;
  Lateva 2010) at thirds-level resolution — read the positions as approximate; the Barbero atlas corroborates them.</figcaption>
</figure>
<p>A fourth, source-physics fact ties these together: the action potential propagates <strong>±L<sub>f</sub>/2</strong>
from the innervation zone and extinguishes at the tendon, producing the end-of-fibre potential there (Rosenfalck
1969; Gootzen 1991). Short fibres therefore produce a <em>short</em> propagation extent — the quantity HD-sEMG
measures directly.</p>

<h2><span class="n">03</span>The new method — short physiological fibres on a contained field</h2>
<p>The method separates the two things the anatomy provides: a smooth <strong>direction field</strong> derived from
the muscle's shape, and <strong>begin/end/IZ positions</strong> imposed from measured architecture.</p>
<h3>How the direction field is solved</h3>
<p>Solve for one number <strong>φ</strong> at every interior point — like a temperature. φ obeys Laplace's equation
(<span class="mono">∇²φ = 0</span>): fix φ&nbsp;=&nbsp;0 on the origin aponeurosis, φ&nbsp;=&nbsp;1 on the insertion
aponeurosis, and allow no flow through the muscle's outer surface (insulated walls). The fibre direction is
<span class="mono">∇φ</span>, and a fibre is a streamline that follows it.</p>
<div class="callout">
  <p class="q">The heat analogy</p>
  <p>Hold one tendon sheet at 0&nbsp;°C, the other at 100&nbsp;°C, wrap the sides in insulation. Heat flows through
  the muscle along smooth curves that hug the insulated surface — those curves are the fibres. Insulated walls ⇒ flow
  can't leave (<strong>contained</strong>); one gradient direction per point ⇒ flow lines <strong>can't cross</strong>.
  Both are theorems of the field, not tuning. This is the established rule-based approach used for cardiac fibre
  fields (Bayer 2012) and validated for skeletal muscle against DTI: Choi &amp; Blemker 2013 report 96.8% of
  simulated fibre directions within 30° (51% within 10°) of diffusion imaging in tibialis anterior, and Handsfield
  2017 find no significant pennation-angle or fascicle-length difference from DTI in gastrocnemius.</p>
</div>
<h3>How length and innervation are imposed</h3>
<p>Streamlines are cut to the atlas fibre length L<sub>f</sub>; each motor unit is given <strong>one</strong>
innervation zone drawn from the muscle's atlas IZ distribution (with a Tukey end-window for the tendon extinction),
and the pennation tilt comes from the cross-section shape. Across the whole muscle these single-IZ units form the
banded innervation pattern the anatomy shows.</p>
<figure>
  <img alt="FCU bipennate herringbone from three perspectives, each a central section, with begin/end/IZ markers" src="data:image/png;base64,{her_img}">
  <figcaption><b>Figure 4 — the method on the bipennate FCU (the flagship result).</b> Shown from <b>three
  perspectives</b>, each a <b>central section</b> in its own plane so every fibre genuinely begins on <em>that</em>
  outline: <b>(1) side section</b> (length × thickness) — the herringbone chevron, with a zoom because the 12° lean is
  shallow on a long, thin muscle; <b>(2) face section</b> (length × width); <b>(3) zoom</b> of the chevron;
  <b>(4) transverse cross-section</b> through the distal innervation zone (φ=0.62) — each fibre is a <b>dot</b> (its
  cut end) and the ticks show the in-plane direction <b>converging radially onto the tendon</b>
  (<span style="color:#d62728;font-weight:600">★</span>). Markers: <span style="color:#e08214;font-weight:600">orange
  ring</span> = begin (on the outline), <b>red square</b> = end (on the tendon), <b>NMJ dot</b> = mid-fibre endplate.
  Fibres are grouped into in-series innervation compartments pinned to the atlas IZ fractions (FCU φ = 0.17 and 0.62)
  so no gap exceeds L<sub>f</sub>/L<sub>m</sub>. <b>Panels (5)–(6)</b> render every fibre (2,580) as a density check:
  the field tiles the whole belly with no gaps, and each fibre inserts on the aponeurosis at its own ~12° heading.</figcaption>
</figure>
<p>Against the two existing methods (Figure&nbsp;1), the difference is structural: <strong>many short fibres with
anatomically-placed innervation zones</strong>, contained and non-crossing by construction, in place of one
full-length fibre with a single mid-muscle NMJ. Every ingredient is an established, cited result:</p>
<table>
  <thead><tr><th>Ingredient</th><th>What we use</th><th>Precedent</th></tr></thead>
  <tbody>
    <tr><td>Direction field</td><td>Fibres = streamlines of a harmonic field with aponeurosis boundary conditions</td><td>Choi &amp; Blemker 2013; cardiac LDRB, Bayer 2012</td></tr>
    <tr><td>Validation of the field</td><td>vs DTI: 96.8% within 30°, 51% within 10°; vs NeuroDec hand-annotated planes on PM: median 0.9° (long axis, Fig&nbsp;S6)</td><td>Choi &amp; Blemker 2013 (DTI, tib.&nbsp;ant.); Handsfield <em>et&nbsp;al.</em> 2017 (gastroc.); this work (PM)</td></tr>
    <tr><td>Finite length + IZ + end-of-fibre</td><td>Finite fibres, L₁/L₂ semi-lengths, one NMJ per unit, Tukey end-window</td><td>NeuroDec, Maksymenko <em>et&nbsp;al.</em> 2023</td></tr>
    <tr><td>Per-muscle priors</td><td>Fibre length, pennation angle &amp; type, IZ position</td><td>Lieber 1990/92; Holzbaur 2005; Safwat 2007; Saitou 2000; Lateva 2010</td></tr>
    <tr><td>Source physics</td><td>Rosenfalck IAP → travelling tripole → tendon extinction</td><td>Rosenfalck 1969; Andreassen &amp; Rosenfalck 1981; Plonsey 1977</td></tr>
    <tr><td>Conductivity</td><td>σ aligned to fibres, anisotropy ≈ 4.5:1 (0.40 / 0.09 S/m)</td><td>Gielen 1984; Botelho, Curran &amp; Lowery 2019</td></tr>
  </tbody>
</table>
<h3>What the field does — and how much it actually changes the direction</h3>
<p>To be precise about where the improvement comes from, it helps to separate three things that are easy to
conflate:</p>
<ul>
  <li><strong>The harmonic solve</strong> (fix φ on the tendon caps, insulated walls) provides the
  <strong>longitudinal direction plus two guarantees</strong>: fibres stay <em>inside</em> the muscle and
  <em>never cross</em>. This is the field's indispensable job — it is what lets you tile <em>many short</em> fibres
  in series at all. Red/grey have one long fibre, so they never need it.</li>
  <li><strong>Pennation is imposed from the atlas</strong>, as an analytic lean of that longitudinal direction toward
  the aponeurosis — <em>not</em> discovered by the solve. (A pure harmonic solve with naive aponeurosis boundaries
  overshoots to ~60°, far past the real ~5–12°, so the measured angle is imposed rather than emergent.)</li>
  <li><strong>The resulting direction differs from red/grey by exactly the pennation angle</strong> — and that scales
  with architecture, from a fusiform muscle where the field is almost identical to the centre-line, to a bipennate one
  where it is not.</li>
</ul>
<p>Quantified across muscles (supplementary <b>Figure&nbsp;S5</b>): the field-vs-centre-line angle rises from
<b>2.4°</b> (fusiform brachioradialis) to <b>12°</b> (bipennate FCU), where the fibres visibly converge on the
central tendon — a chevron red/grey cannot represent at any angle. The important qualifier, picked up in §04: a linear
surface array is only <em>weakly</em> sensitive to this direction difference, so the dominant <em>measurable</em> win
remains fibre <strong>length</strong>, not direction.</p>
<p>Simulated through the full <span class="mono">emgforge</span> synthesis, the three geometries are graded against
published sEMG values in supplementary <b>Figure&nbsp;S7</b>: the one row that discriminates is propagation extent
(NEW stops at the aponeurosis, red/grey run the whole muscle); conduction velocity and MUAP duration don't separate
the methods. That is a <em>simulated</em> cross-check — the decisive test is against the subject's real EMG, next.</p>
<div class="callout">
  <p class="q">Honest limitations</p>
  <p>Pennation angle is imposed from the atlas, not measured — it can't be read off a belly-only T2 scan. The
  per-muscle priors (L<sub>f</sub>, θ, IZ) are Lieber's cadaver/literature values for each muscle <em>type</em>,
  correctly assigned per muscle through WR's label→name map — but not measured from <em>this subject's</em> own MRI,
  and that map currently exists only for WR (extending to other subjects needs their own label→muscle maps).
  Thick muscles need a multipennate boundary condition to keep fibres both short and shallow.
  The simulated MUAP numbers (Fig&nbsp;S7) are analytical-monopole (no FEM mesh available here), so treat the
  millisecond figures as indicative. None of these affect the robust output the validation below tests — physiological fibre length and
  innervation-zone placement. (The direction field is validated separately, against the DTI literature, not by this
  EMG.)</p>
</div>
<p>It is a drop-in for red/grey: same input (segmentation → centre-line + cross-section), same outputs (per-fibre
path + tangents + innervation position for the source; a direction field for the σ tensor), same config schema.</p>

<h2><span class="n">04</span>Validation against experimental EMG</h2>
<p>The subject WR — the <em>same person</em> as the MRI — has real HD-sEMG on disk: a 39-electrode single-differential
linear array along the forearm at 2&nbsp;kHz (<span class="mono">emg-decomposition/data/processed/WR2</span>). Because
the propagation extent of a motor unit's action potential <em>is</em> its fibre length made visible, this directly
tests which fibre model is right. (The model muscle is the bipennate FCU; exactly what this recording does and does
not pin down is spelled out in the callout after the figures.)</p>
<figure>
  <img alt="Motor-unit model shown as propagating-wave waterfalls: narrow V for NEW, wide V for RED/GREY" src="data:image/png;base64,{fixed_img}">
  <figcaption><b>Figure 5 — propagation extent as the travelling wave (the 'V').</b> Each faint line is one electrode's
  MUAP, stacked by position along the muscle; the potential is generated at the innervation zone (dashed) and
  propagates away in both directions, tracing the two arms of a <b>'V'</b>. <b>Left — RED = GREY:</b> one full-length
  fibre, so the wave runs to <em>both</em> tendon tips — a <b>wide V</b>. <b>Right — NEW:</b> short pennate fibres, so
  the wave extinguishes at the aponeurosis ~L<sub>f</sub>/2 out — a <b>narrow V</b>. The width of the V is the
  propagation extent, which equals fibre length — the quantity to compare against the real recording.</figcaption>
</figure>
<figure>
  <img alt="Fibre length and propagation extent: model vs cadaver literature vs real WR HD-sEMG" src="data:image/png;base64,{flen_img}">
  <figcaption><b>Figure 6 — fibre length and propagation extent: model vs cadaver vs real WR.</b> <b>(1) Fibre
  length:</b> FCU fascicles are ~4–6&nbsp;cm across cadaver studies (Lieber 1990 ~42&nbsp;mm; Fridén, Lovering &amp;
  Lieber 2004 ~63&nbsp;mm). The model's fibres (mean 44, range 25–65&nbsp;mm) sit inside that band; a single 226&nbsp;mm
  fibre is several times too long (≈4–5×). <b>(2) Propagation extent:</b> a typical real WR unit's action potential
  spans ~40–50&nbsp;mm (about 5 of 39 electrodes at 8–10&nbsp;mm pitch) — matching <span class="tag green">NEW</span>'s
  ~44&nbsp;mm, not <span class="tag red">RED/GREY</span>'s 226&nbsp;mm. <b>(3) Pitch-independent:</b> a real unit lights
  up ~13–18% of the array — like NEW (~19%), nothing like RED/GREY (100%).</figcaption>
</figure>
<p>To move beyond a single unit, the recording was decomposed blind into individual motor units with native cBSS
(convolutive blind source separation / fastICA), and the propagation extent of each was measured from its
spike-triggered-averaged MUAP.</p>
<figure>
  <img alt="WR2 blind decomposition: 15 motor-unit propagation extents vs NEW and RED/GREY, plus IZ scatter" src="data:image/png;base64,{ica_img}">
  <figcaption><b>Figure 7 — 15 real motor units, decomposed blind, land on the new method.</b> <b>(left)</b> the
  propagation extent of each of 15 decomposed motor units (~12 distinct; silhouette 0.94–0.98) from five WR2
  contractions: <b>median 40&nbsp;mm</b> at 8&nbsp;mm pitch, <b>50&nbsp;mm</b> at 10&nbsp;mm pitch. The distribution sits on top of
  <span class="tag green">NEW</span> (~44&nbsp;mm) and has <em>zero</em> units near <span class="tag red">RED/GREY</span>
  (226&nbsp;mm). <b>(right)</b> each unit carries <b>one</b> innervation zone; pooled, they scatter along the muscle —
  one IZ per unit, many units forming a scattered population. Caveat: electrode pitch is assumed 8–10&nbsp;mm (the raw
  OTB file is not on disk); a couple of units read artefactually long, so the median is the honest statistic.</figcaption>
</figure>
<p>Finally, the real units and the simulated ones can be put on a <em>single</em> axis. The trick that makes this
fair is to align every trace to its <strong>own</strong> innervation zone: then the only thing being compared is how
far the potential spreads from that zone — no assumption about where the muscle sits under the array, and no reliance
on waveform shape (which depends on fibre depth, conduction velocity and tissue).</p>
<figure>
  <img alt="Registration-free overlay: the whole real WR motor-unit population vs the NEW and RED/GREY models, amplitude vs distance from innervation zone" src="data:image/png;base64,{overlay_img}">
  <figcaption><b>Figure 8 — real motor units and the models on one axis, each aligned at its innervation zone.</b>
  <b>Top:</b> one representative real unit (cBSS, n=301, silhouette&nbsp;0.94) and the two models as travelling-wave
  'V's on a common ±150&nbsp;mm scale — the real 'V' and NEW's are narrow; RED/GREY's runs the whole muscle.
  <b>Bottom:</b> the peak-to-peak spatial envelope of <b>all 14 decomposed real units</b> (the 15th of Fig&nbsp;7
  fails the stricter envelope-quality filter; faint grey) with their population median, against the two models. Real amplitude collapses within a few tens of mm of the IZ (median
  <b>≈40–50&nbsp;mm</b> at 8–10&nbsp;mm pitch), tracking <span class="tag green">NEW</span> (imposed
  L<sub>f</sub>≈51&nbsp;mm, mean cut fibre ~44&nbsp;mm); <span class="tag red">RED/GREY</span> stays above threshold
  across the whole ±113&nbsp;mm muscle. Because each trace is referenced to its own IZ, this holds for <em>any</em>
  electrode pitch. It is a spread-and-single-IZ comparison, not a waveform-shape fit — the honest conclusion is
  order-of-magnitude: real units are several-fold shorter than RED/GREY and of NEW's order.</figcaption>
</figure>
<div class="callout">
  <p class="q">Verdict</p>
  <p>Across 15 decomposed motor units (~12 distinct), the measured propagation extent is <b>40–50&nbsp;mm</b> — in the
  tens-of-mm range the new method predicts (~44&nbsp;mm), and <b>several times shorter</b> than the 226&nbsp;mm
  full-muscle (FCU) fibre of the two existing methods. Each unit spans only 3–9 electrodes, so no plausible electrode pitch
  reconciles the data with a whole-muscle fibre. The real innervation structure — one zone per unit, scattered across
  the population — is what the new method imposes and what the existing fixed-mid-muscle NMJ cannot represent. On the
  two quantities that HD-sEMG can actually measure — propagation extent and innervation-zone scatter — the new method
  matches the subject's own EMG and the existing methods do not. (The exact millimetres assume the OTB-standard
  8–10&nbsp;mm pitch, since the raw file with the true spacing is not on disk; the order-of-magnitude conclusion holds
  for any realistic pitch.)</p>
</div>
<div class="callout">
  <p class="q">What this validates — and what it doesn't</p>
  <p><strong>Validated here</strong> (the Verdict above), on the subject's own muscle: the <em>source model</em> —
  fibre length and innervation-zone placement, the two quantities HD-sEMG can measure. The new method matches WR's EMG
  on both; the two existing methods do not.</p>
  <p><strong>Not tested by this EMG:</strong> the direction field. Its <em>longitudinal</em> axis is nonetheless
  <em>subject-specifically</em> confirmed — on a second subject (PM) it recovers NeuroDec's independent hand-annotated
  fibre planes to median 0.9°, all 14 muscles within 10° (Fig&nbsp;S6) — so the field is no longer only a
  literature-transfer from DTI (Choi &amp; Blemker; Handsfield). What remains literature-only is the field's finer
  structure (pennation, curvature): NeuroDec's planes encode just the long axis, WR's MRI is T2-only, and the bipennate
  direction difference (§03) is anyway invisible to a linear array — a 12° lean shifts the along-muscle projection by
  ~2%, and decomposing on the transverse grid dimension shows no convergence signature. Demonstrating that finer
  structure would need diffusion imaging or a denser tendon-registered 2-D array.</p>
</div>

<h2><span class="n">05</span>Supplementary figures</h2>
<p>Supporting views kept out of the main line.</p>
<figure>
  <img alt="Diagnostic: direction field streamlines fill the muscle" src="data:image/png;base64,{diag_img}">
  <figcaption><b>Figure S1 — the raw direction field.</b> Streamlines of the harmonic field (no finite length, no
  IZ) fill the muscle, run longitudinally, follow its shape, stay inside and don't cross — the field before length
  and innervation are imposed.</figcaption>
</figure>
<figure>
  <img alt="3D fibres in muscle volume and non-crossing proof" src="data:image/png;base64,{show_img}">
  <figcaption><b>Figure S2 — filled in 3D, and non-crossing.</b> Left: FCU filled in 3D with finite fibres (coloured
  by in-series band) inside the muscle volume. Right: streamlines seeded on a 4&nbsp;mm grid stay distinct through the
  whole muscle (min gap 1.6&nbsp;mm) — a guarantee of the single gradient field.</figcaption>
</figure>
<figure>
  <img alt="Bipennate fibres on the real FCU muscle converging on a central tendon" src="data:image/png;base64,{fcu_img}">
  <figcaption><b>Figure S3 — the three methods on FCU (label 8).</b> Same layout and markers as Figure 1:
  <span class="tag red">RED</span>&nbsp;/&nbsp;<span class="tag grey">GREY</span> one full-length fibre with a single
  mid-muscle NMJ; <span class="tag green">NEW</span> short in-series fibres with NMJs at FCU's atlas IZ (0.17, 0.62),
  shown both representative and full-density (179 fibres), with the non-crossing check.</figcaption>
</figure>
<figure>
  <img alt="FCU MUAPs for the three fibre-geometry methods through the emgforge synthesis" src="data:image/png;base64,{muap_img}">
  <figcaption><b>Figure S4 — single-electrode MUAPs through the emgforge synthesis.</b> All three geometries run
  through the real MUAP synthesis (analytical monopole lead field, r=0.990 vs the Farina&nbsp;2004 model). Single-fibre
  SFAPs (top) and motor-unit MUAPs (bottom) are all brief and in-range — a single electrode barely distinguishes the
  methods, which is why the discriminating evidence is spatial (Figures 5–8), not in the single-channel waveform.</figcaption>
</figure>
<figure>
  <img alt="Field direction departure from the centre-line by architecture class, and transverse convergence cross-sections" src="data:image/png;base64,{fa_img}">
  <figcaption><b>Figure S5 — how much the field's direction departs from red/grey, by architecture.</b> <b>Top:</b> the
  angle between the new field and the centre-line equals the imposed pennation — <b>2.4°</b> for a fusiform muscle
  (brachioradialis) up to <b>12°</b> for a bipennate one (FCU); red/grey assume <b>0° for every muscle</b>.
  <b>Bottom:</b> transverse cross-sections — green ticks are the in-plane fibre direction; for the bipennate FCU they
  <b>converge on the central tendon</b> (a chevron), which red/grey cannot represent at any angle, while for the
  fusiform muscle there is almost nothing to represent. Direction gain is <b>near-zero for straight muscles, categorical
  for bipennate</b> — but a linear surface array is only weakly sensitive to it (a 12° lean shortens the along-muscle
  projection by ~2%), which is why the <em>measured</em> EMG win is fibre length, not direction.</figcaption>
</figure>
<figure>
  <img alt="Harmonic field validated against NeuroDec hand-annotated fibre planes on subject PM, per-muscle angle" src="data:image/png;base64,{nd_img}">
  <figcaption><b>Figure S6 — the direction field checked against an independent hand annotation on a second subject
  (PM).</b> A DWI-free, subject-specific analogue of the DTI validation: for each muscle the harmonic field (from PM's
  shape alone) is compared to the NeuroDec digital asset's hand-annotated aponeurosis planes. The field recovers
  NeuroDec's fibre axis to <b>median 0.9°, all 14 muscles within 10°</b>. <em>Honest read:</em> NeuroDec's two-plane
  model encodes only the <em>longitudinal</em> axis (not pennation or curvature), so a naive PCA long-axis agrees
  equally well — this moves the field's <em>longitudinal direction</em> from literature-transfer to subject-confirmed,
  but the field's distinctive features (pennation, curvature, containment) still need finer ground truth (DWI) to
  test.</figcaption>
</figure>
<figure>
  <img alt="Scorecard of simulated sEMG characteristics for the three methods vs literature" src="data:image/png;base64,{score_img}">
  <figcaption><b>Figure S7 — simulated sEMG characteristics graded against published values.</b> Green = matches, red =
  mismatch, yellow = approximate, blue = non-discriminating (an imposed input). The one discriminating row is
  <b>propagation extent</b>. The scorecard reads the one-sided <em>semi-length</em> (IZ→tendon): NEW ~31&nbsp;mm ≈
  L<sub>f</sub>/2, RED/GREY ~102&nbsp;mm ≈ half the muscle. The array records both arms, so a unit's full band is
  tens of mm for NEW versus the whole ~226&nbsp;mm muscle for RED/GREY — the same length contrast §04 measures
  directly against real data (NEW fibre ~44&nbsp;mm). Conduction velocity (an input) and MUAP duration
  (dispersion-driven) don't distinguish the methods; IZ position is yellow because NEW's specific fractions only
  approximate a broad real distribution (Saitou 2000).</figcaption>
</figure>

<div class="refs">
  <h2>Key references</h2>
  <p>Choi &amp; Blemker 2013, <em>PLoS ONE</em> 8:e77576 · Handsfield, Bolsterlee <em>et al.</em> 2017,
  <em>Biomech Model Mechanobiol</em> 16:1845 · Bayer <em>et al.</em> 2012, <em>Ann Biomed Eng</em>
  40:2243 · Maksymenko <em>et al.</em> 2023, <em>Nat Commun</em> 14:1600 · Lieber &amp; Fridén 2000,
  <em>Muscle&nbsp;&amp;&nbsp;Nerve</em> 23:1647 · Lieber <em>et al.</em> 1990, <em>J Hand Surg</em>
  15A:244 · Lieber, Jacobson <em>et al.</em> 1992, <em>J Hand Surg</em>
  17A:787 · Fridén, Lovering &amp; Lieber 2004, <em>J Hand Surg</em> 29A:909 (PMID 15465243) · Holzbaur,
  Murray &amp; Delp 2005, <em>Ann Biomed Eng</em> 33:829 · Barbero, Merletti &amp;
  Rainoldi 2012, <em>Atlas of Muscle Innervation Zones</em> · Safwat &amp; Abdel-Meguid 2007, <em>Folia
  Morphol</em> 66:83 · Saitou <em>et al.</em> 2000, <em>J Human Ergol</em> 29:35 · Lateva, McGill &amp; Johanson 2010,
  <em>J Appl Physiol</em> 108:1530 · Rosenfalck 1969, <em>Acta Physiol
  Scand</em> Suppl 321 · Andreassen &amp; Rosenfalck 1981, <em>CRC Crit Rev Bioeng</em> 6:267 · Plonsey 1977,
  <em>Proc IEEE</em> 65:601 · Gootzen, Stegeman &amp; van Oosterom 1991, <em>Electroenceph Clin Neurophysiol</em>
  81:152 · Rodriguez-Falces &amp; Place 2018, <em>Eur J Appl Physiol</em> 118:501 · Nandedkar, Sanders &amp;
  Stålberg 1988, <em>Muscle &amp; Nerve</em> 11:151 · Botelho, Curran &amp; Lowery 2019, <em>PLoS Comput Biol</em>
  15:e1007267 · Gielen <em>et al.</em> 1984, <em>Med Biol Eng Comput</em> 22:569 · Farina, Mesin, Martina &amp;
  Merletti 2004, <em>IEEE Trans Biomed Eng</em> 51:415 · Bischoff, Stålberg <em>et al.</em> 1994,
  <em>Muscle &amp; Nerve</em> 17:842.</p>
  <hr class="divider">
  <p>Prototype &amp; figures: <span class="mono">emgforge/_results/mri_check/noura_fibre/</span> ·
  citations independently web-verified 2026-07-19.</p>
</div>
</div>
"""
pathlib.Path("artifact.html").write_text(HTML)
print("wrote artifact.html", len(HTML), "bytes")
