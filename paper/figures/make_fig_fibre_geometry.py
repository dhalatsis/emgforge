"""Study figure — does fibre geometry matter? Straight (Poisson, morphing-disk) vs
harmonic-streamline fibre beds of the WR FCU: the same 5×5 grid, the same motor units
(centre + size of the released pool, fibres re-drawn on each bed), the same direct
line-source recipe. Panels: (a) side views with the S/M/L/XL units' fibres; (b) fibre-level
facts (length distributions, depth below skin along the fibre); (c) MUAPs of the S/M/L/XL
units at the centre electrode, straight vs harmonic; (d) peak-to-peak and duration,
harmonic vs straight; (e) waveform correlation, end-of-fibre fraction and onset fraction;
(f) column waterfalls of the largest unit on both beds with the conduction velocity read
off the grid. Hollow markers: the harmonic unit re-drawn from full-span streamlines only.

Compute stages (cached) live in ``fibre_geometry_common.py``. Numbers →
``key_numbers_study_fibres.json``.

Run: /home/dc23/miniconda3/envs/fenicsx-env/bin/python paper/figures/make_fig_fibre_geometry.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import distance_transform_edt, map_coordinates

from style import use, save, letter, COL, W2
import f2_common as C
import fibre_geometry_common as G

use()
T0 = time.time()
KN = {}
BEDS = ("straight", "harmonic")
BCOL = {"straight": COL["poisson"], "harmonic": COL["harmonic"]}
REACH_CUT = 0.9            # a unit is "truncated" when < 90 % of its streamlines reach the IZ plane

# --------------------------------------------------------------------------- data
lf = C.ensure_grid_leadfields(); ten = C.ensure_muap_tensor()
hl = G.ensure_harmonic_leadfields()
S = G.ensure_study_muaps(n_workers=2)
FS_ = G.ensure_fullspan_muaps(n_workers=2)
fm = C.load_fibre_model(); pbed = C.poisson_bed(fm); pool = C.henneman_pool(pbed)
hbed, frame = G.harmonic_bed_iz(fm, pbed)
units = [int(u) for u in S["units"]]; U = len(units)
W = S["W"]; t_ms = S["t_ms"]; dt = float(t_ms[1] - t_ms[0])
n_elec = S["n_elec"]; elecs = S["elecs"]; column = S["column"]; hidx = S["hidx"]; hrad = S["hrad"]
Wfs = FS_["W"]; changed = [int(k) for k in FS_["changed"]]; hidx_fs = FS_["hidx"]; hrad_fs = FS_["hrad"]
sizes = np.array([pool[k].size for k in units])
e0 = lf["elec_xyz"][C.M // 2, C.M // 2]
cxy = np.array([pool[k].centre_xy for k in units]); depth = np.linalg.norm(cxy - e0[:2], axis=1)
prad = np.array([pool[k].territory_radius_mm for k in units])
z_iz = float(hbed.z_iz_plane); ied_z = float(lf["ied_along_mm"])
reach_u = np.array([hbed.reaches_iz[hidx[u]].mean() for u in range(U)])
trunc = reach_u < REACH_CUT; full = ~trunc
KN["design"] = dict(
    n_units=U, units=units, forced_units=G.FORCED, strata=G.STRATA, sizes=sizes.tolist(), depth_mm=depth.tolist(),
    electrodes="the grid column holding the largest |MUAP| of the unit in the released straight tensor (5 rows) + the centre electrode",
    column_per_unit=column.tolist(), n_electrodes_per_unit=n_elec.tolist(),
    straight_bed=dict(n_fibres=int(len(pbed.r_norms)), method="poisson", density=C.DENSITY),
    harmonic_bed=dict(n_fibres=int(len(hbed.r_norms)), seed_grid_mm=C.HARM_GRID_MM, single_nmj=True,
                      iz_plane_z_mm=z_iz, iz_fraction_frame=float(hbed.iz_frame),
                      n_streamlines_reaching_iz=int(hbed.reaches_iz.sum()),
                      frac_streamlines_reaching_iz=float(hbed.reaches_iz.mean()),
                      junction_rule="long_fibers placement: NMJ at the streamline sample nearest the IZ plane; a streamline that does not reach the plane gets its NMJ at its nearer end (f2_common.harmonic_bed_at_iz)"),
    matched_unit_rule="the unit's `size` nearest streamlines to the released centre in the mid-z plane (z = %.1f mm) — the pool's own territory rule" % float(S["z_plane"]),
    fullspan_variant="the same rule restricted to the %d streamlines that reach the IZ plane; re-synthesized for the units whose fibre set changes: %s" % (int(FS_["n_reaching"]), changed),
    innervation="NMJ on the straight bed's IZ plane (z = %.1f mm) on both beds; jitter N(0, %.2f mm) = IZ_JITTER × mean straight fibre length, per-MU seed = MU index" % (z_iz, float(S["sigma_jitter_mm"])),
    recipe="direct line-source synthesis: field_to_muap with f2_common.SPCFG (%s), φ sampled along each fibre path with per-fibre arc-length dz" % C.spcfg_dict(),
    lead_fields=dict(straight="cached phigrid_f2 (%d electrodes)" % (C.M * C.M), harmonic=str(G.HARM_LF.name),
                     harmonic_solve_s=float(hl["solve_s"]), harmonic_sample_s=float(hl["sample_s"]),
                     centre_electrode_straight_phi_rel_maxdiff=float(hl["straight_phi_rel_maxdiff"])),
    synthesis_cpu_s=float(S["secs"].sum()), synthesis_s_per_unit_bed=np.round(S["secs"], 1).tolist(),
    fullspan_cpu_s=float(FS_["secs"].sum()),
    truncated_group=dict(rule=f"< {REACH_CUT:.0%} of the unit's streamlines reach the IZ plane", units=[units[u] for u in np.where(trunc)[0]]),
)

# --------------------------------------------------------------------------- consistency: straight = released tensor
Wt = ten["W"]
chk = np.array([np.max(np.abs(W[u, 0, n] - Wt[k, e])) for u, k in enumerate(units) for n, e in enumerate(elecs[u][:n_elec[u]])])
KN["check_straight_equals_released_tensor"] = dict(max_abs_diff_V=float(chk.max()), identical=bool(chk.max() == 0.0))
print(f"straight recompute vs released tensor: max |Δ| = {chk.max():.3g} V (identical: {chk.max() == 0.0})")

# --------------------------------------------------------------------------- per-unit metrics at the centre electrode
ie0 = [list(elecs[u][:n_elec[u]]).index(C.E0) for u in range(U)]          # position of the centre electrode
Wc = np.array([[W[u, b, ie0[u]] for b in range(2)] for u in range(U)])     # (U, 2, w) V
Wc_fs = np.array([Wfs[u, ie0[u]] for u in range(U)])


def duration_ms(m):
    above = np.abs(m) > 0.1 * np.ptp(m)
    first = int(np.argmax(above)); last = len(above) - 1 - int(np.argmax(above[::-1]))
    return (last - first) * dt


def _sub_lag(a, b):
    """Lag of b vs a (samples, sub-sample parabolic), the estimator of Fig 8 / simulator_sanity."""
    cc = np.correlate(b, a, "full"); k = int(np.argmax(cc)); delta = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]; den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return (k + delta) - (len(a) - 1)


def fit_column(colw):
    """Lags of rows 1..4 vs row 0 → straight line → CV = IED / slope (Fig 8's read-off)."""
    if np.ptp(colw) == 0:
        return None
    lags = np.array([0.0] + [_sub_lag(colw[0], colw[i]) for i in range(1, C.M)])
    A = np.vstack([np.arange(C.M), np.ones(C.M)]).T
    sol = np.linalg.lstsq(A, lags, rcond=None)[0]; slope, icpt = sol; fit = A @ sol
    r2 = 1 - np.sum((lags - fit) ** 2) / max(np.sum((lags - lags.mean()) ** 2), 1e-9)
    if abs(slope) < 1e-6:
        return None
    return dict(cv=ied_z / (abs(slope) * dt), r2=float(r2), slope_ms=slope * dt, icpt_ms=icpt * dt, lags_ms=lags * dt)


# end-of-fibre / onset fractions: the propagating lobe reaches the electrode plane at
# |z_e − z_IZ| / v; lobe = max |m| within ±LOBE ms of that; EOF = max |m| later than
# t_arr + LOBE; onset = max |m| in [−1, 3] ms (a fibre whose NMJ sits at its end fires a
# terminal-like potential at t ≈ 0 on every electrode at once)
LOBE = 4.0
t_arr = abs(float(e0[2]) - z_iz) / C.CV
in_lobe = (t_ms >= t_arr - LOBE) & (t_ms <= t_arr + LOBE); after = t_ms > t_arr + LOBE
ONSET = (t_ms >= -1.0) & (t_ms <= 3.0)
KN["eof_definition"] = dict(t_arrival_ms=t_arr, lobe_half_width_ms=LOBE,
                            rule="EOF: max|m| for t > t_arr + LOBE over max|m| within ±LOBE of t_arr, centre electrode; "
                                 "onset: max|m| for −1 ≤ t ≤ 3 ms over the same lobe amplitude")


def metrics(m_str, m_harm, Wcol):
    """All per-unit numbers for one straight/harmonic pair (centre electrode + column)."""
    lobe = np.array([np.abs(m[in_lobe]).max() for m in (m_str, m_harm)])
    d = dict(p2p=np.ptp([m_str, m_harm], axis=1) * 1e6,
             dur=np.array([duration_ms(m_str), duration_ms(m_harm)]),
             r0=float(np.corrcoef(m_str, m_harm)[0, 1]),
             rmax=float(max(np.corrcoef(m_str, np.roll(m_harm, -s))[0, 1] for s in range(-20, 21))),
             lag=float(_sub_lag(m_str, m_harm) * dt),
             eof=np.array([np.abs(m[after]).max() for m in (m_str, m_harm)]) / lobe,
             onset=np.array([np.abs(m[ONSET]).max() for m in (m_str, m_harm)]) / lobe)
    fits = [fit_column(w) for w in Wcol]
    d["cvfit"] = fits
    d["cv"] = np.array([f["cv"] if f else np.nan for f in fits]); d["cv_r2"] = np.array([f["r2"] if f else np.nan for f in fits])
    d["p2p_col"] = np.ptp(Wcol, axis=2) * 1e6
    return d


MET = [metrics(Wc[u, 0], Wc[u, 1], W[u, :, :C.M]) for u in range(U)]
MET_FS = [metrics(Wc[u, 0], Wc_fs[u], np.stack([W[u, 0, :C.M], Wfs[u, :C.M]])) for u in range(U)]
p2p = np.array([m["p2p"] for m in MET]); dur = np.array([m["dur"] for m in MET]); r0 = np.array([m["r0"] for m in MET])
rmax = np.array([m["rmax"] for m in MET]); lag_ms = np.array([m["lag"] for m in MET])
eof = np.array([m["eof"] for m in MET]); onset = np.array([m["onset"] for m in MET])
cv = np.array([m["cv"] for m in MET]); cv_r2 = np.array([m["cv_r2"] for m in MET]); p2p_col = np.array([m["p2p_col"] for m in MET])
p2p_fs = np.array([m["p2p"][1] for m in MET_FS]); dur_fs = np.array([m["dur"][1] for m in MET_FS]); r0_fs = np.array([m["r0"] for m in MET_FS])
eof_fs = np.array([m["eof"][1] for m in MET_FS]); onset_fs = np.array([m["onset"][1] for m in MET_FS])
cv_fs = np.array([m["cv"][1] for m in MET_FS]); cv_r2_fs = np.array([m["cv_r2"][1] for m in MET_FS])
ratio = p2p[:, 1] / p2p[:, 0]; ratio_fs = p2p_fs / p2p[:, 0]

# --------------------------------------------------------------------------- fibre-level facts
Dskin = distance_transform_edt(fm.seg_data > 0, sampling=fm.voxel_size)     # depth below the skin, mm


def depth_along(path):
    p = np.asarray(path); c = (p / fm.voxel_size).T
    return map_coordinates(Dskin, c, order=1, mode="nearest")


arc_p, L_p = C.bed_arc_geometry(pbed)
L_h = np.array([float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum()) for p in hbed.paths])
z_start_h = np.array([float(np.asarray(p)[:, 2].min()) for p in hbed.paths])
_, cont_p = C.mask_containment(pbed.paths, fm); _, cont_h = C.mask_containment(hbed.paths, fm)
dep_p = [depth_along(p) for p in pbed.paths]; dep_h = [depth_along(p) for p in hbed.paths]
hxy, _ = G.xy_at_plane(hbed.paths, float(S["z_plane"]))
nr = ~hbed.reaches_iz
fib = {}
for u, k in enumerate(units):
    ip, ih, ifs = pool[k].fiber_idxs, hidx[u], hidx_fs[u]
    fib[k] = dict(
        straight=dict(length_mm=dict(mean=L_p[ip].mean(), min=L_p[ip].min(), max=L_p[ip].max()),
                      depth_below_skin_mm=dict(mean=float(np.mean([dep_p[i].mean() for i in ip])),
                                               along_fibre_ptp_mean=float(np.mean([np.ptp(dep_p[i]) for i in ip]))),
                      containment=float(cont_p[ip].mean()), containment_min=float(cont_p[ip].min()), territory_radius_mm=float(prad[u])),
        harmonic=dict(length_mm=dict(mean=L_h[ih].mean(), min=L_h[ih].min(), max=L_h[ih].max()),
                      depth_below_skin_mm=dict(mean=float(np.mean([dep_h[i].mean() for i in ih])),
                                               along_fibre_ptp_mean=float(np.mean([np.ptp(dep_h[i]) for i in ih]))),
                      containment=float(cont_h[ih].mean()), territory_radius_mm=float(hrad[u]),
                      frac_reaching_iz=float(reach_u[u]),
                      half1_mean_mm=float(np.mean(S["h_len1"][u])), half2_mean_mm=float(np.mean(S["h_len2"][u]))),
        harmonic_fullspan=dict(length_mm=dict(mean=L_h[ifs].mean(), min=L_h[ifs].min()), territory_radius_mm=float(hrad_fs[u]),
                               changed=bool(k in changed)))
Lh_u = np.array([fib[k]["harmonic"]["length_mm"]["mean"] for k in units])
dsk = np.array([[fib[k][b]["depth_below_skin_mm"]["mean"] for b in BEDS] for k in units])
dptp = np.array([[fib[k][b]["depth_below_skin_mm"]["along_fibre_ptp_mean"] for b in BEDS] for k in units])
KN["fibre_level"] = dict(
    bed=dict(straight=dict(length_mm=dict(mean=L_p.mean(), min=L_p.min(), max=L_p.max()), containment=float(cont_p.mean()),
                           frac_fibres_containment_below_0p9=float((cont_p < 0.9).mean()),
                           depth_below_skin_mm=dict(mean=float(np.mean([d.mean() for d in dep_p])),
                                                    along_fibre_ptp_mean=float(np.mean([np.ptp(d) for d in dep_p])),
                                                    along_fibre_ptp_max=float(np.max([np.ptp(d) for d in dep_p])))),
             harmonic=dict(length_mm=dict(mean=L_h.mean(), median=float(np.median(L_h)), min=L_h.min(), max=L_h.max(),
                                          frac_below_120=float((L_h < 120).mean())),
                           containment=float(cont_h.mean()),
                           depth_below_skin_mm=dict(mean=float(np.mean([d.mean() for d in dep_h])),
                                                    along_fibre_ptp_mean=float(np.mean([np.ptp(d) for d in dep_h])),
                                                    along_fibre_ptp_max=float(np.max([np.ptp(d) for d in dep_h]))),
                           truncated_streamlines=dict(n=int(nr.sum()), proximal_end_z_mm=dict(min=float(z_start_h[nr].min()), median=float(np.median(z_start_h[nr])), max=float(z_start_h[nr].max())),
                                                      reaching_proximal_end_z_median=float(np.median(z_start_h[~nr])),
                                                      depth_from_centre_electrode_mm_median=float(np.median(np.linalg.norm(hxy[nr] - e0[:2], axis=1))),
                                                      reaching_depth_median=float(np.median(np.linalg.norm(hxy[~nr] - e0[:2], axis=1))),
                                                      x_mean=float(hxy[nr, 0].mean()), reaching_x_mean=float(hxy[~nr, 0].mean()),
                                                      note="all start distal to the IZ plane on the deep (large-x) side of the muscle; their NMJ is placed at that proximal end"))),
    per_unit={str(k): fib[k] for k in units},
    territory_radius_ratio_harm_over_straight=dict(mean=float(np.mean(hrad / prad)), min=float(np.min(hrad / prad)), max=float(np.max(hrad / prad))),
)

# --------------------------------------------------------------------------- summary numbers
rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])
stat = lambda x: dict(median=float(np.median(x)), min=float(np.min(x)), max=float(np.max(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))
okcv = lambda c, r2, sel: c[sel & (r2 > 0.9)]
KN["per_unit"] = {str(k): dict(size=int(sizes[u]), depth_mm=float(depth[u]), column=int(column[u]), group="truncated" if trunc[u] else "full-span",
                               p2p_uV=dict(straight=float(p2p[u, 0]), harmonic=float(p2p[u, 1]), ratio=float(ratio[u]), harmonic_fullspan=float(p2p_fs[u]), ratio_fullspan=float(ratio_fs[u])),
                               duration_ms=dict(straight=float(dur[u, 0]), harmonic=float(dur[u, 1]), harmonic_fullspan=float(dur_fs[u])),
                               corr_zero_lag=float(r0[u]), corr_max=float(rmax[u]), lag_harm_minus_straight_ms=float(lag_ms[u]), corr_zero_lag_fullspan=float(r0_fs[u]),
                               eof_fraction=dict(straight=float(eof[u, 0]), harmonic=float(eof[u, 1]), harmonic_fullspan=float(eof_fs[u])),
                               onset_fraction=dict(straight=float(onset[u, 0]), harmonic=float(onset[u, 1]), harmonic_fullspan=float(onset_fs[u])),
                               harmonic_frac_reaching_iz=float(reach_u[u]), harmonic_mean_length_mm=float(Lh_u[u]),
                               cv_m_per_s=dict(straight=float(cv[u, 0]), harmonic=float(cv[u, 1]), harmonic_fullspan=float(cv_fs[u])),
                               cv_r2=dict(straight=float(cv_r2[u, 0]), harmonic=float(cv_r2[u, 1]), harmonic_fullspan=float(cv_r2_fs[u])),
                               p2p_uV_per_row=dict(straight=p2p_col[u, 0].tolist(), harmonic=p2p_col[u, 1].tolist()))
                  for u, k in enumerate(units)}
KN["summary"] = dict(
    groups=dict(full_span=dict(n=int(full.sum()), units=[units[u] for u in np.where(full)[0]]),
                truncated=dict(n=int(trunc.sum()), units=[units[u] for u in np.where(trunc)[0]], frac_reaching_iz=reach_u[trunc].tolist())),
    p2p_ratio_harm_over_straight=dict(all=stat(ratio), full_span=stat(ratio[full]), truncated=stat(ratio[trunc]),
                                      fullspan_variant_all=stat(ratio_fs), fullspan_variant_truncated_units=stat(ratio_fs[trunc]),
                                      corr_with_depth=rho(depth, ratio), corr_with_log_size=rho(np.log(sizes), ratio),
                                      corr_with_frac_reaching_iz=rho(reach_u, ratio),
                                      full_span_corr_with_depth=rho(depth[full], ratio[full]), full_span_corr_with_log_size=rho(np.log(sizes[full]), ratio[full]),
                                      full_span_corr_with_harm_length=rho(Lh_u[full], ratio[full])),
    log_p2p_harm_vs_straight_full_span=dict(slope=float(np.polyfit(np.log10(p2p[full, 0]), np.log10(p2p[full, 1]), 1)[0]),
                                            r=rho(np.log10(p2p[full, 0]), np.log10(p2p[full, 1]))),
    duration_ms=dict(straight=stat(dur[:, 0]), harmonic=stat(dur[:, 1]), diff_full_span=stat(dur[full, 1] - dur[full, 0]),
                     diff_fullspan_variant=stat(dur_fs - dur[:, 0]),
                     full_span_diff_corr_with_depth=rho(depth[full], dur[full, 1] - dur[full, 0])),
    corr_zero_lag=dict(all=stat(r0), full_span=stat(r0[full]), truncated=stat(r0[trunc]), fullspan_variant_all=stat(r0_fs),
                       fullspan_variant_truncated_units=stat(r0_fs[trunc]),
                       n_below_0p9=int((r0 < 0.9).sum()), n_below_0p9_fullspan_variant=int((r0_fs < 0.9).sum()),
                       corr_with_depth=rho(depth, r0), corr_with_frac_reaching_iz=rho(reach_u, r0)),
    corr_max=dict(all=stat(rmax)), lag_ms=dict(full_span=stat(lag_ms[full])),
    eof_fraction=dict(straight=stat(eof[:, 0]), harmonic=stat(eof[:, 1]), ratio_full_span=stat(eof[full, 1] / eof[full, 0]),
                      ratio_fullspan_variant=stat(eof_fs / eof[:, 0])),
    onset_fraction=dict(straight=stat(onset[:, 0]), harmonic_full_span=stat(onset[full, 1]), harmonic_truncated=stat(onset[trunc, 1]),
                        harmonic_fullspan_variant=stat(onset_fs), n_harmonic_above_0p5=int((onset[:, 1] > 0.5).sum()),
                        n_fullspan_variant_above_0p5=int((onset_fs > 0.5).sum()),
                        corr_with_frac_reaching_iz=rho(reach_u, np.log(onset[:, 1]))),
    cv_m_per_s=dict(set=C.CV, r2_filter=0.9,
                    straight=stat(okcv(cv[:, 0], cv_r2[:, 0], np.ones(U, bool))), straight_n=int((cv_r2[:, 0] > 0.9).sum()),
                    harmonic_full_span=stat(okcv(cv[:, 1], cv_r2[:, 1], full)), harmonic_full_span_n=int((full & (cv_r2[:, 1] > 0.9)).sum()),
                    harmonic_truncated_n_r2_above_0p9=int((trunc & (cv_r2[:, 1] > 0.9)).sum()),
                    harmonic_truncated_values=cv[trunc, 1].tolist(),
                    harmonic_fullspan_variant=stat(okcv(cv_fs, cv_r2_fs, np.ones(U, bool))), harmonic_fullspan_variant_n=int((cv_r2_fs > 0.9).sum()),
                    harm_over_straight_full_span=stat(cv[full, 1] / cv[full, 0])),
    depth_below_skin_mm=dict(unit_mean_straight_median=float(np.median(dsk[:, 0])), unit_mean_harmonic_median=float(np.median(dsk[:, 1])),
                             along_fibre_ptp_straight_median=float(np.median(dptp[:, 0])), along_fibre_ptp_harmonic_median=float(np.median(dptp[:, 1]))),
    harmonic_unit_length_mm=dict(full_span=stat(Lh_u[full]), truncated=stat(Lh_u[trunc])),
    frac_reaching_iz=stat(reach_u),
)
for u, k in enumerate(units):
    print(f"MU {k:3d} n={sizes[u]:3d} d={depth[u]:4.1f} {'T' if trunc[u] else ' '} p2p {p2p[u,0]:7.2f}/{p2p[u,1]:7.2f} µV (×{ratio[u]:.2f}; full-span ×{ratio_fs[u]:.2f})  "
          f"dur {dur[u,0]:.1f}/{dur[u,1]:.1f}  r0 {r0[u]:.3f}/{r0_fs[u]:.3f}  EOF {eof[u,0]:.2f}/{eof[u,1]:.2f}  onset {onset[u,0]:.2f}/{onset[u,1]:.2f}/{onset_fs[u]:.2f}  "
          f"CV {cv[u,0]:.2f}/{cv[u,1]:.2f}/{cv_fs[u]:.2f}  L_h {Lh_u[u]:.0f} reach {reach_u[u]:.2f}  R {prad[u]:.1f}/{hrad[u]:.1f}/{hrad_fs[u]:.1f}")
print("summary:", json.dumps({k: KN["summary"][k] for k in ("p2p_ratio_harm_over_straight", "corr_zero_lag", "eof_fraction", "onset_fraction", "cv_m_per_s", "duration_ms")}, indent=1, default=float))

# --------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W2, 8.8))
gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.05, 1.0],
                      left=0.075, right=0.95, top=0.97, bottom=0.05, wspace=0.3, hspace=0.45)
PK = {"S": 6, "M": 58, "L": 95, "XL": 99}
PKCOL = {"S": "#e6ab02", "M": "#e7298a", "L": "#d95f02", "XL": "#1f5fbf"}
pu = {lab: units.index(k) for lab, k in PK.items()}
szs = 6 + 10 * (np.log(sizes) - np.log(sizes.min())) / (np.log(sizes.max()) - np.log(sizes.min()))
MS = szs ** 1.6

# (a) side views -------------------------------------------------------------------
gsa = gs[0, 0].subgridspec(2, 1, hspace=0.12)
LAT = 0                                                    # Fig 4's side-view axis (x)
fcu_vox = np.argwhere(fm.seg_data == C.FCU); vs = fm.voxel_size
zs = np.unique(fcu_vox[:, 2]); z_sil = zs * vs[2]
sil_lo = np.array([fcu_vox[fcu_vox[:, 2] == k, LAT].min() for k in zs]) * vs[LAT]
sil_hi = np.array([fcu_vox[fcu_vox[:, 2] == k, LAT].max() for k in zs]) * vs[LAT]
for r, (name, bed) in enumerate((("straight", pbed), ("harmonic", hbed))):
    ax = fig.add_subplot(gsa[r])
    paths = [np.asarray(p) for p in bed.paths]
    ax.fill_between(z_sil, sil_lo, sil_hi, color="0.93", lw=0, step="mid")
    ax.add_collection(LineCollection([p[:, [2, LAT]] for p in paths[::8]], colors="0.72", linewidths=0.3, alpha=0.7))
    for lab in ("XL", "L", "M", "S"):
        k = PK[lab]; idx = pool[k].fiber_idxs if name == "straight" else hidx[pu[lab]]
        sub = idx[::max(1, len(idx) // 10)]
        ax.add_collection(LineCollection([paths[i][:, [2, LAT]] for i in sub], colors=PKCOL[lab], linewidths=0.45, alpha=0.9, zorder=3))
    ax.axvline(z_iz, color="k", lw=0.5, ls=":", zorder=4)
    ax.axvline(e0[2], color="0.3", lw=0.5, ls="--", zorder=4)
    ax.set_xlim(z_sil.min() - 4, z_sil.max() + 4); ax.autoscale_view(scalex=False); ax.set_aspect("equal")
    ax.set_ylabel("x (mm)", labelpad=1)
    ax.text(0.01, 0.97, f"{name} · {len(paths)} fibres", transform=ax.transAxes, fontsize=6.5, va="top", color=BCOL[name], fontweight="bold")
    if r == 0:
        ax.set_xticklabels([])
        ax.text(z_iz + 1.5, ax.get_ylim()[1] - 1, "IZ", fontsize=5.5, va="top", ha="left")
        ax.text(e0[2] + 1.5, ax.get_ylim()[1] - 1, "grid centre", fontsize=5.5, va="top", ha="left", color="0.3")
        for n, lab in enumerate(PK):
            ax.text(0.99 - 0.075 * (3 - n), 0.04, lab, transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
                    color=PKCOL[lab], fontweight="bold")
        letter(ax, "a", dx=-0.1)
    else:
        ax.set_xlabel("z along the arm (mm)")
        ax.text(0.99, 0.04, f"{int(nr.sum())} truncated streamlines (start distal to the IZ)", transform=ax.transAxes,
                fontsize=5.5, ha="right", va="bottom", color="0.3")

# (b) fibre-level facts ------------------------------------------------------------
gsf = gs[0, 1].subgridspec(2, 1, hspace=0.6)
axl = fig.add_subplot(gsf[0])
bins = np.linspace(60, 210, 31)
axl.hist(L_p, bins=bins, color=BCOL["straight"], alpha=0.8, lw=0, label="straight (all fibres)")
axl.hist(L_h, bins=bins, color=BCOL["harmonic"], alpha=0.7, lw=0, label="harmonic (all fibres)")
axl.set_xlabel("fibre length (mm)"); axl.set_ylabel("# fibres"); axl.set_yscale("log")
axl.legend(fontsize=5.5, loc="upper left")
for lab in ("L", "XL"):
    axl.axvline(Lh_u[pu[lab]], color=PKCOL[lab], lw=0.8, ls="--")
    axl.text(Lh_u[pu[lab]] - 2, axl.get_ylim()[1] * 0.5, f"{lab} mean", fontsize=5.5, rotation=90, ha="right", va="top", color=PKCOL[lab])
letter(axl, "b", dx=-0.2)
axd = fig.add_subplot(gsf[1])
k = PK["L"]
for name, bed, idx, dep in (("straight", pbed, pool[k].fiber_idxs, dep_p), ("harmonic", hbed, hidx[pu["L"]], dep_h)):
    zg = np.arange(60, 275, 2.0)
    prof = np.full((len(idx), len(zg)), np.nan)
    for n, i in enumerate(idx):
        p = np.asarray(bed.paths[i]); ok = (zg >= p[:, 2].min()) & (zg <= p[:, 2].max())
        prof[n, ok] = np.interp(zg[ok], p[:, 2], dep[i])
    good = np.sum(~np.isnan(prof), 0) >= 3                # columns with ≥ 3 fibres present
    col = lambda f: np.array([f(prof[~np.isnan(prof[:, j]), j]) if good[j] else np.nan for j in range(len(zg))])
    m, lo, hi = col(np.mean), col(lambda x: np.percentile(x, 10)), col(lambda x: np.percentile(x, 90))
    axd.fill_between(zg, lo, hi, color=BCOL[name], alpha=0.2, lw=0)
    axd.plot(zg, m, color=BCOL[name], lw=1.0, label=name)
axd.axvline(z_iz, color="k", lw=0.5, ls=":"); axd.axvline(e0[2], color="0.3", lw=0.5, ls="--")
axd.set_xlabel("z along the arm (mm)"); axd.set_ylabel("depth below skin (mm)")
axd.text(0.02, 0.96, f"unit L (MU {k}, {sizes[pu['L']]} fibres): mean, 10–90 %", transform=axd.transAxes, fontsize=6, va="top")
axd.legend(fontsize=5.5, loc="lower right"); axd.set_xlim(60, 275)

# (c) S/M/L/XL MUAPs at the centre electrode ---------------------------------------
gsb = gs[1, 0].subgridspec(4, 1, hspace=0.15)
for r, lab in enumerate(PK):
    u = pu[lab]; ax = fig.add_subplot(gsb[r])
    ax.axhline(0, color="0.8", lw=0.4)
    for b, name in enumerate(BEDS):
        ax.plot(t_ms, Wc[u, b] * 1e6, color=BCOL[name], lw=0.9, label=name if r == 0 else None)
    if units[u] in changed:
        ax.plot(t_ms, Wc_fs[u] * 1e6, color=BCOL["harmonic"], lw=0.8, ls="--", label="harmonic, full-span\nstreamlines only")
    ym = np.abs(Wc[u]).max() * 1e6 * 1.2
    ax.set_ylim(-ym, ym); ax.set_xlim(-2, 45)
    ax.text(0.99, 0.95, f"{lab}: MU {units[u]}, {sizes[u]} fibres, d = {depth[u]:.0f} mm · p2p ×{ratio[u]:.2f}, r = {r0[u]:.2f}"
            + (f"\nfull-span streamlines only: ×{ratio_fs[u]:.2f}, r = {r0_fs[u]:.2f}" if units[u] in changed else ""),
            transform=ax.transAxes, fontsize=5.6, ha="right", va="top")
    if r < 3:
        ax.set_xticklabels([])
    if r == 0:
        ax.legend(fontsize=5.5, loc="lower right", ncol=2, handlelength=1.2); letter(ax, "c", dx=-0.1)
    if units[u] in changed:
        ax.legend(fontsize=5.5, loc="lower right", handlelength=1.6)
    if r == 1:
        ax.set_ylabel("MUAP at the centre electrode (µV)", labelpad=2); ax.yaxis.set_label_coords(-0.115, -0.1)
ax.set_xlabel("t (ms), 0 = NMJ firing")

# (d) p2p and duration, harmonic vs straight ----------------------------------------
gsc = gs[1, 1].subgridspec(1, 2, wspace=0.55)
axp = fig.add_subplot(gsc[0])
sc = axp.scatter(p2p[:, 0], p2p[:, 1], s=14, c=depth, cmap="magma_r", lw=0.3, edgecolor="k", zorder=3)
axp.scatter(p2p[trunc, 0], p2p_fs[trunc], s=16, facecolor="none", edgecolor=BCOL["harmonic"], lw=0.8, zorder=4)
for u in np.where(trunc)[0]:
    axp.plot([p2p[u, 0]] * 2, [p2p[u, 1], p2p_fs[u]], color=BCOL["harmonic"], lw=0.4, zorder=2)
lim = [p2p.min() * 0.7, p2p.max() * 1.4]
axp.plot(lim, lim, color="0.5", lw=0.6, ls="--"); axp.set_xscale("log"); axp.set_yscale("log")
axp.set_xlim(lim); axp.set_ylim(lim)
for lab in PK:
    u = pu[lab]; axp.annotate(lab, (p2p[u, 0], p2p[u, 1]), xytext=(3, 2), textcoords="offset points", fontsize=6, fontweight="bold", color=PKCOL[lab])
axp.set_xlabel("p2p, straight (µV)"); axp.set_ylabel("p2p, harmonic (µV)")
axp.text(0.97, 0.2, "○ full-span streamlines only", transform=axp.transAxes, fontsize=5.3, va="bottom", ha="right")
axp.text(0.97, 0.28, f"full-span units:\nratio {np.median(ratio[full]):.2f} ({ratio[full].min():.2f}–{ratio[full].max():.2f})",
         transform=axp.transAxes, fontsize=5.3, va="bottom", ha="right")
cax = axp.inset_axes([0.06, 0.9, 0.36, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("unit depth (mm)", fontsize=5.5, labelpad=1); cb.ax.tick_params(labelsize=5, length=1.5)
letter(axp, "d", dx=-0.3)
axq = fig.add_subplot(gsc[1])
axq.scatter(dur[:, 0], dur[:, 1], s=14, c=depth, cmap="magma_r", lw=0.3, edgecolor="k", zorder=3)
axq.scatter(dur[trunc, 0], dur_fs[trunc], s=16, facecolor="none", edgecolor=BCOL["harmonic"], lw=0.8, zorder=4)
dl = [0, dur.max() + 1.5]
axq.plot(dl, dl, color="0.5", lw=0.6, ls="--"); axq.set_xlim(dl); axq.set_ylim(dl)
for lab in PK:
    u = pu[lab]; axq.annotate(lab, (dur[u, 0], dur[u, 1]), xytext=(3, 2), textcoords="offset points", fontsize=6, fontweight="bold", color=PKCOL[lab])
axq.set_xlabel("duration, straight (ms)", loc="right"); axq.set_ylabel("duration, harmonic (ms)")
axq.text(0.03, 0.97, f"Δ median {np.median(dur[full,1]-dur[full,0]):+.1f} ms\n(full-span units)", transform=axq.transAxes, fontsize=5.5, va="top")
axq.set_xticks([0, 10, 20])

# (e) correlation, EOF fraction and onset fraction ------------------------------------
gsd = gs[2, 0].subgridspec(1, 3, wspace=0.55)
axr = fig.add_subplot(gsd[0])
scr = axr.scatter(depth, r0, s=MS, c=reach_u, cmap="viridis", vmin=min(0.5, reach_u.min()), vmax=1.0, lw=0.3, edgecolor="k", zorder=3)
axr.scatter(depth[trunc], r0_fs[trunc], s=MS[trunc], facecolor="none", edgecolor=BCOL["harmonic"], lw=0.8, zorder=4)
for u in np.where(trunc)[0]:
    axr.plot([depth[u]] * 2, [r0[u], r0_fs[u]], color=BCOL["harmonic"], lw=0.4, zorder=2)
for lab in PK:
    u = pu[lab]; axr.annotate(lab, (depth[u], r0[u]), xytext=(3, 2), textcoords="offset points", fontsize=6, fontweight="bold", color=PKCOL[lab])
axr.set_xlabel("unit depth (mm)"); axr.set_ylabel("r, straight vs harmonic (zero lag)")
axr.set_ylim(min(0.0, r0.min() - 0.05), 1.04)
axr.text(0.03, 0.04, f"full-span units:\nmedian r = {np.median(r0[full]):.2f}\n○ full-span\nstreamlines only", transform=axr.transAxes, fontsize=5.3, va="bottom")
cax = axr.inset_axes([0.08, 0.5, 0.34, 0.035])
cb = fig.colorbar(scr, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=5, length=1.5)
cb.set_label("reaching the IZ", fontsize=5.2, labelpad=1)
axr.text(0.08, 0.62, "size ∝ log fibres", transform=axr.transAxes, fontsize=5.2, va="bottom")
letter(axr, "e", dx=-0.4)
axe = fig.add_subplot(gsd[1])
for b, name in enumerate(BEDS):
    axe.scatter(depth, eof[:, b], s=MS, color=BCOL[name], lw=0.3, edgecolor="k", alpha=0.85, label=name, zorder=3)
for u in range(U):
    axe.plot([depth[u]] * 2, eof[u], color="0.6", lw=0.4, zorder=2)
axe.set_xlabel("unit depth (mm)"); axe.set_ylabel("end-of-fibre fraction")
axe.text(0.03, 0.97, "max |m| after the\npropagating lobe / lobe", transform=axe.transAxes, fontsize=5.3, va="top")
axe.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(0.0, 0.62), handletextpad=0.3); axe.set_ylim(0, max(eof.max() * 1.25, 0.5))
axo = fig.add_subplot(gsd[2])
for b, name in enumerate(BEDS):
    axo.scatter(reach_u, onset[:, b], s=MS, color=BCOL[name], lw=0.3, edgecolor="k", alpha=0.85, label=name, zorder=3)
axo.scatter(reach_u[trunc], onset_fs[trunc], s=MS[trunc], facecolor="none", edgecolor=BCOL["harmonic"], lw=0.8, zorder=4)
for lab in PK:
    u = pu[lab]; axo.annotate(lab, (reach_u[u], onset[u, 1]), xytext=(3, 2), textcoords="offset points", fontsize=6, fontweight="bold", color=PKCOL[lab])
axo.set_yscale("log"); axo.set_ylim(0.04, max(onset.max() * 2.0, 1.0)); axo.set_xlim(min(0.5, reach_u.min() - 0.05), 1.03)
axo.axhline(1.0, color="0.7", lw=0.4, ls=":")
axo.set_xlabel("streamlines reaching\nthe IZ plane (fraction)"); axo.set_ylabel("onset fraction", labelpad=1)
axo.text(0.03, 0.97, "max |m| at −1…3 ms / lobe:\nNMJ at a truncated\nstreamline's end →\nnon-propagating\nonset potential", transform=axo.transAxes, fontsize=5.3, va="top")

# (f) column waterfalls of the XL unit on both beds, with the CV read-off ---------------
gse = gs[2, 1].subgridspec(1, 2, wspace=0.12)
u = pu["XL"]; T0W, T1W = 0.0, 30.0; wsel = (t_ms >= T0W) & (t_ms <= T1W)
amp_w = np.abs(W[u, :, :C.M][:, :, wsel]).max() * 1e6
scale = 0.42 * ied_z / amp_w
for b, name in enumerate(BEDS):
    ax = fig.add_subplot(gse[b]); colw = W[u, b, :C.M] * 1e6; fw = MET[u]["cvfit"][b]
    for i in range(C.M):
        ax.plot(t_ms[wsel], i * ied_z + colw[i, wsel] * scale, lw=0.7, color=BCOL[name], zorder=3)
    if fw is not None:
        i_pk = int(np.argmax(np.abs(colw[0, wsel]))); t_pk0 = float(t_ms[wsel][i_pk])
        zz = np.array([-0.45, C.M - 1 + 0.45]) * ied_z
        ax.plot(t_pk0 + fw["icpt_ms"] + fw["slope_ms"] * zz / ied_z, zz, ls="--", lw=0.7, color="k", zorder=2)
        ax.text(0.98, 0.985, f"CV {fw['cv']:.2f} m/s\nR² {fw['r2']:.3f}", transform=ax.transAxes, fontsize=6, ha="right", va="top")
    ax.set_xlim(T0W, T1W); ax.set_ylim(zz[1] + 0.05 * ied_z, zz[0] - 0.05 * ied_z)
    ax.set_yticks(np.arange(C.M) * ied_z)
    if b == 0:
        ax.set_yticklabels([f"{i * ied_z:.0f}" for i in range(C.M)]); ax.set_ylabel("z along the arm (mm)", labelpad=1)
        letter(ax, "f", dx=-0.3)
        ax.text(0.0, 1.06, f"MU {units[u]} ({sizes[u]} fibres), column {column[u]}", transform=ax.transAxes, fontsize=6, va="bottom")
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel("t (ms), 0 = NMJ firing", fontsize=6.5, labelpad=2)
    ax.text(0.02, 0.985, name, transform=ax.transAxes, fontsize=6.5, va="top", color=BCOL[name], fontweight="bold")
sb_uv = 50.0 if amp_w > 80 else 20.0
y_sb = (C.M - 1) * ied_z + 0.2 * ied_z
ax.plot([T1W - 1.5] * 2, [y_sb, y_sb + sb_uv * scale], color="k", lw=1.0)
ax.text(T1W - 2.3, y_sb + 0.5 * sb_uv * scale, f"{sb_uv:.0f} µV", fontsize=5, ha="right", va="center")

save(fig, "fig_fibre_geometry")
KN["total_s"] = time.time() - T0


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


with open(G.KEY_JSON, "w") as fh:
    json.dump(clean(KN), fh, indent=2)
print("key numbers →", G.KEY_JSON)
