from typing import Dict, Iterable, List, Tuple

import numpy as np

from .lhs import latin_hypercube


def _ellipse_support_radius(a: float, b: float, theta: float) -> float:
    """
    Support function (radius) of axis-aligned ellipse (x/a)^2 + (y/b)^2 <= 1
    in direction theta. This gives the maximum distance from center along that ray.
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    denom = (c / a) ** 2 + (s / b) ** 2
    return float(1.0 / np.sqrt(denom))


def sample_parameters(
    n: int,
    seed: int,
    radius_multiplicative_factor: float,
    length_ratio_range: Tuple[float, float] = (4.0, 8.0),
    ranges_normalized: Dict[str, Tuple[float, float]] | None = None,
    shape: str = "circle",
    ellipse_ratio_range: Tuple[float, float] = (1.0, 1.0),
    bone_offset_frac_range: Tuple[float, float] = (0.0, 0.0),
    bone_count: int = 1,
    interbone_frac_range: Tuple[float, float] = (0.0, 0.0),
    bone2_scale_range: Tuple[float, float] = (1.0, 1.0),
    bone_angle_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
    z_profile: str = "constant",
    taper_scale_range: Tuple[float, float] = (1.0, 1.0),
) -> List[Dict[str, float]]:
    """
    Sample geometry parameters in "normalized" units, then scale by
    radius_multiplicative_factor. Returns list of dicts with final
    radii/thicknesses/length.
    """
    if ranges_normalized is None:
        ranges_normalized = {
            "radius_canc_bone": (0.12, 0.25),
            "thickness_cort_bone": (0.03, 0.11),
            "thickness_muscle": (0.35, 0.95),
            "thickness_fat": (0.03, 0.50),
            "thickness_skin": (0.01, 0.06),
        }

    keys: Iterable[str] = ranges_normalized.keys()
    # + length_ratio + ellipse_ratio + taper_scale
    # + bone_offset_x + bone_offset_y
    # + interbone_dist + bone2_scale + bone_angle
    d = len(ranges_normalized) + 8
    u = latin_hypercube(n, d, seed=seed)

    params: List[Dict[str, float]] = []
    for i in range(n):
        p: Dict[str, float] = {}
        for j, k in enumerate(keys):
            lo, hi = ranges_normalized[k]
            p[k] = lo + (hi - lo) * u[i, j]

        lr_lo, lr_hi = length_ratio_range
        length_ratio = lr_lo + (lr_hi - lr_lo) * u[i, -8]

        er_lo, er_hi = ellipse_ratio_range
        ellipse_ratio = er_lo + (er_hi - er_lo) * u[i, -7]

        ts_lo, ts_hi = taper_scale_range
        taper_scale_z1 = ts_lo + (ts_hi - ts_lo) * u[i, -6]

        off_lo, off_hi = bone_offset_frac_range
        # symmetric offsets in [-off_hi, +off_hi]
        max_off_frac = max(0.0, float(off_hi))
        offx_frac = (u[i, -5] - 0.5) * 2.0 * max_off_frac
        offy_frac = (u[i, -4] - 0.5) * 2.0 * max_off_frac

        ib_lo, ib_hi = interbone_frac_range
        interbone_frac = ib_lo + (ib_hi - ib_lo) * u[i, -3]
        s2_lo, s2_hi = bone2_scale_range
        bone2_scale = s2_lo + (s2_hi - s2_lo) * u[i, -2]
        a_lo, a_hi = bone_angle_range
        bone_angle = a_lo + (a_hi - a_lo) * u[i, -1]

        if shape not in ("circle", "ellipse"):
            raise ValueError("shape must be 'circle' or 'ellipse'")
        if shape == "circle":
            ellipse_ratio = 1.0

        if z_profile not in ("constant", "taper"):
            raise ValueError("z_profile must be 'constant' or 'taper'")
        if z_profile == "constant":
            taper_scale_z1 = 1.0

        if p["thickness_cort_bone"] < 0.02:
            p["thickness_cort_bone"] = 0.02
        if p["thickness_skin"] < 0.008:
            p["thickness_skin"] = 0.008

        for k in keys:
            p[k] *= radius_multiplicative_factor

        length = radius_multiplicative_factor * length_ratio

        r1 = p["radius_canc_bone"]
        r2 = r1 + p["thickness_cort_bone"]
        r3 = r2 + p["thickness_muscle"]
        r4 = r3 + p["thickness_fat"]
        r5 = r4 + p["thickness_skin"]

        p.update(
            {
                "shape": shape,
                "z_profile": z_profile,
                "radius_multiplicative_factor": radius_multiplicative_factor,
                "length_ratio": float(length_ratio),
                "ellipse_ratio": float(ellipse_ratio),
                "taper_scale_z0": 1.0,
                "taper_scale_z1": float(taper_scale_z1),
                "length": float(length),
                "radius_cort_bone": float(r2),
                "radius_muscle": float(r3),
                "radius_fat": float(r4),
                "radius_skin": float(r5),
            }
        )

        # Off-center bone (applies to cancellous+cortical as a unit for bone_count=1,
        # and to the whole two-bone system center for bone_count=2).
        # We interpret offsets in world units, as a fraction of outer size, then clamp so cortical stays inside muscle.
        dx = offx_frac * r5
        dy = offy_frac * r5
        margin = 0.5
        if shape == "ellipse":
            # Use ellipse axes for clamp (a along x, b along y)
            a_cort = float(ellipse_ratio * r2)
            b_cort = float(r2)
            a_musc = float(ellipse_ratio * r3)
            b_musc = float(r3)
            max_dx = max(0.0, (a_musc - a_cort) - margin)
            max_dy = max(0.0, (b_musc - b_cort) - margin)
        else:
            max_dx = max(0.0, (r3 - r2) - margin)
            max_dy = max(0.0, (r3 - r2) - margin)

        dx = float(np.clip(dx, -max_dx, max_dx))
        dy = float(np.clip(dy, -max_dy, max_dy))
        p.update({"bone_count": int(bone_count), "bone_offset_x": dx, "bone_offset_y": dy})

        if bone_count not in (1, 2):
            raise ValueError("bone_count must be 1 or 2")

        if bone_count == 2:
            # Two bones: define centers relative to global offset (dx,dy)
            # interbone distance is a fraction of muscle size
            if shape == "ellipse":
                a_musc = float(ellipse_ratio * r3)
                b_musc = float(r3)
                max_sep = 0.9 * min(a_musc, b_musc)
            else:
                max_sep = 0.9 * float(r3)

            sep = float(np.clip(interbone_frac * max_sep, 0.0, max_sep))

            # Bone2 base size (cancellous radius only; cortical thickness shared)
            # This must be defined before any non-intersection math uses r2_2.
            bone2_scale = float(np.clip(bone2_scale, 0.4, 1.4))
            r1_2 = float(r1 * bone2_scale)
            r2_2 = float(r1_2 + p["thickness_cort_bone"])
            # Ensure bones do not intersect: enforce sep >= min_sep based on cortical outer shapes
            safety = 1.05  # small buffer so bones are not just tangent
            margin = 0.5   # absolute extra clearance in world units
            if shape == "ellipse":
                # bone1 cortical axes
                a_cort1 = float(ellipse_ratio * r2)
                b_cort1 = float(r2)
                # bone2 cortical axes (depends on scale)
                a_cort2 = float(ellipse_ratio * r2_2)
                b_cort2 = float(r2_2)
                rdir1 = _ellipse_support_radius(a_cort1, b_cort1, bone_angle)
                rdir2 = _ellipse_support_radius(a_cort2, b_cort2, bone_angle)
                min_sep = safety * (rdir1 + rdir2) + margin
            else:
                min_sep = safety * (r2 + r2_2) + margin

            if min_sep > max_sep:
                # Try shrinking bone2 to fit. Solve for max feasible cortical size for bone2.
                if shape == "ellipse":
                    # approximate using direction radius along bone_angle: rdir2 <= (max_sep - margin)/safety - rdir1
                    rhs = (max_sep - margin) / safety - rdir1
                    if rhs <= 0:
                        raise ValueError(
                            "Two-bone geometry cannot fit: bone1 cortical already too large for muscle envelope. "
                            "Increase muscle thickness / decrease bone radii / increase grid factor."
                        )
                    # Convert rhs direction radius to a conservative scale on r2_2 (since a,b scale with r2_2)
                    # For axis-aligned ellipse with fixed aspect ratio, support radius scales linearly with r2_2.
                    scale_needed = rhs / rdir2 if rdir2 > 0 else 0.0
                else:
                    rhs = (max_sep - margin) / safety - r2
                    if rhs <= 0:
                        raise ValueError(
                            "Two-bone geometry cannot fit: bone1 cortical already too large for muscle envelope. "
                            "Increase muscle thickness / decrease bone radii / increase grid factor."
                        )
                    scale_needed = rhs / r2_2 if r2_2 > 0 else 0.0

                # Apply scale to bone2 cancellous radius (cortical follows from thickness)
                bone2_scale = float(np.clip(bone2_scale * scale_needed, 0.4, 1.4))
                r1_2 = float(r1 * bone2_scale)
                r2_2 = float(r1_2 + p["thickness_cort_bone"])
                p["bone2_scale"] = float(bone2_scale)
                p["radius_canc_bone_2"] = float(r1_2)
                p["radius_cort_bone_2"] = float(r2_2)

                # recompute min_sep with updated bone2 size
                if shape == "ellipse":
                    a_cort2 = float(ellipse_ratio * r2_2)
                    b_cort2 = float(r2_2)
                    rdir2 = _ellipse_support_radius(a_cort2, b_cort2, bone_angle)
                    min_sep = safety * (rdir1 + rdir2) + margin
                else:
                    min_sep = safety * (r2 + r2_2) + margin

            sep = float(np.clip(max(sep, min_sep), 0.0, max_sep))

            # Place bones at +/- sep/2 along angle
            ux = float(np.cos(bone_angle))
            uy = float(np.sin(bone_angle))
            x1 = dx + 0.5 * sep * ux
            y1 = dy + 0.5 * sep * uy
            x2 = dx - 0.5 * sep * ux
            y2 = dy - 0.5 * sep * uy

            # Persist final bone2 size (may have been shrunk above)
            p["bone2_scale"] = float(p.get("bone2_scale", bone2_scale))
            p["radius_canc_bone_2"] = float(p.get("radius_canc_bone_2", r1_2))
            p["radius_cort_bone_2"] = float(p.get("radius_cort_bone_2", r2_2))

            p.update(
                {
                    "interbone_frac": float(interbone_frac),
                    "interbone_sep": float(sep),
                    "bone_angle": float(bone_angle),
                    "bone2_scale": float(p["bone2_scale"]),
                    "bone1_center_x": float(x1),
                    "bone1_center_y": float(y1),
                    "bone2_center_x": float(x2),
                    "bone2_center_y": float(y2),
                    "radius_canc_bone_2": float(p["radius_canc_bone_2"]),
                    "radius_cort_bone_2": float(p["radius_cort_bone_2"]),
                }
            )

        # For ellipse: build similar ellipses for each layer by scaling from radii.
        # We interpret ellipse_ratio = a/b with b aligned to y and a aligned to x.
        if shape == "ellipse":
            p.update(
                {
                    "a_canc_bone": float(ellipse_ratio * r1),
                    "b_canc_bone": float(r1),
                    "a_cort_bone": float(ellipse_ratio * r2),
                    "b_cort_bone": float(r2),
                    "a_muscle": float(ellipse_ratio * r3),
                    "b_muscle": float(r3),
                    "a_fat": float(ellipse_ratio * r4),
                    "b_fat": float(r4),
                    "a_skin": float(ellipse_ratio * r5),
                    "b_skin": float(r5),
                }
            )

            if bone_count == 2:
                p.update(
                    {
                        "a_canc_bone_2": float(ellipse_ratio * p["radius_canc_bone_2"]),
                        "b_canc_bone_2": float(p["radius_canc_bone_2"]),
                        "a_cort_bone_2": float(ellipse_ratio * p["radius_cort_bone_2"]),
                        "b_cort_bone_2": float(p["radius_cort_bone_2"]),
                    }
                )

        params.append(p)

    return params


__all__ = ["sample_parameters"]

