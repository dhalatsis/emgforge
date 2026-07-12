from __future__ import annotations

import json
import time
from pathlib import Path

import gmsh

from .gmsh_helpers import find_volume_containing_point, assign_volumes_by_seed_points, assign_volumes_to_layers, compute_mesh_counts


def _extrude_disk_to_volume(a: float, b: float, length: float, cx: float = 0.0, cy: float = 0.0) -> int:
    """
    Create an elliptical disk (a along x, b along y) at z=0 and extrude to z=length.
    Returns volume tag.
    """
    surf = gmsh.model.occ.addDisk(cx, cy, 0, a, b)
    out = gmsh.model.occ.extrude([(2, surf)], 0, 0, length)
    # out contains lateral surfaces and one volume entity
    vols = [tag for dim, tag in out if dim == 3]
    if not vols:
        raise RuntimeError("Extrude did not produce a volume")
    return int(vols[0])


def _loft_disk_to_volume(
    a0: float,
    b0: float,
    z0: float,
    a1: float,
    b1: float,
    z1: float,
    cx: float = 0.0,
    cy: float = 0.0,
) -> int:
    """
    Loft between two elliptical disks (a,b) at z0 and z1 into a solid volume.
    Uses OCC addThruSections on wires extracted from disk boundaries.
    """
    s0 = gmsh.model.occ.addDisk(cx, cy, z0, a0, b0)
    s1 = gmsh.model.occ.addDisk(cx, cy, z1, a1, b1)
    gmsh.model.occ.synchronize()

    def wire_from_disk_surface(stag: int) -> int:
        bnd = gmsh.model.occ.getBoundary([(2, stag)], oriented=False)
        curves = [tag for dim, tag in bnd if dim == 1]
        if not curves:
            raise RuntimeError("Disk surface boundary has no curves")
        return gmsh.model.occ.addWire(curves)

    w0 = wire_from_disk_surface(s0)
    w1 = wire_from_disk_surface(s1)
    gmsh.model.occ.synchronize()

    vol = gmsh.model.occ.addThruSections([w0, w1], makeSolid=True)
    gmsh.model.occ.synchronize()
    return int(vol)


def _z_scale(p: dict, z: float, length: float) -> float:
    """
    Linear scale along z for taper mode.
    """
    prof = p.get("z_profile", "constant")
    if prof == "constant":
        return 1.0
    s0 = float(p.get("taper_scale_z0", 1.0))
    s1 = float(p.get("taper_scale_z1", 1.0))
    t = 0.0 if length == 0 else float(z / length)
    return float(s0 + (s1 - s0) * t)


def _choose_seed(p: dict, kind: str, z: float) -> tuple[float, float, float]:
    """
    Heuristic seed selection for stage2 off-center bone.
    kind in {"cancellous","cortical","muscle","fat","skin"}.
    """
    dx = float(p.get("bone_offset_x", 0.0))
    dy = float(p.get("bone_offset_y", 0.0))
    shape = p.get("shape", "circle")

    if shape == "circle":
        length = float(p["length"])
        s = _z_scale(p, z, length)
        r_canc = float(p["radius_canc_bone"]) * s
        r_cort = float(p["radius_cort_bone"]) * s
        r_musc = float(p["radius_muscle"]) * s
        r_fat = float(p["radius_fat"]) * s
        r_skin = float(p["radius_skin"]) * s

        if kind == "cancellous":
            return (dx, dy, z)
        if kind == "cortical":
            return (dx + 0.5 * (r_canc + r_cort), dy, z)
        if kind == "fat":
            return (0.5 * (r_musc + r_fat), 0.0, z)
        if kind == "skin":
            return (0.5 * (r_fat + r_skin), 0.0, z)
        # muscle: pick a point likely away from shifted bone
        candidates = [
            (-0.5 * r_musc, 0.0, z),
            (0.0, -0.5 * r_musc, z),
            (-0.25 * r_musc, -0.25 * r_musc, z),
        ]
        for cx, cy, cz in candidates:
            # inside muscle (origin-centered) and outside cortical (bone-centered)
            if (cx * cx + cy * cy) <= (0.9 * r_musc) ** 2 and ((cx - dx) ** 2 + (cy - dy) ** 2) >= (1.05 * r_cort) ** 2:
                return (cx, cy, cz)
        return candidates[0]

    # ellipse
    length = float(p["length"])
    s = _z_scale(p, z, length)
    a_canc = float(p["a_canc_bone"]) * s
    b_canc = float(p["b_canc_bone"]) * s
    a_cort = float(p["a_cort_bone"]) * s
    b_cort = float(p["b_cort_bone"]) * s
    a_musc = float(p["a_muscle"]) * s
    b_musc = float(p["b_muscle"]) * s
    a_fat = float(p["a_fat"]) * s
    b_fat = float(p["b_fat"]) * s
    a_skin = float(p["a_skin"]) * s
    b_skin = float(p["b_skin"]) * s

    if kind == "cancellous":
        return (dx, dy, z)
    if kind == "cortical":
        return (dx + 0.5 * (a_canc + a_cort), dy, z)
    if kind == "fat":
        return (0.5 * (a_musc + a_fat), 0.0, z)
    if kind == "skin":
        return (0.5 * (a_fat + a_skin), 0.0, z)
    # muscle
    candidates = [
        (-0.5 * a_musc, 0.0, z),
        (0.0, -0.5 * b_musc, z),
        (-0.25 * a_musc, -0.25 * b_musc, z),
    ]
    for cx, cy, cz in candidates:
        inside_m = (cx / a_musc) ** 2 + (cy / b_musc) ** 2 <= 0.9
        outside_c = ((cx - dx) / a_cort) ** 2 + ((cy - dy) / b_cort) ** 2 >= 1.05
        if inside_m and outside_c:
            return (cx, cy, cz)
    return candidates[0]


def build_one_mesh(
    p: dict,
    out_msh: Path,
    out_json: Path,
    mesh_char_length_factor: float = 0.2,
    size_min_factor: float = 0.1,
    size_max_factor: float = 0.2,
    dist_min_factor: float = 0.05,
    dist_max_factor: float = 0.10,
    refine_on: str = "skin",
    verbose: bool = False,
):
    """
    Build one layered (circular or elliptical) cylinder mesh and save .msh + metadata json.
    """
    canc_tag = 1
    cort_tag = 2
    muscle_tag = 3
    fat_tag = 4
    skin_tag = 5

    gmsh.clear()
    gmsh.model.add("concentric_cylinders")

    length = p["length"]
    z_profile = p.get("z_profile", "constant")

    shape = p.get("shape", "circle")
    if shape not in ("circle", "ellipse"):
        raise ValueError("geometry param 'shape' must be 'circle' or 'ellipse'")

    bone_count = int(p.get("bone_count", 1))
    if bone_count not in (1, 2):
        raise ValueError("geometry param 'bone_count' must be 1 or 2")

    if shape == "circle":
        r_canc = p["radius_canc_bone"]
        r_cort = p["radius_cort_bone"]
        r_musc = p["radius_muscle"]
        r_fat = p["radius_fat"]
        r_skin = p["radius_skin"]

        s0 = float(p.get("taper_scale_z0", 1.0))
        s1 = float(p.get("taper_scale_z1", 1.0))

        if bone_count == 1:
            dx = float(p.get("bone_offset_x", 0.0))
            dy = float(p.get("bone_offset_y", 0.0))
            if z_profile == "taper":
                canc = gmsh.model.occ.addCone(dx, dy, 0, 0, 0, length, r_canc * s0, r_canc * s1)
                cort = gmsh.model.occ.addCone(dx, dy, 0, 0, 0, length, r_cort * s0, r_cort * s1)
            else:
                canc = gmsh.model.occ.addCylinder(dx, dy, 0, 0, 0, length, r_canc)
                cort = gmsh.model.occ.addCylinder(dx, dy, 0, 0, 0, length, r_cort)
            canc2 = cort2 = None
        else:
            x1 = float(p["bone1_center_x"])
            y1 = float(p["bone1_center_y"])
            x2 = float(p["bone2_center_x"])
            y2 = float(p["bone2_center_y"])
            r_canc2 = float(p["radius_canc_bone_2"])
            r_cort2 = float(p["radius_cort_bone_2"])

            if z_profile == "taper":
                canc = gmsh.model.occ.addCone(x1, y1, 0, 0, 0, length, r_canc * s0, r_canc * s1)
                cort = gmsh.model.occ.addCone(x1, y1, 0, 0, 0, length, r_cort * s0, r_cort * s1)
                canc2 = gmsh.model.occ.addCone(x2, y2, 0, 0, 0, length, r_canc2 * s0, r_canc2 * s1)
                cort2 = gmsh.model.occ.addCone(x2, y2, 0, 0, 0, length, r_cort2 * s0, r_cort2 * s1)
            else:
                canc = gmsh.model.occ.addCylinder(x1, y1, 0, 0, 0, length, r_canc)
                cort = gmsh.model.occ.addCylinder(x1, y1, 0, 0, 0, length, r_cort)
                canc2 = gmsh.model.occ.addCylinder(x2, y2, 0, 0, 0, length, r_canc2)
                cort2 = gmsh.model.occ.addCylinder(x2, y2, 0, 0, 0, length, r_cort2)
        if z_profile == "taper":
            musc = gmsh.model.occ.addCone(0, 0, 0, 0, 0, length, r_musc * s0, r_musc * s1)
            fat = gmsh.model.occ.addCone(0, 0, 0, 0, 0, length, r_fat * s0, r_fat * s1)
            skin = gmsh.model.occ.addCone(0, 0, 0, 0, 0, length, r_skin * s0, r_skin * s1)
        else:
            musc = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, r_musc)
            fat = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, r_fat)
            skin = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, r_skin)
        target = {
            "cancellous": r_canc,
            "cortical": r_cort,
            "muscle": r_musc,
            "fat": r_fat,
            "skin": r_skin,
        }
    else:
        # Similar nested ellipses (same aspect ratio), using (a,b) for each layer.
        a_canc = p["a_canc_bone"]
        b_canc = p["b_canc_bone"]
        a_cort = p["a_cort_bone"]
        b_cort = p["b_cort_bone"]
        a_musc = p["a_muscle"]
        b_musc = p["b_muscle"]
        a_fat = p["a_fat"]
        b_fat = p["b_fat"]
        a_skin = p["a_skin"]
        b_skin = p["b_skin"]

        s0 = float(p.get("taper_scale_z0", 1.0))
        s1 = float(p.get("taper_scale_z1", 1.0))

        if bone_count == 1:
            dx = float(p.get("bone_offset_x", 0.0))
            dy = float(p.get("bone_offset_y", 0.0))
            if z_profile == "taper":
                canc = _loft_disk_to_volume(a_canc * s0, b_canc * s0, 0.0, a_canc * s1, b_canc * s1, length, cx=dx, cy=dy)
                cort = _loft_disk_to_volume(a_cort * s0, b_cort * s0, 0.0, a_cort * s1, b_cort * s1, length, cx=dx, cy=dy)
            else:
                canc = _extrude_disk_to_volume(a_canc, b_canc, length, cx=dx, cy=dy)
                cort = _extrude_disk_to_volume(a_cort, b_cort, length, cx=dx, cy=dy)
            canc2 = cort2 = None
        else:
            x1 = float(p["bone1_center_x"])
            y1 = float(p["bone1_center_y"])
            x2 = float(p["bone2_center_x"])
            y2 = float(p["bone2_center_y"])
            a_canc2 = float(p["a_canc_bone_2"])
            b_canc2 = float(p["b_canc_bone_2"])
            a_cort2 = float(p["a_cort_bone_2"])
            b_cort2 = float(p["b_cort_bone_2"])

            if z_profile == "taper":
                canc = _loft_disk_to_volume(a_canc * s0, b_canc * s0, 0.0, a_canc * s1, b_canc * s1, length, cx=x1, cy=y1)
                cort = _loft_disk_to_volume(a_cort * s0, b_cort * s0, 0.0, a_cort * s1, b_cort * s1, length, cx=x1, cy=y1)
                canc2 = _loft_disk_to_volume(a_canc2 * s0, b_canc2 * s0, 0.0, a_canc2 * s1, b_canc2 * s1, length, cx=x2, cy=y2)
                cort2 = _loft_disk_to_volume(a_cort2 * s0, b_cort2 * s0, 0.0, a_cort2 * s1, b_cort2 * s1, length, cx=x2, cy=y2)
            else:
                canc = _extrude_disk_to_volume(a_canc, b_canc, length, cx=x1, cy=y1)
                cort = _extrude_disk_to_volume(a_cort, b_cort, length, cx=x1, cy=y1)
                canc2 = _extrude_disk_to_volume(a_canc2, b_canc2, length, cx=x2, cy=y2)
                cort2 = _extrude_disk_to_volume(a_cort2, b_cort2, length, cx=x2, cy=y2)
        if z_profile == "taper":
            musc = _loft_disk_to_volume(a_musc * s0, b_musc * s0, 0.0, a_musc * s1, b_musc * s1, length)
            fat = _loft_disk_to_volume(a_fat * s0, b_fat * s0, 0.0, a_fat * s1, b_fat * s1, length)
            skin = _loft_disk_to_volume(a_skin * s0, b_skin * s0, 0.0, a_skin * s1, b_skin * s1, length)
        else:
            musc = _extrude_disk_to_volume(a_musc, b_musc, length)
            fat = _extrude_disk_to_volume(a_fat, b_fat, length)
            skin = _extrude_disk_to_volume(a_skin, b_skin, length)
        target = {
            "cancellous": (a_canc, b_canc),
            "cortical": (a_cort, b_cort),
            "muscle": (a_musc, b_musc),
            "fat": (a_fat, b_fat),
            "skin": (a_skin, b_skin),
        }

    gmsh.model.occ.synchronize()

    ents = [(3, canc), (3, cort), (3, musc), (3, fat), (3, skin)]
    if bone_count == 2:
        ents.extend([(3, canc2), (3, cort2)])

    gmsh.model.occ.fragment(ents, [])
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    eps = max(1e-2, 1e-3 * float(p.get("radius_multiplicative_factor", 40.0)))
    z_seed = 0.5 * float(length)

    if bone_count == 1:
        dx = float(p.get("bone_offset_x", 0.0))
        dy = float(p.get("bone_offset_y", 0.0))
        if abs(dx) > 1e-9 or abs(dy) > 1e-9 or z_profile != "constant":
            seeds = {
                "cancellous": _choose_seed(p, "cancellous", z_seed),
                "cortical": _choose_seed(p, "cortical", z_seed),
                "muscle": _choose_seed(p, "muscle", z_seed),
                "fat": _choose_seed(p, "fat", z_seed),
                "skin": _choose_seed(p, "skin", z_seed),
            }
            assigned = assign_volumes_by_seed_points(vols, seeds=seeds, eps=eps)
        else:
            assigned = assign_volumes_to_layers(vols, target)
    else:
        # Two-bone: assign by seed points for each bone region, then union into physical groups.
        # Find cancellous/cortical volumes for each bone separately.
        s_c1 = (float(p["bone1_center_x"]), float(p["bone1_center_y"]), z_seed)
        s_c2 = (float(p["bone2_center_x"]), float(p["bone2_center_y"]), z_seed)

        # Cortical seeds: along +x direction from each bone center
        if shape == "circle":
            r_canc = float(p["radius_canc_bone"])
            r_cort = float(p["radius_cort_bone"])
            r_canc2 = float(p["radius_canc_bone_2"])
            r_cort2 = float(p["radius_cort_bone_2"])
            s_k1 = (s_c1[0] + 0.5 * (r_canc + r_cort), s_c1[1], z_seed)
            s_k2 = (s_c2[0] + 0.5 * (r_canc2 + r_cort2), s_c2[1], z_seed)
        else:
            a_canc = float(p["a_canc_bone"])
            a_cort = float(p["a_cort_bone"])
            a_canc2 = float(p["a_canc_bone_2"])
            a_cort2 = float(p["a_cort_bone_2"])
            s_k1 = (s_c1[0] + 0.5 * (a_canc + a_cort), s_c1[1], z_seed)
            s_k2 = (s_c2[0] + 0.5 * (a_canc2 + a_cort2), s_c2[1], z_seed)

        canc_tag1 = find_volume_containing_point(vols, s_c1, eps=eps)[1]
        cort_tag1 = find_volume_containing_point(vols, s_k1, eps=eps)[1]
        canc_tag2 = find_volume_containing_point(vols, s_c2, eps=eps)[1]
        cort_tag2 = find_volume_containing_point(vols, s_k2, eps=eps)[1]

        # Outer shells are nested; bbox-based "point containment" is ambiguous. Use extents matching instead.
        z_profile = p.get("z_profile", "constant")
        if z_profile == "taper":
            smax = max(float(p.get("taper_scale_z0", 1.0)), float(p.get("taper_scale_z1", 1.0)))
        else:
            smax = 1.0

        if shape == "circle":
            r_musc = float(p["radius_muscle"]) * smax
            r_fat = float(p["radius_fat"]) * smax
            r_skin = float(p["radius_skin"]) * smax
            target_outer = {"muscle": r_musc, "fat": r_fat, "skin": r_skin}
        else:
            a_musc = float(p["a_muscle"]) * smax
            b_musc = float(p["b_muscle"]) * smax
            a_fat = float(p["a_fat"]) * smax
            b_fat = float(p["b_fat"]) * smax
            a_skin = float(p["a_skin"]) * smax
            b_skin = float(p["b_skin"]) * smax
            target_outer = {"muscle": (a_musc, b_musc), "fat": (a_fat, b_fat), "skin": (a_skin, b_skin)}

        used_bones = {canc_tag1, cort_tag1, canc_tag2, cort_tag2}
        vol_candidates = [(3, tag) for (dim, tag) in vols if dim == 3 and tag not in used_bones]
        outer = assign_volumes_to_layers(vol_candidates, target_outer)
        assigned = {
            "cancellous": [canc_tag1, canc_tag2],
            "cortical": [cort_tag1, cort_tag2],
            "muscle": outer["muscle"],
            "fat": outer["fat"],
            "skin": outer["skin"],
        }

    canc_vols = assigned["cancellous"] if isinstance(assigned["cancellous"], list) else [assigned["cancellous"]]
    cort_vols = assigned["cortical"] if isinstance(assigned["cortical"], list) else [assigned["cortical"]]

    gmsh.model.addPhysicalGroup(3, canc_vols, canc_tag)
    gmsh.model.setPhysicalName(3, canc_tag, "Cancellous Bone")

    gmsh.model.addPhysicalGroup(3, cort_vols, cort_tag)
    gmsh.model.setPhysicalName(3, cort_tag, "Cortical Bone")

    gmsh.model.addPhysicalGroup(3, [assigned["muscle"]], muscle_tag)
    gmsh.model.setPhysicalName(3, muscle_tag, "Muscle")

    gmsh.model.addPhysicalGroup(3, [assigned["fat"]], fat_tag)
    gmsh.model.setPhysicalName(3, fat_tag, "Fat")

    gmsh.model.addPhysicalGroup(3, [assigned["skin"]], skin_tag)
    gmsh.model.setPhysicalName(3, skin_tag, "Skin")

    fields_to_min = []

    def add_threshold_from_surfaces(surface_tags, field_id_offset, scale):
        gmsh.model.mesh.field.add("Distance", field_id_offset)
        gmsh.model.mesh.field.setNumbers(field_id_offset, "FacesList", surface_tags)

        gmsh.model.mesh.field.add("Threshold", field_id_offset + 1)
        gmsh.model.mesh.field.setNumber(field_id_offset + 1, "InField", field_id_offset)
        gmsh.model.mesh.field.setNumber(field_id_offset + 1, "SizeMin", size_min_factor * scale)
        gmsh.model.mesh.field.setNumber(field_id_offset + 1, "SizeMax", size_max_factor * scale)
        gmsh.model.mesh.field.setNumber(field_id_offset + 1, "DistMin", dist_min_factor * scale)
        gmsh.model.mesh.field.setNumber(field_id_offset + 1, "DistMax", dist_max_factor * scale)
        return field_id_offset + 1

    scale = p["radius_multiplicative_factor"]

    if refine_on == "skin":
        boundary_entities = gmsh.model.getBoundary([(3, assigned["skin"])], oriented=False)
        surface_tags = [b[1] for b in boundary_entities]
        th_id = add_threshold_from_surfaces(surface_tags, field_id_offset=2, scale=scale)
        gmsh.model.mesh.field.setAsBackgroundMesh(th_id)
    elif refine_on == "all_interfaces":
        fid = 2
        for layer in ["cancellous", "cortical", "muscle", "fat", "skin"]:
            bnd = gmsh.model.getBoundary([(3, assigned[layer])], oriented=False)
            surf = [b[1] for b in bnd]
            th_id = add_threshold_from_surfaces(surf, field_id_offset=fid, scale=scale)
            fields_to_min.append(th_id)
            fid += 2

        gmsh.model.mesh.field.add("Min", fid)
        gmsh.model.mesh.field.setNumbers(fid, "FieldsList", fields_to_min)
        gmsh.model.mesh.field.setAsBackgroundMesh(fid)
    else:
        raise ValueError("refine_on must be 'skin' or 'all_interfaces'")

    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", mesh_char_length_factor)
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

    t0 = time.time()
    gmsh.model.mesh.generate(3)
    gen_time = time.time() - t0

    num_nodes, num_elems_3d = compute_mesh_counts()

    out_msh.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    gmsh.write(str(out_msh))
    if bool(p.get("debug_gui", False)):
        gmsh.fltk.run()
    metadata = {
        "mesh_file": str(out_msh),
        "generated_seconds": float(gen_time),
        "num_nodes": num_nodes,
        "num_elems_3d": num_elems_3d,
        "geometry_params": p,
        "assigned_volume_tags": assigned,
        "mesh_settings": {
            "Mesh.CharacteristicLengthFactor": mesh_char_length_factor,
            "refine_on": refine_on,
            "size_min_factor": size_min_factor,
            "size_max_factor": size_max_factor,
            "dist_min_factor": dist_min_factor,
            "dist_max_factor": dist_max_factor,
        },
    }

    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


__all__ = ["build_one_mesh"]

