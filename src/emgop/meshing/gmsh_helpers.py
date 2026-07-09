from __future__ import annotations

import gmsh


def radial_extent_from_bbox(bbox) -> float:
    """
    bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
    Returns maximum radial extent in XY.
    """
    xmin, ymin, _, xmax, ymax, _ = bbox
    return float(max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)))


def xy_extents_from_bbox(bbox) -> tuple[float, float]:
    """
    bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
    Returns (x_extent, y_extent) where extent = max(abs(min), abs(max)).
    """
    xmin, ymin, _, xmax, ymax, _ = bbox
    return float(max(abs(xmin), abs(xmax))), float(max(abs(ymin), abs(ymax)))


def assign_volumes_to_layers(vol_entities, target_radii):
    """
    vol_entities: list of (3, tag) returned by gmsh.model.getEntities(3)
    target_radii: dict layer_name -> target size descriptor:
      - float: expected max radial extent in XY (works for circles; also works for ellipses if you pass max(a,b))
      - tuple[float,float]: expected (x_extent, y_extent) in XY (preferred for ellipses)
    Returns dict layer_name -> volume_tag
    """
    vol_info = []
    for _, tag in vol_entities:
        bbox = gmsh.model.occ.getBoundingBox(3, tag)
        rx_est = radial_extent_from_bbox(bbox)
        ex, ey = xy_extents_from_bbox(bbox)
        vol_info.append((tag, rx_est, ex, ey, bbox))

    assigned = {}
    used = set()
    for layer, r_target in target_radii.items():
        best = None
        best_err = 1e99
        for tag, r_est, ex, ey, bbox in vol_info:
            if tag in used:
                continue
            if isinstance(r_target, (tuple, list)) and len(r_target) == 2:
                tx, ty = float(r_target[0]), float(r_target[1])
                err = (abs(ex - tx) + abs(ey - ty))
            else:
                err = abs(r_est - float(r_target))
            if err < best_err:
                best = (tag, r_est, bbox)
                best_err = err
        if best is None:
            raise RuntimeError(f"Could not assign a volume to layer {layer}")
        assigned[layer] = best[0]
        used.add(best[0])

    return assigned


def _bbox_volume(bbox) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    return float(max(0.0, xmax - xmin) * max(0.0, ymax - ymin) * max(0.0, zmax - zmin))


def find_volume_containing_point(vol_entities, point_xyz, eps: float):
    """
    Best-effort: return the (3, tag) whose bbox contains the point (within eps),
    preferring the smallest bbox volume.
    """
    px, py, pz = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])

    # Try gmsh query first
    try:
        candidates = gmsh.model.getEntitiesInBoundingBox(
            px - eps, py - eps, pz - eps, px + eps, py + eps, pz + eps, 3
        )
    except Exception:
        candidates = []

    vol_tags = {tag for _, tag in vol_entities}
    candidates = [(3, tag) for dim, tag in candidates if dim == 3 and tag in vol_tags]

    # Fallback: bbox test
    if not candidates:
        for _, tag in vol_entities:
            bbox = gmsh.model.occ.getBoundingBox(3, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            if (px >= xmin - eps and px <= xmax + eps and py >= ymin - eps and py <= ymax + eps and pz >= zmin - eps and pz <= zmax + eps):
                candidates.append((3, tag))

    if not candidates:
        raise RuntimeError(f"No volume found containing point {point_xyz}")

    # Choose smallest bbox volume to disambiguate
    best = None
    best_vol = 1e99
    for _, tag in candidates:
        bbox = gmsh.model.occ.getBoundingBox(3, tag)
        bv = _bbox_volume(bbox)
        if bv < best_vol:
            best = (3, tag)
            best_vol = bv
    return best


def assign_volumes_by_seed_points(vol_entities, seeds: dict[str, tuple[float, float, float]], eps: float):
    """
    Assign volumes by seed points. This is robust for off-center/multi-object geometries.
    seeds: layer_name -> (x,y,z) expected to lie inside that layer.
    """
    assigned = {}
    used = set()
    for layer, pt in seeds.items():
        dim_tag = find_volume_containing_point(vol_entities, pt, eps=eps)
        tag = dim_tag[1]
        if tag in used:
            raise RuntimeError(f"Seed for {layer} mapped to already-used volume tag {tag}")
        assigned[layer] = tag
        used.add(tag)
    return assigned


def compute_mesh_counts():
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    num_nodes = len(node_tags)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    num_elems_3d = sum(len(tags) for tags in elem_tags)
    return int(num_nodes), int(num_elems_3d)


__all__ = [
    "radial_extent_from_bbox",
    "xy_extents_from_bbox",
    "assign_volumes_to_layers",
    "assign_volumes_by_seed_points",
    "compute_mesh_counts",
]

