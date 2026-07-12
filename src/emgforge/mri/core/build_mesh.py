#!/usr/bin/env python3
"""
MRI Segmentation → Tetrahedral FEM Mesh Pipeline

Converts a labeled NIfTI segmentation of forearm anatomy into a
tetrahedral mesh with tissue labels, compatible with FEniCSx/DOLFINx.

Pipeline:
  1. Load NIfTI segmentation (full.nii.gz)
  2. Resample z-axis to near-isotropic voxels
  3. Extract outer boundary surface (marching cubes)
  4. Smooth and decimate surface
  5. Tetrahedralize (pytetwild — robust to imperfect surfaces)
  6. Assign tissue labels from segmentation lookup
  7. Export as Gmsh .msh with physical groups

Usage:
  python mri/build_mesh.py \
    --nifti mri/data/PD_PROPELLER_5MM_FATS_FLX_0012/full.nii.gz \
    --out mri/mesh/forearm.msh \
    --edge-length 0.02

Requirements (system python):
  pip install nibabel scikit-image trimesh fast-simplification pytetwild meshio
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Label → tissue type mapping
# ---------------------------------------------------------------------------
# Tissue types for FEM conductivity assignment
TISSUE_TYPES = {
    "background": 0,
    "fat_skin": 1,
    "connective": 2,
    "muscle": 3,
    # "bone": 4,  # TODO: identify bone labels from anatomy
}

# Segmentation label → FEM tissue type
# Label 25 = subcutaneous fat + skin (outer shell)
# Labels 15, 22 = interosseous membrane / deep fascia
# All other non-zero labels = muscle (or bone — needs atlas confirmation)
LABEL_TO_TISSUE = {
    0: "background",
    25: "fat_skin",
    15: "connective",
    22: "connective",
}
# Everything else maps to "muscle" (default for non-zero labels)

# Tissue conductivities (S/m) — from src/emgforge/fem/constants.py + literature
CONDUCTIVITIES = {
    "fat_skin": {
        "description": "Subcutaneous fat + skin (combined layer)",
        "sigma": [0.0379, 0.0379, 0.0379],  # isotropic, using fat value
        "note": "Ideally split into fat (0.0379) and skin (4.55e-4) layers",
    },
    "connective": {
        "description": "Interosseous membrane, fascia, tendons",
        "sigma": [0.2, 0.2, 0.2],  # isotropic, similar to average tissue
    },
    "muscle": {
        "description": "Skeletal muscle (anisotropic along fiber direction)",
        "sigma_cross": 0.2455,  # transverse
        "sigma_fiber": 1.2275,  # along fiber (5x ratio)
        "note": "Fiber direction must be estimated per muscle",
    },
    "bone": {
        "description": "Cortical + cancellous bone (low conductivity)",
        "sigma": [0.02, 0.02, 0.02],  # isotropic, cortical bone
    },
}


def get_tissue_type(label: int) -> str:
    """Map a segmentation label to FEM tissue type."""
    if label in LABEL_TO_TISSUE:
        return LABEL_TO_TISSUE[label]
    elif label > 0:
        return "muscle"
    else:
        return "background"


def get_tissue_id(label: int) -> int:
    """Map a segmentation label to FEM tissue type ID."""
    return TISSUE_TYPES[get_tissue_type(label)]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_and_resample(nifti_path: str, target_z: float = 1.0):
    """Load NIfTI segmentation and resample z to target spacing."""
    import nibabel as nib
    from scipy.ndimage import zoom

    img = nib.load(nifti_path)
    data = img.get_fdata().astype(int)
    voxel_size = np.array(img.header.get_zooms())

    print(f"Loaded {nifti_path}")
    print(f"  Shape: {data.shape}, Voxel: {voxel_size} mm")
    print(f"  z anisotropy: {voxel_size[2]/voxel_size[0]:.1f}x")

    if voxel_size[2] > target_z * 1.5:
        z_factor = voxel_size[2] / target_z
        print(f"  Resampling z: {voxel_size[2]:.1f}mm → {target_z:.1f}mm (factor {z_factor:.1f})")
        data = zoom(data.astype(np.float32), (1, 1, z_factor), order=0).astype(int)
        voxel_size = np.array([voxel_size[0], voxel_size[1], target_z])
        print(f"  New shape: {data.shape}")
    else:
        print(f"  z spacing OK ({voxel_size[2]:.2f}mm), no resampling needed")

    return data, voxel_size


def extract_surface(seg_data, voxel_size):
    """Extract outer boundary surface using marching cubes."""
    from skimage import measure

    forearm_mask = (seg_data > 0).astype(float)
    verts, faces, normals, values = measure.marching_cubes(
        forearm_mask, level=0.5, spacing=voxel_size, step_size=1
    )
    print(f"Marching cubes: {len(verts)} verts, {len(faces)} faces")
    return verts, faces


def smooth_and_decimate(verts, faces, target_faces=50000, smooth_iters=10):
    """Smooth surface with Laplacian filter and decimate."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()

    # Smooth
    if smooth_iters > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)
        print(f"Smoothed ({smooth_iters} iterations)")

    # Decimate if needed
    if len(mesh.faces) > target_faces * 1.2:
        import fast_simplification
        reduction = 1.0 - target_faces / len(mesh.faces)
        verts_dec, faces_dec = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces,
            target_reduction=reduction,
        )
        mesh = trimesh.Trimesh(vertices=verts_dec, faces=faces_dec)
        mesh.fix_normals()
        print(f"Decimated: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print(f"Surface: watertight={mesh.is_watertight}, volume={abs(mesh.volume):.0f} mm³")
    return mesh.vertices, mesh.faces


def tetrahedralize(verts, faces, edge_length_fac=0.02):
    """Generate tetrahedral volume mesh using pytetwild."""
    import pytetwild

    t0 = time.time()
    print(f"Tetrahedralizing (edge_length_fac={edge_length_fac})...")
    tet_verts, tet_cells = pytetwild.tetrahedralize(
        verts, faces,
        edge_length_fac=edge_length_fac,
        optimize=True,
    )
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s: {len(tet_verts)} vertices, {len(tet_cells)} tetrahedra")

    # Quality check
    v = [tet_verts[tet_cells[:, i]] for i in range(4)]
    volumes = np.abs(np.sum(np.cross(v[1]-v[0], v[2]-v[0]) * (v[3]-v[0]), axis=1)) / 6
    print(f"  Tet volumes (mm³): min={volumes.min():.4f}, "
          f"max={volumes.max():.2f}, median={np.median(volumes):.2f}")
    print(f"  Total volume: {volumes.sum():.0f} mm³")

    return tet_verts, tet_cells


def assign_labels(tet_verts, tet_cells, seg_data, voxel_size):
    """Assign tissue labels to tetrahedra by centroid lookup in segmentation."""
    centroids = tet_verts[tet_cells].mean(axis=1)

    # Convert physical coords → voxel indices
    voxel_idx = (centroids / voxel_size).astype(int)
    for d in range(3):
        voxel_idx[:, d] = np.clip(voxel_idx[:, d], 0, seg_data.shape[d] - 1)

    seg_labels = seg_data[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]]

    # Map to FEM tissue types
    tissue_ids = np.array([get_tissue_id(l) for l in seg_labels])

    # Report
    print("Tissue assignment:")
    for tname, tid in TISSUE_TYPES.items():
        n = np.sum(tissue_ids == tid)
        print(f"  {tname} (id={tid}): {n} tets ({100*n/len(tissue_ids):.1f}%)")

    return seg_labels, tissue_ids


def export_msh(tet_verts, tet_cells, tissue_ids, seg_labels, out_path):
    """Export as Gmsh .msh v2 format with physical groups per tissue type.

    Uses the Gmsh API to produce a .msh file that DOLFINx
    ``dolfinx.io.gmshio.read_from_msh()`` can load with cell_markers.
    Falls back to a direct ASCII writer if gmsh is not available.
    """
    try:
        return _export_msh_gmsh_api(tet_verts, tet_cells, tissue_ids, out_path)
    except ImportError:
        return _export_msh_ascii(tet_verts, tet_cells, tissue_ids, seg_labels, out_path)


def _export_msh_gmsh_api(tet_verts, tet_cells, tissue_ids, out_path):
    """Write .msh using the Gmsh Python API (reliable physical groups)."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("forearm")

    # Physical group names
    phys_names = {0: "background", 1: "fat_skin", 2: "connective", 3: "muscle"}

    # Add nodes (1-indexed)
    node_tags = np.arange(1, len(tet_verts) + 1, dtype=np.int64)
    gmsh.model.mesh.addNodes(
        dim=3, tag=1,
        nodeTags=node_tags,
        coord=tet_verts.astype(np.float64).flatten(),
    )

    # Group tets by tissue type and add as elements in physical groups
    unique_types = np.unique(tissue_ids)
    for tid in unique_types:
        mask = tissue_ids == tid
        cells_subset = tet_cells[mask] + 1  # 1-indexed
        n = cells_subset.shape[0]

        # Add physical group for this tissue type
        name = phys_names.get(int(tid), f"tissue_{tid}")
        phys_tag = int(tid) + 1  # physical tags must be >= 1 for DOLFINx
        gmsh.model.addDiscreteEntity(3, phys_tag)
        gmsh.model.addPhysicalGroup(3, [phys_tag], tag=phys_tag, name=name)

        # Add elements to this entity
        elem_tags = np.arange(1, n + 1, dtype=np.int64) + mask.nonzero()[0][0]
        gmsh.model.mesh.addNodes(
            dim=3, tag=phys_tag,
            nodeTags=[],
            coord=[],
        )
        gmsh.model.mesh.addElementsByType(
            tag=phys_tag,
            elementType=4,  # 4-node tetrahedron
            elementTags=[],  # auto-number
            nodeTags=cells_subset.flatten().astype(np.int64),
        )

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(out_path))
    gmsh.finalize()
    print(f"Saved {out_path} (Gmsh API, {len(tet_verts)} nodes, {len(tet_cells)} tets)")


def _export_msh_ascii(tet_verts, tet_cells, tissue_ids, seg_labels, out_path):
    """Write Gmsh MSH 2.2 ASCII directly (no gmsh dependency needed).

    Format spec: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    DOLFINx gmshio expects physical entity tags (column 4 of element data).
    """
    phys_names = {0: "background", 1: "fat_skin", 2: "connective", 3: "muscle"}
    unique_types = sorted(np.unique(tissue_ids))

    with open(out_path, "w") as f:
        # Header
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Physical names
        f.write(f"$PhysicalNames\n{len(unique_types)}\n")
        for tid in unique_types:
            # physical-dimension physical-tag "name"
            name = phys_names.get(int(tid), f"tissue_{tid}")
            tag = int(tid) + 1
            f.write(f'3 {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")

        # Nodes (1-indexed)
        n_nodes = len(tet_verts)
        f.write(f"$Nodes\n{n_nodes}\n")
        for i, (x, y, z) in enumerate(tet_verts, 1):
            f.write(f"{i} {x:.10g} {y:.10g} {z:.10g}\n")
        f.write("$EndNodes\n")

        # Elements (tetrahedra, type 4)
        # Format: elm-number elm-type number-of-tags <tags> node-list
        # tags: physical-entity-tag, elementary-entity-tag
        n_elems = len(tet_cells)
        f.write(f"$Elements\n{n_elems}\n")
        for i, (tet, tid) in enumerate(zip(tet_cells, tissue_ids), 1):
            phys_tag = int(tid) + 1  # 1-indexed
            geom_tag = phys_tag     # use same as elementary entity
            n1, n2, n3, n4 = tet + 1  # 1-indexed nodes
            f.write(f"{i} 4 2 {phys_tag} {geom_tag} {n1} {n2} {n3} {n4}\n")
        f.write("$EndElements\n")

    print(f"Saved {out_path} (MSH 2.2 ASCII, {n_nodes} nodes, {n_elems} tets)")


def export_xdmf(tet_verts, tet_cells, tissue_ids, seg_labels, out_dir):
    """Export as XDMF (HDF5-backed) for direct DOLFINx loading."""
    import meshio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mesh file
    mesh = meshio.Mesh(
        points=tet_verts,
        cells=[("tetra", tet_cells)],
    )
    meshio.xdmf.write(out_dir / "mesh.xdmf", mesh)

    # Cell tags (tissue type IDs)
    tag_mesh = meshio.Mesh(
        points=tet_verts,
        cells=[("tetra", tet_cells)],
        cell_data={"tissue_type": [tissue_ids.astype(np.int32)]},
    )
    meshio.xdmf.write(out_dir / "cell_tags.xdmf", tag_mesh)

    # Per-muscle labels
    label_mesh = meshio.Mesh(
        points=tet_verts,
        cells=[("tetra", tet_cells)],
        cell_data={"seg_label": [seg_labels.astype(np.int32)]},
    )
    meshio.xdmf.write(out_dir / "seg_labels.xdmf", label_mesh)

    print(f"Saved XDMF files to {out_dir}/")


def save_metadata(out_path, nifti_path, seg_data, voxel_size,
                  tet_verts, tet_cells, tissue_ids, seg_labels):
    """Save mesh metadata as JSON."""
    unique_labels, counts = np.unique(seg_labels, return_counts=True)
    label_stats = {int(l): int(c) for l, c in zip(unique_labels, counts)}

    meta = {
        "source_nifti": str(nifti_path),
        "segmentation_shape": list(seg_data.shape),
        "voxel_size_mm": list(voxel_size),
        "n_vertices": int(len(tet_verts)),
        "n_tetrahedra": int(len(tet_cells)),
        "bounds_min_mm": tet_verts.min(axis=0).tolist(),
        "bounds_max_mm": tet_verts.max(axis=0).tolist(),
        "tissue_types": TISSUE_TYPES,
        "label_to_tissue": {str(k): v for k, v in LABEL_TO_TISSUE.items()},
        "conductivities": CONDUCTIVITIES,
        "seg_label_counts": label_stats,
        "tissue_type_counts": {
            name: int(np.sum(tissue_ids == tid))
            for name, tid in TISSUE_TYPES.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MRI segmentation → FEM mesh")
    parser.add_argument("--nifti", required=True,
                        help="Path to labeled NIfTI segmentation (.nii.gz)")
    parser.add_argument("--out", default="mri/mesh/forearm.msh",
                        help="Output mesh path (.msh)")
    parser.add_argument("--target-z", type=float, default=1.0,
                        help="Target z-spacing after resampling (mm)")
    parser.add_argument("--edge-length", type=float, default=0.02,
                        help="pytetwild edge_length_fac (smaller=finer, 0.02→~200K tets)")
    parser.add_argument("--surface-faces", type=int, default=50000,
                        help="Target surface face count after decimation")
    parser.add_argument("--smooth-iters", type=int, default=10,
                        help="Laplacian smoothing iterations")
    parser.add_argument("--xdmf", action="store_true",
                        help="Also export as XDMF (for DOLFINx)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    stem = out_path.stem

    t_total = time.time()

    # 1. Load and resample
    seg_data, voxel_size = load_and_resample(args.nifti, args.target_z)

    # 2. Extract surface
    verts, faces = extract_surface(seg_data, voxel_size)

    # 3. Smooth and decimate
    verts, faces = smooth_and_decimate(
        verts, faces,
        target_faces=args.surface_faces,
        smooth_iters=args.smooth_iters,
    )

    # 4. Tetrahedralize
    tet_verts, tet_cells = tetrahedralize(verts, faces, args.edge_length)

    # 5. Assign labels
    seg_labels, tissue_ids = assign_labels(tet_verts, tet_cells, seg_data, voxel_size)

    # 6. Export
    export_msh(tet_verts, tet_cells, tissue_ids, seg_labels, str(out_path))

    if args.xdmf:
        export_xdmf(tet_verts, tet_cells, tissue_ids, seg_labels,
                     out_dir / f"{stem}_xdmf")

    # 7. Metadata
    save_metadata(
        str(out_dir / f"{stem}_metadata.json"),
        args.nifti, seg_data, voxel_size,
        tet_verts, tet_cells, tissue_ids, seg_labels,
    )

    dt = time.time() - t_total
    print(f"\nTotal pipeline time: {dt:.1f}s")


if __name__ == "__main__":
    main()
