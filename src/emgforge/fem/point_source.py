"""
Native FEniCSx point source implementation.

Replicates scifem.PointSource behavior without external dependencies.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem, geometry
from dolfinx.fem import Function, FunctionSpace


class NativePointSource:
    """
    Point source implementation using pure FEniCSx/dolfinx.

    Replicates scifem.PointSource behavior without external dependencies.
    Evaluates basis functions at point locations and adds contributions
    to a vector.

    Parameters
    ----------
    V : FunctionSpace
        The function space for the point source
    points : np.ndarray
        Array of shape (n, 3) or (3,) with point coordinates
    magnitude : float
        The magnitude of each point source contribution

    Attributes
    ----------
    V : FunctionSpace
        The function space
    points : np.ndarray
        Array of shape (n, 3) with point coordinates
    magnitude : float
        The magnitude of each point source
    mesh : Mesh
        The mesh from the function space

    Notes
    -----
    This implementation follows the scifem.PointSource approach:
    1. Locate points in mesh cells using bounding box tree
    2. Transform points to reference coordinates using cmap.pull_back()
    3. Evaluate basis functions at reference coordinates
    4. Add magnitude * basis_value to DoFs of containing cells

    Points outside the mesh are ignored with a warning.
    """

    def __init__(
        self,
        V: FunctionSpace,
        points: np.ndarray,
        magnitude: float = 1.0,
    ):
        self.V = V
        self.points = np.atleast_2d(points).astype(np.float64)
        self.magnitude = magnitude
        self.mesh = V.mesh

        # Ensure points have correct shape (n, 3)
        if self.points.ndim == 1:
            self.points = self.points.reshape(1, -1)
        if self.points.shape[1] != 3:
            raise ValueError(f"Points must have shape (n, 3), got {self.points.shape}")

        # Compute cell contributions on initialization
        self._cells, self._basis_values, self._valid_indices = self._compute_cell_contributions()

    def _compute_cell_contributions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Locate points in cells and evaluate basis functions.

        Returns
        -------
        cells : np.ndarray
            Cell indices for each valid point
        basis_values : np.ndarray
            Basis function values at each point, shape (n_valid, n_basis)
        valid_indices : np.ndarray
            Indices of points that were found inside the mesh
        """
        mesh = self.mesh
        tdim = mesh.topology.dim

        # Step 1: Find cells containing each point using bounding box tree
        tree = geometry.bb_tree(mesh, tdim)
        cell_candidates = geometry.compute_collisions_points(tree, self.points)
        cells = geometry.compute_colliding_cells(mesh, cell_candidates, self.points)

        # Get first colliding cell for each point
        point_cells = []
        valid_indices = []
        for i in range(len(self.points)):
            cell_list = cells.links(i)
            if len(cell_list) > 0:
                point_cells.append(cell_list[0])
                valid_indices.append(i)

        if len(valid_indices) != len(self.points):
            missing = len(self.points) - len(valid_indices)
            print(f"Warning: {missing} point(s) outside mesh domain, ignored")

        if len(valid_indices) == 0:
            return np.array([], dtype=np.int32), np.array([]), np.array([], dtype=np.int32)

        point_cells = np.array(point_cells, dtype=np.int32)
        valid_indices = np.array(valid_indices, dtype=np.int32)

        # Step 2: Transform points to reference coordinates
        # Handle API change in FEniCSx: cmap -> cmaps (list of cmaps per cell type)
        if hasattr(mesh.geometry, 'cmap'):
            cmap = mesh.geometry.cmap
        else:
            # New API: cmaps is a list, use first one (works for single cell type meshes)
            cmap = mesh.geometry.cmaps[0]

        # Get geometric DoFs for cells
        x_dofs = mesh.geometry.dofmap
        x_coords = mesh.geometry.x

        ref_points = []
        for i, (pt_idx, cell) in enumerate(zip(valid_indices, point_cells)):
            # Get cell geometry
            cell_dofs = x_dofs[cell]
            cell_coords = x_coords[cell_dofs]

            # Pull back to reference coordinates
            physical_pt = self.points[pt_idx].reshape(1, -1)
            ref_pt = cmap.pull_back(physical_pt, cell_coords)
            ref_points.append(ref_pt[0])

        ref_points = np.array(ref_points, dtype=np.float64)

        # Step 3: Evaluate basis functions at reference points
        basis_values = self._tabulate_basis(ref_points)

        return point_cells, basis_values, valid_indices

    def _tabulate_basis(self, ref_points: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions at reference points.

        Parameters
        ----------
        ref_points : np.ndarray
            Reference coordinates, shape (n_points, tdim)

        Returns
        -------
        basis_vals : np.ndarray
            Basis function values, shape (n_points, n_basis)
        """
        element = self.V.element

        # Tabulate basis functions using basix
        # For derivative order 0, output shape is (1, n_points, n_basis, value_size)
        # or (n_derivs, n_points, n_basis) for scalar elements
        tab = element.basix_element.tabulate(0, ref_points)

        # Extract values (derivative order 0)
        # tab[0] has shape (n_points, n_basis, value_size) or (n_points, n_basis)
        basis_vals = tab[0]

        # For scalar elements (value_size == 1), squeeze the last dimension
        if basis_vals.ndim == 3 and basis_vals.shape[-1] == 1:
            basis_vals = basis_vals.squeeze(-1)

        return basis_vals

    def apply_to_vector(self, b: Function | fem.Function):
        """
        Apply point source contributions to vector.

        Adds magnitude * phi_i(x) to b[dof_i] for each point x and
        each basis function phi_i in the containing cell.

        Parameters
        ----------
        b : Function
            Target function/vector to modify. Can be a dolfinx.fem.Function
            or similar object with a vector.array attribute.

        Notes
        -----
        This modifies b in-place.
        """
        if len(self._cells) == 0:
            return  # No valid points

        # Get the array to modify
        if hasattr(b, 'x'):
            # dolfinx.fem.Function
            b_array = b.x.array
        elif hasattr(b, 'vector'):
            # Function with vector attribute
            b_array = b.vector.array
        else:
            # Assume direct array access
            b_array = b.array

        dofmap = self.V.dofmap

        for i, cell in enumerate(self._cells):
            # Get DoFs for this cell
            cell_dofs = dofmap.cell_dofs(cell)

            # Add contributions: magnitude * basis_value
            for j, dof in enumerate(cell_dofs):
                b_array[dof] += self.magnitude * self._basis_values[i, j]

    @property
    def n_points(self) -> int:
        """Number of input points."""
        return len(self.points)

    @property
    def n_valid_points(self) -> int:
        """Number of points found inside the mesh."""
        return len(self._valid_indices)

    def __repr__(self) -> str:
        return (
            f"NativePointSource(V={self.V}, "
            f"n_points={self.n_points}, "
            f"n_valid={self.n_valid_points}, "
            f"magnitude={self.magnitude})"
        )
