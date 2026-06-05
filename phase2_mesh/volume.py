"""
Differentiable mesh volume via the Divergence Theorem.

The signed volume of a closed triangle mesh is:

    V = (1/6) * sum over triangles of [ v1 · (v2 × v3) ]

This is the divergence theorem applied to F(x) = x/3 (whose divergence = 1).
Each triangle contributes its "signed tetrahedral volume" relative to the origin.

Why this works:
  - Exact for any closed mesh (no approximation)
  - O(M) in the number of faces
  - Fully differentiable w.r.t. vertex positions via autograd
  - Returns the correct sign: positive for outward-normal meshes

Reference: Zhang & Chen, "Efficient Feature Extraction for 2D/3D Objects in
Mesh Representation" (ICIP 2001)
"""

import torch


def mesh_volume(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Signed volume of a closed triangle mesh.

    Args:
        V: (N, 3) vertex positions — differentiable
        F: (M, 3) face indices (integer, not differentiable)

    Returns:
        vol: scalar tensor — signed volume (positive for correctly oriented mesh)
    """
    v1 = V[F[:, 0]]  # (M, 3)
    v2 = V[F[:, 1]]  # (M, 3)
    v3 = V[F[:, 2]]  # (M, 3)

    # Signed tetrahedral volume for each triangle
    # v1 · (v2 × v3) = scalar triple product
    cross = torch.cross(v2, v3, dim=1)   # (M, 3)
    signed_tet_vols = (v1 * cross).sum(dim=1)  # (M,)

    return signed_tet_vols.sum() / 6.0


def mesh_volume_abs(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Absolute volume — use this for the optimization objective.

    The sign depends on face orientation convention (outward vs inward normals).
    Taking abs() removes the dependency on convention. Gradients still flow
    correctly because torch.abs is differentiable everywhere except at zero,
    which can't happen for a valid mesh.
    """
    return torch.abs(mesh_volume(V, F))


def mesh_surface_area(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Total surface area of the mesh.

    Area of triangle = 0.5 * ||(v2 - v1) × (v3 - v1)||

    Differentiable w.r.t. V. Used as a regularizer to prevent
    degenerate triangles during optimization.
    """
    v1 = V[F[:, 0]]
    v2 = V[F[:, 1]]
    v3 = V[F[:, 2]]

    cross = torch.cross(v2 - v1, v3 - v1, dim=1)  # (M, 3)
    face_areas = cross.norm(dim=1) * 0.5           # (M,)
    return face_areas.sum()


def laplacian_smoothness(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Mesh Laplacian regularizer: penalizes deviation from uniform vertex spacing.

    For each vertex, computes its mean displacement from its neighbors.
    Sum of squared magnitudes → scalar penalty.

    Used during 3D optimization (Phase 4) to prevent self-intersections
    and mesh collapse.
    """
    N = V.shape[0]

    # Build edge list from faces
    edges = torch.cat([
        F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
        F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]],
    ], dim=0)  # (2M*3, 2) — directed edges both ways

    src = edges[:, 0]
    dst = edges[:, 1]

    # For each vertex: mean of neighbor positions
    neighbor_sum = torch.zeros_like(V)
    neighbor_count = torch.zeros(N, 1, dtype=V.dtype, device=V.device)

    neighbor_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), V[src])
    neighbor_count.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, dtype=V.dtype, device=V.device))

    neighbor_mean = neighbor_sum / (neighbor_count + 1e-8)

    # Laplacian = vertex - mean(neighbors)
    lap = V - neighbor_mean
    return (lap * lap).sum()
