"""
3D L-shaped hallway Signed Distance Function (differentiable).

The 3D hallway is the 2D L-shape extruded infinitely in z.
In practice we constrain z ∈ [-0.5, 0.5] (the sofa has height ≤ 1).

Coordinate system (matches Phase 1):
  Vertical corridor:   x ∈ [-1, 0], z ∈ [-0.5, 0.5],  any y
  Horizontal corridor: y ∈ [0,  1], z ∈ [-0.5, 0.5],  x ≤ 0
  Height constraint:   z ∈ [-0.5, 0.5]

A point is inside the hallway if it satisfies the 2D L-shape condition AND
the height constraint. SDF is positive outside (= wall penetration depth).

For rotation during the corner turn: we rotate about the z-axis (the sofa
slides flat). Rotation matrix R_z(θ) rotates x-y plane, z unchanged.
"""

import torch
import numpy as np


def hallway_sdf_3d(pts: torch.Tensor) -> torch.Tensor:
    """
    3D hallway SDF — positive outside (= penetration into wall).

    The hallway is: (2D L-shape) × [-0.5, 0.5] in z.

    Args:
        pts: shape (..., 3)

    Returns:
        sdf: shape (...,) — positive = outside hallway
    """
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]

    # ── 2D L-shape SDF (same logic as Phase 1 collision.py) ──────────────────

    # Vertical corridor: x ∈ [-1, 0], any y
    sdf_vert = -torch.minimum(x + 1.0, -x)
    # sdf_vert < 0 iff x ∈ (-1, 0)

    # Horizontal corridor: x ≤ 0, y ∈ [0, 1]
    sdf_horiz = -torch.minimum(
        torch.minimum(-x, -(-y)),   # -x >= 0 iff x<=0, -(-y) = y >= 0 iff y>=0
        -(y - 1.0)                  # -(y-1) >= 0 iff y<=1
    )
    # sdf_horiz < 0 iff x ≤ 0 and y ∈ (0, 1)

    # Union (inside either corridor)
    sdf_2d = torch.minimum(sdf_vert, sdf_horiz)

    # ── Height constraint: z ∈ [-0.5, 0.5] ───────────────────────────────────
    sdf_height = -torch.minimum(z + 0.5, 0.5 - z)
    # sdf_height < 0 iff z ∈ (-0.5, 0.5)

    # ── Combined: point must satisfy BOTH 2D and height constraints ───────────
    # Inside hallway iff both sdf_2d < 0 AND sdf_height < 0
    # SDF of intersection = max(sdf_2d, sdf_height)
    sdf = torch.maximum(sdf_2d, sdf_height)

    return sdf


def rotation_matrix_3d_z(theta: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix about the z-axis by angle theta.

    The sofa lies in the x-y plane and rotates about z — this is the 3D
    extension of the 2D rotation from Phase 1.

    Returns shape (3, 3).
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    O = torch.zeros_like(c)
    I = torch.ones_like(c)

    R = torch.stack([
        torch.stack([c, -s, O]),
        torch.stack([s,  c, O]),
        torch.stack([O,  O, I]),
    ])
    return R


def constraint_penalty_3d(
    V: torch.Tensor,
    thetas: torch.Tensor,
) -> torch.Tensor:
    """
    Total collision penalty over all rotation angles for a 3D mesh.

    For each theta: rotate V about z-axis by theta, then measure how much
    any vertex penetrates the hallway walls.

    Args:
        V:      (N, 3) mesh vertices
        thetas: (A,) rotation angles in [0, pi/2]

    Returns:
        penalty: scalar — sum of mean squared penetrations over all angles
    """
    total = torch.zeros(1, dtype=V.dtype, device=V.device)

    for theta in thetas:
        R = rotation_matrix_3d_z(theta).to(V.device)  # (3, 3)
        V_rot = V @ R.T                                # (N, 3)
        sdf = hallway_sdf_3d(V_rot)                   # (N,)
        penalty = torch.relu(sdf).pow(2).mean()
        total = total + penalty

    return total.squeeze()


def max_penetration_3d(V: torch.Tensor, thetas: torch.Tensor) -> float:
    """
    Maximum SDF violation across all angles. No gradients needed.
    Used for feasibility labeling and result verification.
    """
    worst = 0.0
    with torch.no_grad():
        for theta in thetas:
            R = rotation_matrix_3d_z(theta)
            V_rot = V @ R.T
            sdf = hallway_sdf_3d(V_rot)
            worst = max(worst, sdf.max().item())
    return worst


def is_feasible(V: torch.Tensor, thetas: torch.Tensor, tol: float = 1e-3) -> bool:
    """
    Boolean feasibility check: can this mesh navigate the corner?

    A mesh is feasible if max_penetration is below tolerance.
    tol=1e-3 accounts for floating-point and discretization noise.
    """
    return max_penetration_3d(V, thetas) < tol
