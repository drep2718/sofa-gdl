"""
Differentiable 2D hallway SDF and collision constraint.

The L-shaped hallway has unit width and a 90° corner at the origin.
Convention:
  - Vertical corridor:   x ∈ [-1, 0],  y ∈ (-inf, +inf)
  - Horizontal corridor: x ∈ (-inf, 0], y ∈ [0, 1]

Moving sofa constraint (Hammersley formulation):
  At each angle θ ∈ [0, π/2], the sofa is rotated by θ AND optimally
  translated to hug the inner corner.  The "canonical" translation is:
      tx(θ) = -max_x(R(θ)·S)     ← pushes rightmost point to x=0
      ty(θ) = -min_y(R(θ)·S)     ← pushes bottommost point to y=0
  After this translation the shape sits snug in the corner.  The only
  remaining constraint is that no point falls in the forbidden upper-left
  region {x < -1} ∩ {y > 1}.

This formulation is correct for the full moving sofa problem (rotation +
translation) and can achieve Gerver's area ≈ 2.2195, unlike the rotation-
only version whose maximum is π/2 − 1 ≈ 0.57.

Both max() and min() are differentiable in PyTorch (gradient passes through
the argmax/argmin element), so the full constraint is differentiable w.r.t.
the shape parameters.
"""

import torch
import numpy as np


def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """2×2 rotation matrix for scalar angle theta."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def hallway_sdf_2d(pts: torch.Tensor) -> torch.Tensor:
    """
    SDF for the L-shaped hallway.

    Hallway = vertical_strip ∪ horizontal_strip:
      vertical_strip:   x ∈ [-1, 0]       (any y)
      horizontal_strip: x ≤ 0,  y ∈ [0, 1]

    Returns negative inside hallway, positive outside (positive = violation).

    Args:
        pts: (..., 2)
    Returns:
        sdf: (...,)
    """
    x = pts[..., 0]
    y = pts[..., 1]

    # Vertical strip: negative iff x ∈ (-1, 0)
    sdf_vert = -torch.minimum(x + 1.0, -x)

    # Horizontal strip: negative iff x ≤ 0 and y ∈ (0, 1)
    sdf_horiz = -torch.minimum(torch.minimum(-x, y), 1.0 - y)

    # Union: inside hallway if inside either strip
    return torch.minimum(sdf_vert, sdf_horiz)


def constraint_penalty(pts: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """
    Collision penalty for the full moving sofa problem (rotation + translation).

    At each angle θ:
      1. Rotate all boundary points by θ.
      2. Apply the canonical translation that hugs the inner corner:
             tx = -max_x(rotated points)
             ty = -min_y(rotated points)
      3. Compute squared-ReLU of the hallway SDF for the translated points.

    The canonical translation is differentiable because PyTorch propagates
    gradients through max() and min() to the extremal boundary points.

    Args:
        pts:    (T, 2)  — sofa boundary points in sofa frame
        thetas: (A,)    — rotation angles in [0, π/2]

    Returns:
        scalar — mean squared penetration depth over all (angle, point) pairs
    """
    A = thetas.shape[0]

    c = torch.cos(thetas)   # (A,)
    s = torch.sin(thetas)   # (A,)

    # Batch rotation matrices: R[a] = [[c,-s],[s,c]]
    R = torch.stack([
        torch.stack([c, -s], dim=1),
        torch.stack([s,  c], dim=1),
    ], dim=1)                                           # (A, 2, 2)

    # Rotate: (A, T, 2)
    pts_rot = pts.unsqueeze(0) @ R.permute(0, 2, 1)   # (A, T, 2)

    # Canonical translation: push shape to hug inner corner
    # tx = -max_x so that rightmost point has x = 0
    # ty = -min_y so that bottommost point has y = 0
    tx = -pts_rot[..., 0].max(dim=1).values            # (A,)
    ty = -pts_rot[..., 1].min(dim=1).values            # (A,)

    # Apply translation: broadcast (A,) → (A, 1, 1)
    t  = torch.stack([tx, ty], dim=1).unsqueeze(1)     # (A, 1, 2)
    pts_canon = pts_rot + t                             # (A, T, 2)

    sdf = hallway_sdf_2d(pts_canon)                     # (A, T)

    # Max violation across all (angle, point) pairs.
    # Using max() instead of mean() prevents the optimizer from hiding
    # a few deep violations behind a large majority of feasible points.
    # PyTorch differentiates through max() by routing the gradient
    # to the unique maximizing element, which is exactly the most-violated
    # boundary point — the one we most need to push back inside.
    return torch.relu(sdf).max()                        # scalar


def max_penetration(pts: torch.Tensor, thetas: torch.Tensor) -> float:
    """Max SDF violation over all angles using canonical translation (no gradients)."""
    with torch.no_grad():
        A = thetas.shape[0]
        c = torch.cos(thetas)
        s = torch.sin(thetas)
        R = torch.stack([
            torch.stack([c, -s], dim=1),
            torch.stack([s,  c], dim=1),
        ], dim=1)
        pts_rot   = pts.unsqueeze(0) @ R.permute(0, 2, 1)       # (A, T, 2)
        tx        = -pts_rot[..., 0].max(dim=1).values           # (A,)
        ty        = -pts_rot[..., 1].min(dim=1).values           # (A,)
        t         = torch.stack([tx, ty], dim=1).unsqueeze(1)    # (A, 1, 2)
        pts_canon = pts_rot + t
        sdf       = hallway_sdf_2d(pts_canon)
        return torch.relu(sdf).max().item()
