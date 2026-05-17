"""
Parametric 2D sofa shape via polar B-spline.

WHY POLAR, NOT CARTESIAN:
The original Cartesian B-spline parameterization allowed self-intersection:
gradient descent moved control points into configurations where the spline
boundary crossed itself. The shoelace formula then overcounts area (it counts
each enclosed loop separately), producing physically impossible areas > π.

The fix: represent the boundary in polar coordinates as r(α) for α ∈ [0, 2π).
If r(α) > 0 everywhere, the curve is star-shaped w.r.t. the centroid —
provably no self-intersection. B-spline basis functions are always ≥ 0, so
if every control radius is positive (enforced via softplus), r(α) > 0 always.

The polar area formula:
    Area = (1/2) ∫₀²π r(α)² dα  ≈  (π/T) · Σᵢ r(αᵢ)²
is exact for star-shaped curves and never overcounts.

The centroid (cx, cy) is a second learnable parameter — it shifts the entire
shape in the plane and is initialized at (-0.5, 0.5), the center of the
overlapping region of both corridors where the optimal sofa lives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _bspline_basis(t: torch.Tensor, n_ctrl: int) -> torch.Tensor:
    """
    Build a (T, N) basis matrix for a uniform closed cubic B-spline.

    Partition of unity: each row sums to 1.
    Non-negativity: all entries ≥ 0.
    Together these guarantee: if all control values > 0, the spline > 0 everywhere.

    Args:
        t: (T,) parameter values in [0, 1)
        n_ctrl: number of control points N

    Returns:
        B: (T, N) — non-negative, rows sum to 1
    """
    T = t.shape[0]
    u = t * n_ctrl
    k = u.long() % n_ctrl
    s = u - u.floor()

    s2 = s * s
    s3 = s2 * s
    b0 = (1 - 3*s + 3*s2 - s3) / 6.0
    b1 = (4 - 6*s2 + 3*s3) / 6.0
    b2 = (1 + 3*s + 3*s2 - 3*s3) / 6.0
    b3 = s3 / 6.0

    B = torch.zeros(T, n_ctrl, dtype=t.dtype, device=t.device)
    B.scatter_add_(1, ((k - 1) % n_ctrl).unsqueeze(1), b0.unsqueeze(1))
    B.scatter_add_(1, (k       % n_ctrl).unsqueeze(1), b1.unsqueeze(1))
    B.scatter_add_(1, ((k + 1) % n_ctrl).unsqueeze(1), b2.unsqueeze(1))
    B.scatter_add_(1, ((k + 2) % n_ctrl).unsqueeze(1), b3.unsqueeze(1))
    return B


class SofaShape(nn.Module):
    """
    2D sofa shape in polar coordinates: boundary = centroid + r(α) * (cos α, sin α).

    Learnable parameters:
        raw_r:    (N_ctrl,)  — raw radii before softplus; r_ctrl = softplus(raw_r)
        centroid: (2,)       — (cx, cy) shifts the shape in the plane

    The radius r(α) = B(α) @ softplus(raw_r) is always > 0 → no self-intersection.
    """

    def __init__(
        self,
        n_ctrl:   int   = 64,
        n_sample: int   = 512,
        radius:   float = 0.35,
        # Start centroid at center of overlapping corridor region
        cx0:      float = -0.5,
        cy0:      float =  0.5,
    ):
        super().__init__()
        self.n_ctrl  = n_ctrl
        self.n_sample = n_sample

        # raw_r initialized so softplus(raw_r) ≈ radius
        # softplus(x) = log(1 + exp(x)) ≈ x for x >> 0
        # Inverse: raw = log(exp(radius) - 1)
        raw_init = float(np.log(np.exp(radius) - 1.0 + 1e-8))
        self.raw_r    = nn.Parameter(torch.full((n_ctrl,), raw_init))
        self.centroid = nn.Parameter(torch.tensor([cx0, cy0], dtype=torch.float32))

        # Fixed sample angles and B-spline basis (precomputed once)
        t      = torch.linspace(0, 1, n_sample + 1)[:n_sample]
        alphas = torch.linspace(0, 2 * np.pi, n_sample + 1)[:n_sample]
        B      = _bspline_basis(t, n_ctrl)  # (T, N_ctrl)

        self.register_buffer("t",      t)
        self.register_buffer("alphas", alphas)
        self.register_buffer("B",      B)

    def radii(self) -> torch.Tensor:
        """Per-sample radii r(α): always positive. Shape (T,)."""
        r_ctrl = F.softplus(self.raw_r)   # (N_ctrl,) > 0
        return self.B @ r_ctrl            # (T,)      > 0

    def forward(self) -> torch.Tensor:
        """
        Boundary points in Cartesian coordinates.

        Returns:
            pts: (T, 2) — (cx + r·cos α,  cy + r·sin α)
        """
        r   = self.radii()                              # (T,)
        cos = torch.cos(self.alphas)                    # (T,)
        sin = torch.sin(self.alphas)                    # (T,)
        x   = self.centroid[0] + r * cos               # (T,)
        y   = self.centroid[1] + r * sin               # (T,)
        return torch.stack([x, y], dim=1)               # (T, 2)

    def area(self) -> torch.Tensor:
        """
        Exact polar area: A = (1/2) ∫₀²π r(α)² dα ≈ (π/T) · Σ r(αᵢ)²

        This is exact for star-shaped curves and immune to self-intersection
        because it integrates r² directly — no shoelace double-counting.
        """
        r = self.radii()  # (T,)
        return (np.pi / self.n_sample) * (r * r).sum()
