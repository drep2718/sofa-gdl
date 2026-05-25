"""
Shape parameterization variants for Phase 1 experiments.

Each variant explores a different inductive bias or symmetry group:

  SofaShapeHighRes     — standard polar B-spline, more control points
  SofaShapeSymmetric   — Z/2Z bilateral symmetry enforced in B-spline parameters
  SofaShapeFourier     — trigonometric (Fourier) basis instead of B-spline
  SofaShapeFourierSym  — Fourier + Z/2Z: zero out sine modes, only cosines

GROUP THEORY NOTE:
  Gerver's sofa has a Z/2Z bilateral symmetry: it is mirror-symmetric about
  the angle bisector of the 90° corridor.  Encoding this symmetry directly
  in the parameterization implements the GDL principle of equivariance under
  the problem's symmetry group.  The optimizer then searches only the fixed-
  point subspace of the group action, which:
    (a) halves the effective parameter count,
    (b) eliminates asymmetric local minima,
    (c) biases toward the known optimal symmetry class.

  Fourier basis:
    B-spline basis functions have LOCAL support (each basis function is
    non-zero over only 4 knot spans).  Fourier modes have GLOBAL support
    (each mode contributes to all angles simultaneously).  For smooth,
    globally-structured boundaries like Gerver's sofa, the Fourier basis
    can represent the same shape quality with fewer parameters.

  Combining symmetry + Fourier:
    Z/2Z bilateral symmetry means r(α) = r(-α) = r(2π - α), i.e. r is an
    EVEN function of α.  An even periodic function has zero sine coefficients.
    Enforcing this sets b_k = 0 for all k, cutting parameters in half again.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .shape import SofaShape, _bspline_basis


# ── High-resolution B-spline (no structural change, just more knots) ──────────

class SofaShapeHighRes(SofaShape):
    """
    Standard polar B-spline with user-specified resolution.
    Exists mainly for clarity when building experiment tables.
    Inherits everything from SofaShape; alias with descriptive name.
    """
    pass   # no changes — SofaShape already accepts n_ctrl as argument


# ── Z/2Z symmetric B-spline ───────────────────────────────────────────────────

class SofaShapeSymmetric(nn.Module):
    """
    Polar B-spline with Z/2Z bilateral symmetry enforced by parameter tying.

    Symmetry: r(α) = r(2π - α)  (reflection about the α=0 axis)
    Implementation: parameterize only n_ctrl//2 control radii; mirror them
    before computing the B-spline interpolation.

    Why this symmetry?
      Gerver's sofa is symmetric about the angle bisector of the 90° corner.
      When the sofa is at rotation θ = π/4 (mid-turn), the shape looks
      identical under reflection about the y = x diagonal.  Expressing this
      in polar coordinates centered at the centroid corresponds to
      r(α) = r(π/2 - α) for centroid at (c, c), or equivalently
      r(α) = r(-α) after rotating the coordinate frame 45°.

      Enforcing this collapses the optimisation onto the Z/2Z fixed-point
      subspace — the exact subspace that contains the known optimal solution.
    """

    def __init__(
        self,
        n_ctrl:   int   = 64,
        n_sample: int   = 512,
        radius:   float = 0.35,
        cx0:      float = -0.5,
        cy0:      float =  0.5,
    ):
        super().__init__()
        assert n_ctrl % 2 == 0, "n_ctrl must be even for Z/2Z symmetry"
        self.n_ctrl   = n_ctrl
        self.n_sample = n_sample
        self.n_half   = n_ctrl // 2

        raw_init = float(np.log(np.exp(radius) - 1.0 + 1e-8))
        # Only n_ctrl//2 free parameters; the other half is a mirror copy
        self.raw_r_half = nn.Parameter(torch.full((self.n_half,), raw_init))
        self.centroid   = nn.Parameter(torch.tensor([cx0, cy0], dtype=torch.float32))

        t      = torch.linspace(0, 1, n_sample + 1)[:n_sample]
        alphas = torch.linspace(0, 2 * np.pi, n_sample + 1)[:n_sample]
        B      = _bspline_basis(t, n_ctrl)

        self.register_buffer("t",      t)
        self.register_buffer("alphas", alphas)
        self.register_buffer("B",      B)

    def _full_r_ctrl(self) -> torch.Tensor:
        """Expand half → full by mirroring: enforces r(α) = r(2π - α)."""
        r_half = F.softplus(self.raw_r_half)      # (n_half,) > 0
        return torch.cat([r_half, r_half.flip(0)]) # (n_ctrl,) — symmetric

    def radii(self) -> torch.Tensor:
        return self.B @ self._full_r_ctrl()         # (T,)

    def forward(self) -> torch.Tensor:
        r   = self.radii()
        cos = torch.cos(self.alphas)
        sin = torch.sin(self.alphas)
        x   = self.centroid[0] + r * cos
        y   = self.centroid[1] + r * sin
        return torch.stack([x, y], dim=1)

    def area(self) -> torch.Tensor:
        r = self.radii()
        return (np.pi / self.n_sample) * (r * r).sum()


# ── Fourier-basis polar shape ─────────────────────────────────────────────────

class SofaShapeFourier(nn.Module):
    """
    Polar shape where r(α) is a trigonometric polynomial (Fourier series).

    r(α) = softplus( c + Σ_{k=1}^{K} a_k cos(kα) + b_k sin(kα) )

    Fourier basis has GLOBAL support: each mode affects the entire boundary.
    This allows the optimizer to represent smooth, globally-structured shapes
    efficiently — Gerver's sofa is smooth everywhere, making it a natural
    fit for the Fourier basis.

    Parameter count: 1 + 2K  (vs. B-spline: n_ctrl)
    """

    def __init__(
        self,
        n_modes:  int   = 32,
        n_sample: int   = 512,
        radius:   float = 0.35,
        cx0:      float = -0.5,
        cy0:      float =  0.5,
    ):
        super().__init__()
        self.n_modes  = n_modes
        self.n_sample = n_sample

        # Initialise: constant term = radius, all harmonics = 0
        raw_c_init = float(np.log(np.exp(radius) - 1.0 + 1e-8))
        self.raw_c  = nn.Parameter(torch.tensor(raw_c_init))
        self.a_cos  = nn.Parameter(torch.zeros(n_modes))  # cosine coefficients
        self.b_sin  = nn.Parameter(torch.zeros(n_modes))  # sine  coefficients
        self.centroid = nn.Parameter(torch.tensor([cx0, cy0], dtype=torch.float32))

        alphas = torch.linspace(0, 2 * np.pi, n_sample + 1)[:n_sample]
        # Precompute Fourier basis matrix: (n_sample, n_modes)
        k      = torch.arange(1, n_modes + 1, dtype=torch.float32)
        C_mat  = torch.cos(alphas.unsqueeze(1) * k.unsqueeze(0))  # (T, K)
        S_mat  = torch.sin(alphas.unsqueeze(1) * k.unsqueeze(0))  # (T, K)

        self.register_buffer("alphas", alphas)
        self.register_buffer("C_mat",  C_mat)
        self.register_buffer("S_mat",  S_mat)

    def radii(self) -> torch.Tensor:
        """r(α) via Fourier synthesis, always positive via softplus."""
        f = F.softplus(self.raw_c) + self.C_mat @ self.a_cos + self.S_mat @ self.b_sin  # (T,)
        return F.softplus(f)  # double softplus keeps r > 0 even if sum dips negative

    def forward(self) -> torch.Tensor:
        r   = self.radii()
        cos = torch.cos(self.alphas)
        sin = torch.sin(self.alphas)
        x   = self.centroid[0] + r * cos
        y   = self.centroid[1] + r * sin
        return torch.stack([x, y], dim=1)

    def area(self) -> torch.Tensor:
        r = self.radii()
        return (np.pi / self.n_sample) * (r * r).sum()


# ── Fourier + Z/2Z symmetric ──────────────────────────────────────────────────

class SofaShapeFourierSym(nn.Module):
    """
    Fourier polar shape with Z/2Z symmetry enforced by zeroing sine modes.

    Bilateral symmetry r(α) = r(-α) ↔ r is EVEN ↔ all sine coefficients = 0.
    Only cosine modes are free parameters.

    This is the most constrained but also the most targeted parameterization:
    it directly encodes the known symmetry group of the optimal solution in
    the Fourier domain.  With the symmetry axis aligned, the Fourier cosine
    modes capture the shape deviation from a circle, layer by layer:
      k=1: elongation (elliptical deformation)
      k=2: squareness
      k=3: triangular deviation
      ...etc

    Parameter count: 1 + K  (vs. full Fourier: 1 + 2K)
    """

    def __init__(
        self,
        n_modes:  int   = 32,
        n_sample: int   = 512,
        radius:   float = 0.35,
        cx0:      float = -0.5,
        cy0:      float =  0.5,
    ):
        super().__init__()
        self.n_modes  = n_modes
        self.n_sample = n_sample

        raw_c_init = float(np.log(np.exp(radius) - 1.0 + 1e-8))
        self.raw_c  = nn.Parameter(torch.tensor(raw_c_init))
        self.a_cos  = nn.Parameter(torch.zeros(n_modes))  # only cosines!
        self.centroid = nn.Parameter(torch.tensor([cx0, cy0], dtype=torch.float32))

        alphas = torch.linspace(0, 2 * np.pi, n_sample + 1)[:n_sample]
        k      = torch.arange(1, n_modes + 1, dtype=torch.float32)
        C_mat  = torch.cos(alphas.unsqueeze(1) * k.unsqueeze(0))

        self.register_buffer("alphas", alphas)
        self.register_buffer("C_mat",  C_mat)

    def radii(self) -> torch.Tensor:
        f = F.softplus(self.raw_c) + self.C_mat @ self.a_cos
        return F.softplus(f)

    def forward(self) -> torch.Tensor:
        r   = self.radii()
        cos = torch.cos(self.alphas)
        sin = torch.sin(self.alphas)
        x   = self.centroid[0] + r * cos
        y   = self.centroid[1] + r * sin
        return torch.stack([x, y], dim=1)

    def area(self) -> torch.Tensor:
        r = self.radii()
        return (np.pi / self.n_sample) * (r * r).sum()
