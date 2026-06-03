"""
Phase 1 tests: polar shape, SDF, and optimization primitives.

Run from project root:
    python -m pytest tests/test_phase1.py -v
"""

import pytest
import torch
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase1_2d.shape import SofaShape, _bspline_basis
from phase1_2d.collision import (
    rotation_matrix_2d,
    hallway_sdf_2d,
    constraint_penalty,
    max_penetration,
)


# ── _bspline_basis ─────────────────────────────────────────────────────────────

class TestBSplineBasis:
    def test_rows_sum_to_one(self):
        """Partition of unity: every row must sum to 1."""
        t = torch.linspace(0, 1, 200)[:-1]
        B = _bspline_basis(t, n_ctrl=16)
        row_sums = B.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_output_shape(self):
        # linspace(0,1,100)[:-1] drops the duplicate endpoint → 99 rows
        t = torch.linspace(0, 1, 100)[:-1]
        B = _bspline_basis(t, n_ctrl=32)
        assert B.shape == (99, 32)

    def test_nonnegative(self):
        """B-spline basis is always ≥ 0 — required for the polar area guarantee."""
        t = torch.linspace(0, 1, 500)
        B = _bspline_basis(t, n_ctrl=20)
        assert B.min().item() >= -1e-6


# ── SofaShape (polar parameterization) ────────────────────────────────────────

class TestSofaShape:
    def test_forward_shape(self):
        sofa = SofaShape(n_ctrl=32, n_sample=256)
        pts  = sofa()
        assert pts.shape == (256, 2)

    def test_area_positive(self):
        sofa = SofaShape(n_ctrl=32, n_sample=256, radius=0.35)
        assert sofa.area().item() > 0

    def test_area_circle_approx(self):
        """
        Polar area ≈ π r² for a uniform-radius circle.

        The polar area formula ∫₀²π (1/2) r² dα is exact for a circle of
        constant radius r: A = π r².
        """
        r    = 0.4
        sofa = SofaShape(n_ctrl=128, n_sample=2048, radius=r)
        area = sofa.area().item()
        expected = np.pi * r**2
        assert abs(area - expected) < 0.005, \
            f"Polar circle area {area:.5f} ≠ π r²={expected:.5f}"

    def test_area_differentiable(self):
        """
        Polar area depends on raw_r (the radii), not on centroid.
        Centroid gradient comes through constraint_penalty which uses forward().
        """
        sofa   = SofaShape(n_ctrl=16, n_sample=64)
        thetas = torch.linspace(0, np.pi / 2, 5)

        # Area grad flows to raw_r
        sofa.area().backward()
        assert sofa.raw_r.grad is not None
        assert not torch.isnan(sofa.raw_r.grad).any()

        # Centroid grad flows through collision penalty (uses forward() = centroid + r*dir)
        sofa.raw_r.grad = None
        sofa.centroid.grad = None
        constraint_penalty(sofa(), thetas).backward()
        assert sofa.centroid.grad is not None

    def test_radii_always_positive(self):
        """
        Core invariant: softplus(raw_r) > 0 and spline thereof > 0.
        This guarantees the polar curve never self-intersects.
        """
        sofa = SofaShape(n_ctrl=64, n_sample=512)
        # Push raw_r very negative to stress-test softplus
        with torch.no_grad():
            sofa.raw_r.fill_(-20.0)
        r = sofa.radii()
        assert (r > 0).all(), "Radii must be strictly positive"

    def test_scale_invariance(self):
        """Doubling radius should quadruple area (polar area ∝ r²)."""
        s1 = SofaShape(n_ctrl=64, n_sample=1024, radius=0.3)
        s2 = SofaShape(n_ctrl=64, n_sample=1024, radius=0.6)
        ratio = s2.area().item() / s1.area().item()
        assert abs(ratio - 4.0) < 0.05, f"Area ratio {ratio:.3f}, expected 4.0"

    def test_area_bounded_by_pi(self):
        """
        Physical sanity: a valid sofa must have area ≤ π (unit-disk bound).
        The initial shape should be well within this.
        """
        sofa = SofaShape(n_ctrl=64, n_sample=512, radius=0.35)
        assert sofa.area().item() < np.pi, \
            "Initial shape already violates theoretical area ceiling"

    def test_centroid_is_parameter(self):
        """Centroid must be a learnable parameter so Adam can move the shape."""
        sofa = SofaShape()
        param_names = [n for n, _ in sofa.named_parameters()]
        assert "centroid" in param_names
        assert "raw_r"    in param_names


# ── rotation_matrix_2d ─────────────────────────────────────────────────────────

class TestRotationMatrix:
    def test_zero_angle_is_identity(self):
        R = rotation_matrix_2d(torch.tensor(0.0))
        assert torch.allclose(R, torch.eye(2), atol=1e-6)

    def test_ninety_degrees(self):
        R = rotation_matrix_2d(torch.tensor(np.pi / 2))
        assert torch.allclose(R, torch.tensor([[0., -1.], [1., 0.]]), atol=1e-6)

    def test_is_orthogonal(self):
        for deg in [0, 30, 45, 60, 90]:
            R = rotation_matrix_2d(torch.tensor(deg * np.pi / 180))
            assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-6), \
                f"Not orthogonal at {deg}°"

    def test_determinant_one(self):
        R   = rotation_matrix_2d(torch.tensor(1.23))
        det = R[0,0]*R[1,1] - R[0,1]*R[1,0]
        assert abs(det.item() - 1.0) < 1e-6


# ── hallway_sdf_2d ─────────────────────────────────────────────────────────────

class TestHallawaySDF:
    def test_inside_vertical_corridor(self):
        pts = torch.tensor([[-0.5, 1.5], [-0.1, 0.5], [-0.9, -1.0]])
        assert (hallway_sdf_2d(pts) < 0).all()

    def test_inside_horizontal_corridor(self):
        pts = torch.tensor([[-1.5, 0.5], [-2.0, 0.8], [-0.3, 0.2]])
        assert (hallway_sdf_2d(pts) < 0).all()

    def test_outside_hallway(self):
        """
        Points past the inner wall (x > 0) and not in any corridor.
        (0.5, 0.5): x > 0 so not in vertical corridor;
                    x > 0 so not in horizontal corridor either.
        """
        pts = torch.tensor([[1.0, 1.0], [0.5, 0.5], [-1.5, -1.0]])
        assert (hallway_sdf_2d(pts) > 0).all(), \
            f"SDF: {hallway_sdf_2d(pts).tolist()}"

    def test_at_inner_corner(self):
        """Inner corner (0,0) is on the hallway boundary → SDF ≈ 0."""
        pt  = torch.tensor([[0.0, 0.0]])
        sdf = hallway_sdf_2d(pt)
        assert abs(sdf.item()) < 1e-5

    def test_gradient_flows(self):
        pts = torch.tensor([[0.5, 0.5]], requires_grad=True)
        hallway_sdf_2d(pts).sum().backward()
        assert pts.grad is not None


# ── constraint_penalty ─────────────────────────────────────────────────────────

class TestConstraintPenalty:
    def test_zero_for_tiny_circle_near_corner(self):
        """
        A tiny circle near (-0.1, 0.1) stays in the vertical corridor
        at all rotation angles θ ∈ [0, π/2]:
          R(θ)·(-0.1, 0.1) has x ≤ 0 and magnitude < 0.15 < 1.
        """
        sofa = SofaShape(n_ctrl=32, n_sample=256, radius=0.05, cx0=-0.1, cy0=0.1)
        thetas = torch.linspace(0, np.pi / 2, 10)
        pen    = constraint_penalty(sofa(), thetas)
        assert pen.item() < 1e-4, f"Tiny circle penalty={pen.item():.2e}"

    def test_positive_for_large_shape(self):
        """A radius-2 shape can't fit in a width-1 hallway."""
        sofa   = SofaShape(n_ctrl=32, n_sample=256, radius=2.0)
        thetas = torch.linspace(0, np.pi / 2, 10)
        assert constraint_penalty(sofa(), thetas).item() > 0

    def test_penalty_differentiable(self):
        sofa   = SofaShape(n_ctrl=16, n_sample=64, radius=0.3)
        thetas = torch.linspace(0, np.pi / 2, 5)
        constraint_penalty(sofa(), thetas).backward()
        assert sofa.raw_r.grad is not None

    def test_penalty_decreases_with_shrinking(self):
        """Smaller shapes have ≤ penalty than larger ones."""
        t = torch.linspace(0, np.pi / 2, 10)
        p_big   = constraint_penalty(SofaShape(n_ctrl=32, n_sample=128, radius=0.8)(), t).item()
        p_small = constraint_penalty(SofaShape(n_ctrl=32, n_sample=128, radius=0.2)(), t).item()
        assert p_small <= p_big


# ── integration: one Adam step ─────────────────────────────────────────────────

class TestOptimizationStep:
    def test_no_nans_after_step(self):
        """One Adam step must not produce NaN vertices or area."""
        sofa = SofaShape(n_ctrl=32, n_sample=128, radius=0.3)
        opt  = torch.optim.Adam(sofa.parameters(), lr=1e-3)
        thetas = torch.linspace(0, np.pi / 2, 10)

        opt.zero_grad()
        (-sofa.area() + 1.0 * constraint_penalty(sofa(), thetas)).backward()
        opt.step()

        assert not torch.isnan(sofa().detach()).any()
        assert sofa.area().item() > 0

    def test_area_stays_below_pi(self):
        """
        After one step, area must remain below π (theoretical ceiling).
        This is the key regression test catching the self-intersection bug.
        """
        sofa = SofaShape(n_ctrl=32, n_sample=128, radius=0.3)
        opt  = torch.optim.Adam(sofa.parameters(), lr=1e-3)
        thetas = torch.linspace(0, np.pi / 2, 10)

        for _ in range(20):
            opt.zero_grad()
            (-sofa.area() + constraint_penalty(sofa(), thetas)).backward()
            opt.step()

        assert sofa.area().item() < np.pi, \
            f"Area {sofa.area().item():.4f} exceeded theoretical max π={np.pi:.4f}"
