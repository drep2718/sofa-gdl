"""
Phase 1 optimization: find the largest 2D sofa that fits around the corner.

Algorithm: Augmented Lagrangian Method (ALM)
  Outer: increase λ when max penetration exceeds tolerance
  Inner: Adam gradient descent on  -Area + λ·collision_penalty

Constraint formulation: at each angle θ the shape is rotated by θ AND given
the canonical translation that hugs the inner corner (see collision.py).
This is the correct moving-sofa formulation; it can reach Gerver's area ≈ 2.2195.

Expected result: area converging toward ~2.2.
"""

import os
import torch
import numpy as np
from .shape import SofaShape
from .collision import constraint_penalty, max_penetration


def optimize_sofa(
    n_ctrl:       int   = 64,
    n_sample:     int   = 512,
    n_angles:     int   = 91,
    n_outer:      int   = 30,
    n_inner:      int   = 600,
    lr:           float = 2e-3,
    lambda_init:  float = 5.0,
    lambda_scale: float = 2.0,
    lambda_max:   float = 500.0,
    verbose:      bool  = True,
) -> SofaShape:
    """
    Run augmented Lagrangian optimization for the moving sofa problem.

    With canonical translation baked into constraint_penalty, the centroid
    in the sofa frame is free — its absolute position doesn't affect feasibility
    (the translator corrects it at each angle).  We keep centroid as a learnable
    parameter for the visualizer but don't regularize it.
    """
    device = torch.device("cpu")

    sofa   = SofaShape(n_ctrl=n_ctrl, n_sample=n_sample, radius=0.35).to(device)
    thetas = torch.linspace(0, np.pi / 2, n_angles, dtype=torch.float32)

    lam = lambda_init

    for outer in range(n_outer):
        optimizer = torch.optim.Adam(sofa.parameters(), lr=lr)

        for inner in range(n_inner):
            optimizer.zero_grad()

            pts  = sofa()
            area = sofa.area()
            pen  = constraint_penalty(pts, thetas)

            loss = -area + lam * pen
            loss.backward()
            optimizer.step()

            if verbose and inner % 100 == 0:
                print(
                    f"  outer={outer:2d} inner={inner:3d} | "
                    f"area={area.item():.4f} | "
                    f"pen={pen.item():.6f} | "
                    f"λ={lam:.2f}"
                )

        with torch.no_grad():
            pts     = sofa()
            max_pen = max_penetration(pts, thetas)

        if verbose:
            print(
                f"Outer {outer:2d} done | area={sofa.area().item():.4f} | "
                f"max_pen={max_pen:.6f} | λ={lam:.2f}"
            )

        if max_pen > 1e-3:
            lam = min(lam * lambda_scale, lambda_max)
        else:
            if verbose:
                print("  Constraint satisfied — refining with fixed λ")

    return sofa


if __name__ == "__main__":
    print("Running Phase 1: 2D Moving Sofa Optimization (polar + canonical translation)")
    print("=" * 70)
    sofa = optimize_sofa(verbose=True)

    with torch.no_grad():
        pts     = sofa()
        area    = sofa.area().item()
        thetas  = torch.linspace(0, np.pi / 2, 91)
        max_pen = max_penetration(pts, thetas)
        cx, cy  = sofa.centroid.tolist()

    print()
    print(f"Final area:          {area:.4f}")
    print(f"Gerver reference:    2.2195")
    print(f"Gap:                 {2.2195 - area:.4f}")
    print(f"Max penetration:     {max_pen:.6f}")
    print(f"Centroid:            ({cx:.3f}, {cy:.3f})")
    print()

    os.makedirs("phase1_2d/results", exist_ok=True)
    torch.save(
        {
            "raw_r":    sofa.raw_r.detach(),
            "centroid": sofa.centroid.detach(),
            "area":     area,
        },
        "phase1_2d/results/best_sofa.pt",
    )
    print("Saved to phase1_2d/results/best_sofa.pt")
    print("Run visualize.py to see the result.")
