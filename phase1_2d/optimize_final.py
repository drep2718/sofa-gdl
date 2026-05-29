"""
Phase 1 final push: warm-start from best checkpoint + higher resolution.

Strategy:
  1. Load best result (area≈2.075, n_ctrl=64) from best_sofa_improved.pt
  2. Upsample control points to n_ctrl=128 by spline interpolation
  3. Continue optimising with more outer loops and denser angle sampling

Why this should push higher:
  - n_ctrl=64 was still improving at outer=29 — not converged
  - n_ctrl=128 adds expressiveness to represent the boundary more precisely
  - Warm-start bypasses the slow initial growth phase (1.1 → 2.0)
  - Denser angles (n_angles=361) reduces the train/test generalisation gap

Usage:
    python -m phase1_2d.optimize_final
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

from .shape import SofaShape, _bspline_basis
from .collision import constraint_penalty, max_penetration


# ── Cosine-annealed Adam (same as optimize_improved) ─────────────────────────

def cosine_lr(optimizer, step, n_steps, lr_max, lr_min):
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / n_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Warm-start: upsample n_ctrl=64 → n_ctrl_new ──────────────────────────────

def upsample_sofa(sofa_old: SofaShape, n_ctrl_new: int, n_sample_new: int) -> SofaShape:
    """
    Upsample control radii from n_ctrl_old to n_ctrl_new using linear interpolation
    in the t-domain, giving the new shape a fine-resolution starting point.
    """
    with torch.no_grad():
        r_old = F.softplus(sofa_old.raw_r).numpy()        # (n_ctrl_old,)
        n_old = sofa_old.n_ctrl

        # Interpolate uniformly in [0, 1) — periodic wrap
        t_old = np.linspace(0, 1, n_old, endpoint=False)
        t_new = np.linspace(0, 1, n_ctrl_new, endpoint=False)

        # Wrap-safe interpolation: duplicate first point at end for continuity
        t_wrap = np.append(t_old, 1.0)
        r_wrap = np.append(r_old, r_old[0])
        r_new  = np.interp(t_new, t_wrap, r_wrap)         # (n_ctrl_new,)

        # Invert softplus: raw = log(exp(r) - 1)
        raw_r_new = np.log(np.exp(r_new) - 1.0 + 1e-8).astype(np.float32)

        centroid_val = sofa_old.centroid.numpy()

    sofa_new = SofaShape(
        n_ctrl=n_ctrl_new,
        n_sample=n_sample_new,
        radius=0.35,              # ignored — raw_r overwritten below
        cx0=float(centroid_val[0]),
        cy0=float(centroid_val[1]),
    )
    with torch.no_grad():
        sofa_new.raw_r.copy_(torch.from_numpy(raw_r_new))

    return sofa_new


# ── Optimisation run (same ALM as optimize_improved) ─────────────────────────

def run_one(
    sofa,
    n_angles:     int   = 181,
    n_outer:      int   = 40,
    n_inner:      int   = 600,
    lr_max:       float = 2e-3,
    lr_min:       float = 3e-4,
    lambda_init:  float = 5.0,
    lambda_scale: float = 2.0,
    lambda_max:   float = 500.0,
    verbose:      bool  = True,
):
    thetas = torch.linspace(0, np.pi / 2, n_angles)
    lam    = lambda_init

    for outer in range(n_outer):
        optimizer = torch.optim.Adam(sofa.parameters(), lr=lr_max)
        for inner in range(n_inner):
            cosine_lr(optimizer, inner, n_inner, lr_max, lr_min)
            optimizer.zero_grad()
            loss = -sofa.area() + lam * constraint_penalty(sofa(), thetas)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            mp = max_penetration(sofa(), thetas)

        if verbose:
            print(f"  outer={outer:2d} area={sofa.area().item():.4f} "
                  f"max_pen={mp:.5f} λ={lam:.1f}", flush=True)

        if mp > 1e-3:
            lam = min(lam * lambda_scale, lambda_max)

    with torch.no_grad():
        final_mp = max_penetration(sofa(), thetas)
    return sofa.area().item(), final_mp


# ── Main ──────────────────────────────────────────────────────────────────────

def optimize_final(
    n_ctrl:    int   = 128,
    n_sample:  int   = 1024,
    n_angles:  int   = 181,
    n_starts:  int   = 3,
    n_outer:   int   = 60,
    n_inner:   int   = 600,
    verbose:   bool  = True,
):
    """
    Multi-start run at higher resolution (n_ctrl=128, n_outer=60).
    Uses the same clean ALM as the successful 64-ctrl run.
    No warm-starting — avoids the lambda-overshoot trap.
    """
    gerver = 2.2195
    os.makedirs("phase1_2d/results", exist_ok=True)

    print(f"\n{'='*65}")
    print(f" Final push — n_ctrl={n_ctrl}, {n_starts} starts × {n_outer} outer × {n_inner} inner")
    print(f" n_angles={n_angles}   Gerver reference: {gerver}")
    print(f"{'='*65}\n")

    start_radii = [0.25, 0.30, 0.20][:n_starts]
    best_area   = -1.0
    best_sofa   = None
    best_pen    = float("inf")
    results     = []

    for seed in range(n_starts):
        torch.manual_seed(seed * 17 + 3)
        radius = start_radii[seed]

        sofa = SofaShape(n_ctrl=n_ctrl, n_sample=n_sample,
                         radius=radius, cx0=-0.5, cy0=0.5)
        with torch.no_grad():
            sofa.raw_r.add_(torch.randn_like(sofa.raw_r) * 0.05)

        print(f"[Start {seed+1}/{n_starts}] radius={radius:.2f}  seed={seed}", flush=True)

        area, max_pen = run_one(
            sofa,
            n_angles=n_angles, n_outer=n_outer, n_inner=n_inner,
            lr_max=3e-3, lr_min=5e-4,
            lambda_init=5.0, lambda_scale=2.0, lambda_max=500.0,
            verbose=verbose,
        )

        feasible = max_pen < 1e-2
        status   = "✓ feasible" if feasible else "✗ infeasible"
        print(f"  → area={area:.4f}  gap={gerver-area:.4f}  "
              f"max_pen={max_pen:.5f}  {status}\n", flush=True)
        results.append((area, max_pen, feasible, sofa))

        if feasible and area > best_area:
            best_area = area
            best_sofa = sofa
            best_pen  = max_pen

    if best_sofa is None:
        best_area, best_pen, _, best_sofa = max(results, key=lambda x: x[0])

    print(f"{'─'*65}")
    print(f" Best feasible area: {best_area:.6f}")
    print(f" Gap to Gerver:      {gerver - best_area:.6f}  "
          f"({best_area/gerver*100:.2f}% of Gerver)")
    print(f" Max penetration:    {best_pen:.6f}")
    print(f"{'─'*65}\n")

    save_path = "phase1_2d/results/best_sofa_final.pt"
    with torch.no_grad():
        torch.save(
            {"raw_r":    best_sofa.raw_r.detach(),
             "centroid": best_sofa.centroid.detach(),
             "area":     best_area,
             "n_ctrl":   n_ctrl},
            save_path,
        )
    print(f"Saved → {save_path}")
    return best_sofa, best_area, best_pen


if __name__ == "__main__":
    import time
    t0 = time.time()
    optimize_final()
    print(f"\nTotal time: {time.time()-t0:.0f}s")
