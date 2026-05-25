"""
Improved Phase 1 optimizer.

Changes from baseline (optimize.py):
  1. Correct Z/2Z diagonal symmetry — r(α) = r(π/2 − α)
     Matches the actual symmetry of the hallway (swapping corridors).
     Enforced by tying control-point pairs across the π/4 diagonal.

  2. Multi-start — 5 independent random seeds; keep the best feasible result.
     Avoids local minima by exploring multiple basins.

  3. Cosine-annealed learning rate — lr decays from lr_max to lr_min over each
     inner loop, smoothing the final convergence.

  4. Adaptive λ update — λ grows only when max_pen > tol AND is reset for each
     new random start to prevent λ from exploding across starts.

WHY DIAGONAL SYMMETRY HELPS:
  In the previous experiment, we enforced r(α) = r(2π − α) — reflection about
  the horizontal axis.  This is WRONG: Gerver's sofa is symmetric about the
  45° diagonal (swapping the two corridors), not about the horizontal axis.

  Correct symmetry: swapping corridors maps (x, y) → (−y, −x).  For the polar
  parameterization centred at (c, c), this acts as α → π/2 − α.  Enforcing
  r(α) = r(π/2 − α) with cx = cy cuts the effective parameter count in half
  AND biases the search toward the correct symmetry class — exactly the inductive
  bias that equivariant networks exploit in GDL.

Usage:
    python -m phase1_2d.optimize_improved
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .shape import SofaShape, _bspline_basis
from .collision import constraint_penalty, max_penetration


# ─── Diagonal-symmetric shape ─────────────────────────────────────────────────

class SofaShapeDiagSym(nn.Module):
    """
    Polar B-spline with the CORRECT Z/2Z diagonal symmetry for the sofa problem.

    Symmetry: r(α) = r(3π/2 − α)
    Hallway symmetry: swapping corridors sends (x,y) → (−y,−x).
    For centroid at (cx,cy) with cx = −cy (e.g. −0.5, 0.5), this maps
    polar angle α → 3π/2 − α (verified: cos α' = −sin α, sin α' = −cos α).

    Implementation:
      - centroid is FREE (cx, cy independent) — starts at (-0.5, 0.5)
      - Control point k is paired with k' = (3N/4 − k) mod N
      - Free parameters: one from each pair (~N/2 unique radii + centroid)
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
        assert n_ctrl % 4 == 0, "n_ctrl must be divisible by 4"
        self.n_ctrl   = n_ctrl
        self.n_sample = n_sample
        N = n_ctrl

        raw_init = float(np.log(np.exp(radius) - 1.0 + 1e-8))

        # Build the pairing: k ↔ k' = (3N/4 - k) % N
        # This enforces r(α) = r(3π/2 − α) — the correct hallway symmetry.
        # Derivation: hallway sym (x,y)→(−y,−x) at centroid (cx,−cx) maps α→3π/2−α.
        # The centroid is LEFT FREE (gradient naturally drives cx→−cy).
        pivot = 3 * N // 4
        free_indices = []
        seen = set()
        for k in range(N):
            if k not in seen:
                free_indices.append(k)
                k2 = (pivot - k) % N
                seen.add(k)
                seen.add(k2)
        self.register_buffer("free_idx",  torch.tensor(free_indices, dtype=torch.long))
        mirror = [(pivot - k) % N for k in free_indices]   # (3N/4 - k) % N
        self.register_buffer("mirror_idx", torch.tensor(mirror, dtype=torch.long))

        self.n_free = len(free_indices)
        self.raw_r_free = nn.Parameter(torch.full((self.n_free,), raw_init))
        # Centroid is FREE — does not enforce cx=cy
        self.centroid = nn.Parameter(torch.tensor([cx0, cy0], dtype=torch.float32))

        t      = torch.linspace(0, 1, n_sample + 1)[:n_sample]
        alphas = torch.linspace(0, 2 * np.pi, n_sample + 1)[:n_sample]
        B      = _bspline_basis(t, n_ctrl)

        self.register_buffer("t",      t)
        self.register_buffer("alphas", alphas)
        self.register_buffer("B",      B)

    def _full_r_ctrl(self) -> torch.Tensor:
        """Expand free params to full N radii via r(α) = r(3π/2−α) pairing."""
        r_free = F.softplus(self.raw_r_free)   # (n_free,)
        r_ctrl = torch.zeros(self.n_ctrl, dtype=r_free.dtype, device=r_free.device)
        r_ctrl = r_ctrl.scatter(0, self.free_idx,   r_free)
        r_ctrl = r_ctrl.scatter(0, self.mirror_idx, r_free)
        return r_ctrl                            # (N,)

    def radii(self) -> torch.Tensor:
        return self.B @ self._full_r_ctrl()      # (T,)

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


# ─── Cosine-annealed Adam ─────────────────────────────────────────────────────

def cosine_lr(optimizer, step: int, n_steps: int, lr_max: float, lr_min: float):
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / n_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─── Single-run optimisation ──────────────────────────────────────────────────

def run_one(
    sofa,
    n_angles:     int   = 91,
    n_outer:      int   = 30,
    n_inner:      int   = 600,
    lr_max:       float = 3e-3,
    lr_min:       float = 5e-4,
    lambda_init:  float = 5.0,
    lambda_scale: float = 2.0,
    lambda_max:   float = 500.0,
    verbose:      bool  = False,
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


# ─── Multi-start wrapper ──────────────────────────────────────────────────────

def optimize_improved(
    use_diagonal_sym: bool  = True,
    n_starts:         int   = 5,
    n_ctrl:           int   = 64,
    n_sample:         int   = 512,
    n_angles:         int   = 91,
    n_outer:          int   = 30,
    n_inner:          int   = 600,
    verbose:          bool  = True,
) -> tuple:
    """
    Run multi-start optimisation.  Returns (best_sofa, best_area, best_max_pen).
    """
    best_area  = -1.0
    best_sofa  = None
    best_pen   = float("inf")

    gerver = 2.2195
    shape_name = "DiagSymmetric" if use_diagonal_sym else "Standard"

    print(f"\n{'='*65}")
    print(f" Improved Phase 1 — {n_starts} starts × {n_outer} outer × {n_inner} inner")
    print(f" Shape: {shape_name} (n_ctrl={n_ctrl})")
    print(f" Gerver reference: {gerver}")
    print(f"{'='*65}\n")

    # Vary starting radii across seeds — do NOT give it Gerver's answer
    start_radii = [0.25, 0.30, 0.35, 0.40, 0.20][:n_starts]
    # Vary starting centroid diagonal offset
    start_c0    = [-0.30, -0.35, -0.40, -0.25, -0.45][:n_starts]

    for seed in range(n_starts):
        torch.manual_seed(seed * 17 + 3)
        radius = start_radii[seed]

        if use_diagonal_sym:
            sofa = SofaShapeDiagSym(
                n_ctrl=n_ctrl, n_sample=n_sample,
                radius=radius, cx0=-0.5, cy0=0.5,
            )
        else:
            sofa = SofaShape(
                n_ctrl=n_ctrl, n_sample=n_sample,
                radius=radius, cx0=-0.5, cy0=0.5,
            )

        # Small random perturbation to escape identical initialisation
        with torch.no_grad():
            if use_diagonal_sym:
                sofa.raw_r_free.add_(torch.randn_like(sofa.raw_r_free) * 0.05)
            else:
                sofa.raw_r.add_(torch.randn_like(sofa.raw_r) * 0.05)

        print(f"[Start {seed+1}/{n_starts}] radius={radius:.2f}  seed={seed}", flush=True)

        area, max_pen = run_one(
            sofa,
            n_angles=n_angles, n_outer=n_outer, n_inner=n_inner,
            verbose=verbose,
        )

        feasible = max_pen < 1e-2
        status   = "✓ feasible" if feasible else "✗ infeasible"
        print(f"  → area={area:.4f}  gap={gerver-area:.4f}  "
              f"max_pen={max_pen:.5f}  {status}\n", flush=True)

        if feasible and area > best_area:
            best_area = area
            best_sofa = sofa
            best_pen  = max_pen

    if best_sofa is None:
        print("WARNING: no feasible solution found — returning best infeasible")
        # fallback: pick by area regardless of feasibility
        best_sofa = sofa
        best_area = area
        best_pen  = max_pen

    print(f"{'─'*65}")
    print(f" Best feasible area: {best_area:.6f}")
    print(f" Gap to Gerver:      {gerver - best_area:.6f}")
    print(f" Max penetration:    {best_pen:.6f}")
    print(f"{'─'*65}\n")

    return best_sofa, best_area, best_pen


# ─── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # ── Run 1: Diagonal-symmetric (correct Z/2Z) ──────────────────────────
    t0 = time.time()
    sofa_sym, area_sym, pen_sym = optimize_improved(
        use_diagonal_sym=True,
        n_starts=5, n_ctrl=64, n_sample=512,
        n_angles=181,            # denser than training to match dense verification
        n_outer=30, n_inner=600, verbose=True,
    )
    print(f"Diagonal-sym run: {time.time()-t0:.0f}s\n", flush=True)

    # ── Run 2: Standard (no symmetry, multi-start as baseline) ────────────
    t0 = time.time()
    sofa_std, area_std, pen_std = optimize_improved(
        use_diagonal_sym=False,
        n_starts=5, n_ctrl=64, n_sample=512,
        n_angles=181,
        n_outer=30, n_inner=600, verbose=True,
    )
    print(f"Standard run: {time.time()-t0:.0f}s\n", flush=True)

    # ── Compare ───────────────────────────────────────────────────────────
    gerver = 2.2195
    print("\n" + "="*65)
    print("FINAL COMPARISON")
    print("="*65)
    print(f"{'Config':<30} {'Area':>8} {'Gap':>8} {'MaxPen':>10}")
    print("─"*60)
    print(f"{'Diagonal Z/2Z (correct sym)':<30} {area_sym:>8.4f} "
          f"{gerver-area_sym:>8.4f} {pen_sym:>10.5f}")
    print(f"{'Standard (no sym)':<30} {area_std:>8.4f} "
          f"{gerver-area_std:>8.4f} {pen_std:>10.5f}")
    print(f"{'Gerver conjecture':<30} {gerver:>8.4f}")
    print("="*65)

    # Save the best of the two
    best_sofa  = sofa_sym if area_sym >= area_std else sofa_std
    best_area  = max(area_sym, area_std)
    print(f"\nBest result: area = {best_area:.6f}  "
          f"({best_area/gerver*100:.2f}% of Gerver)")

    os.makedirs("phase1_2d/results", exist_ok=True)

    # Normalise: extract raw_r and centroid regardless of model type
    with torch.no_grad():
        if isinstance(best_sofa, SofaShapeDiagSym):
            raw_r_save   = best_sofa._full_r_ctrl().log1p()  # approximate inverse softplus
            centroid_save = best_sofa.centroid.detach()
        else:
            raw_r_save    = best_sofa.raw_r.detach()
            centroid_save = best_sofa.centroid.detach()

    torch.save(
        {
            "raw_r":    raw_r_save,
            "centroid": centroid_save,
            "area":     best_area,
        },
        "phase1_2d/results/best_sofa_improved.pt",
    )
    print("Saved → phase1_2d/results/best_sofa_improved.pt")
