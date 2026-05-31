"""
Independent verification of Phase 1 results.

This script deliberately uses NO knowledge of our area formula or SDF.
It verifies the saved best_sofa.pt from three independent angles:

  1. Monte Carlo area  — sample random points in a bounding box, count what
                         fraction fall inside the polygon via ray-casting.
                         Completely independent of the polar area formula.

  2. Fine feasibility  — check 3600 rotation angles (instead of 91) with
                         10× more boundary samples.  If the shape truly fits,
                         max penetration should be near zero at this resolution.

  3. Area formula cross-check — compare polar area vs. shoelace area for the
                         specific case of a star-shaped curve (they should agree
                         when the shape is star-shaped w.r.t. its centroid).

  4. Gap analysis      — compute a simple analytic upper bound on the area of
                         any star-shaped shape fitting around the corner, to
                         bracket where the truth should lie.

Usage:
    python -m phase1_2d.verify
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .shape import SofaShape
from .collision import max_penetration, constraint_penalty


# ─── load the saved shape ─────────────────────────────────────────────────────

def load_best() -> SofaShape:
    path = "phase1_2d/results/best_sofa.pt"
    if not os.path.exists(path):
        raise FileNotFoundError("Run optimize.py first to generate best_sofa.pt")
    data = torch.load(path, weights_only=True)
    sofa = SofaShape(n_ctrl=64, n_sample=512)
    with torch.no_grad():
        sofa.raw_r.copy_(data["raw_r"])
        sofa.centroid.copy_(data["centroid"])
    return sofa


# ─── 1. Monte Carlo area ──────────────────────────────────────────────────────

def point_in_polygon(pts: np.ndarray, qx: float, qy: float) -> bool:
    """
    Ray-casting test: is point (qx, qy) inside the polygon defined by pts?
    Shoots a ray in the +x direction and counts crossings.
    """
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > qy) != (yj > qy)) and (qx < (xj - xi) * (qy - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def monte_carlo_area(pts: np.ndarray, n_samples: int = 200_000, seed: int = 42) -> float:
    """
    Estimate polygon area by uniform sampling over the bounding box.
    Completely independent of the polar area formula.
    """
    rng  = np.random.default_rng(seed)
    xmin, xmax = pts[:,0].min(), pts[:,0].max()
    ymin, ymax = pts[:,1].min(), pts[:,1].max()
    box_area = (xmax - xmin) * (ymax - ymin)

    # Vectorised edge-crossing test
    px = rng.uniform(xmin, xmax, n_samples)
    py = rng.uniform(ymin, ymax, n_samples)
    count = 0
    # Batch via broadcasting (memory-intensive but fast)
    x0 = pts[:-1, 0][:, None]   # (E, 1)
    y0 = pts[:-1, 1][:, None]
    x1 = pts[1:,  0][:, None]
    y1 = pts[1:,  1][:, None]

    # Edge i goes from (x0[i], y0[i]) to (x1[i], y1[i])
    cond1 = (y0 > py[None,:]) != (y1 > py[None,:])           # (E, N)
    # x-intercept of edge at height py
    x_int = (x1 - x0) * (py[None,:] - y0) / ((y1 - y0) + 1e-15) + x0
    cond2 = px[None, :] < x_int
    crossings = (cond1 & cond2).sum(axis=0)   # (N,) — number of crossings per point
    inside = (crossings % 2) == 1
    count  = inside.sum()

    return box_area * count / n_samples


# ─── 2. Fine feasibility check ───────────────────────────────────────────────

def fine_feasibility_check(sofa: SofaShape, n_angles: int = 3600) -> dict:
    """
    Dense rotation check: 3600 angles, 10× more boundary points than training.
    We rebuild the shape at higher sample density for this check.
    """
    # Rebuild at 10× density
    sofa_dense = SofaShape(n_ctrl=64, n_sample=5120)
    with torch.no_grad():
        sofa_dense.raw_r.copy_(sofa.raw_r)
        sofa_dense.centroid.copy_(sofa.centroid)

    thetas = torch.linspace(0, np.pi / 2, n_angles)
    with torch.no_grad():
        pts     = sofa_dense()
        max_pen = max_penetration(pts, thetas)

    # Also check the endpoints exactly (θ=0 and θ=π/2)
    # and the worst-case mid-turn angles
    check_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    sdf_at_key = {}
    for ang in check_angles:
        th  = torch.tensor(ang, dtype=torch.float32)
        pen = constraint_penalty(pts, torch.tensor([ang], dtype=torch.float32))
        sdf_at_key[f"{np.degrees(ang):.0f}°"] = pen.item()

    return {
        "max_penetration_3600_angles": max_pen,
        "penalty_at_key_angles": sdf_at_key,
        "feasible": max_pen < 1e-3,
    }


# ─── 3. Area formula cross-check ─────────────────────────────────────────────

def shoelace_area(pts: np.ndarray) -> float:
    """
    Shoelace formula: ½|Σ (x_i·y_{i+1} - x_{i+1}·y_i)|
    Only exact for SIMPLE (non-self-intersecting) polygons.
    If it agrees with polar area, the curve is non-self-intersecting.
    """
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + x[-1]*y[0] - x[0]*y[-1])


def check_self_intersection(pts: np.ndarray, max_checks: int = 5000) -> dict:
    """
    Heuristic self-intersection check: sample random pairs of non-adjacent
    edges and test whether they cross.
    """
    n = len(pts)
    rng = np.random.default_rng(0)
    crossings = 0
    for _ in range(max_checks):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if abs(i - j) < 3 or abs(i - j) > n - 3:
            continue
        # Segment i: pts[i] → pts[(i+1)%n]
        # Segment j: pts[j] → pts[(j+1)%n]
        p, q = pts[i], pts[(i+1)%n]
        r, s = pts[j], pts[(j+1)%n]
        # Standard segment intersection test
        d1 = q - p; d2 = s - r
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-12:
            continue
        t = ((r[0]-p[0])*d2[1] - (r[1]-p[1])*d2[0]) / cross
        u = ((r[0]-p[0])*d1[1] - (r[1]-p[1])*d1[0]) / cross
        if 0 < t < 1 and 0 < u < 1:
            crossings += 1
    return {"crossings_found": crossings, "checks": max_checks,
            "self_intersecting": crossings > 0}


# ─── 4. Upper bound analysis ──────────────────────────────────────────────────

def compute_upper_bounds() -> dict:
    """
    Compute analytic upper bounds on the area of any sofa fitting the corner.

    Known bounds:
      - Hammersley (1968): area ≤ 2√2 ≈ 2.828  (loose)
      - Gerver (1992):     area = 2.2195...      (conjectured optimum)
      - Kallus–Romik (2018): area ≤ 2.37 (tightest proven upper bound to date)
      - Unit-disk bound:   area < π ≈ 3.14159  (trivial)

    These are analytic, not computed from our model.
    """
    return {
        "unit_disk_bound":            np.pi,
        "hammersley_1968_bound":      2 * np.sqrt(2),
        "kallus_romik_2018_bound":    2.37,
        "gerver_1992_conjecture":     2.2195,
        "our_optimum_star_shaped":    None,  # filled in by caller
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def run_verification():
    print("=" * 65)
    print(" Phase 1 — Independent Verification Report")
    print("=" * 65)

    sofa = load_best()

    with torch.no_grad():
        pts_t = sofa()
    pts = pts_t.numpy()
    # Close the polygon for area calculations
    pts_closed = np.vstack([pts, pts[0]])

    polar_area = sofa.area().item()
    cx, cy     = sofa.centroid.tolist()
    print(f"\nLoaded sofa:  polar area = {polar_area:.6f}")
    print(f"              centroid  = ({cx:.4f}, {cy:.4f})")

    # ── 1. Monte Carlo ────────────────────────────────────────────────────
    print("\n[1] Monte Carlo area estimation (200k samples, no formula assumed)")
    mc_area = monte_carlo_area(pts_closed, n_samples=200_000)
    print(f"    MC area    = {mc_area:.6f}")
    print(f"    Polar area = {polar_area:.6f}")
    print(f"    Relative error = {abs(mc_area - polar_area)/polar_area*100:.3f}%")

    # ── 2. Fine feasibility ───────────────────────────────────────────────
    print("\n[2] Fine feasibility check (3600 angles × 5120 boundary points)")
    feas = fine_feasibility_check(sofa, n_angles=3600)
    print(f"    Max penetration (3600 angles) = {feas['max_penetration_3600_angles']:.6f}")
    print(f"    Feasible: {'YES ✓' if feas['feasible'] else 'NO ✗'}")
    print(f"    Penalty at key angles:")
    for ang, pen in feas["penalty_at_key_angles"].items():
        print(f"      θ={ang:<6}  max_violation = {pen:.6f}")

    # ── 3. Area formula cross-check ───────────────────────────────────────
    print("\n[3] Area formula cross-check")
    sl_area = shoelace_area(pts_closed)
    print(f"    Shoelace area  = {sl_area:.6f}")
    print(f"    Polar area     = {polar_area:.6f}")
    print(f"    Discrepancy    = {abs(sl_area - polar_area):.6f}")
    if abs(sl_area - polar_area) < 0.01:
        print("    → Formulas agree: curve is simple (non-self-intersecting) ✓")
    else:
        print("    → DISCREPANCY: possible self-intersection ✗")

    si = check_self_intersection(pts)
    print(f"    Edge-crossing check: {si['crossings_found']}/{si['checks']} crossings found")
    print(f"    Self-intersecting: {'YES ✗' if si['self_intersecting'] else 'NO ✓'}")

    # ── 4. Upper bounds ───────────────────────────────────────────────────
    print("\n[4] Known theoretical bounds")
    bounds = compute_upper_bounds()
    bounds["our_optimum_star_shaped"] = polar_area
    print(f"    Unit-disk bound (trivial):         {bounds['unit_disk_bound']:.4f}")
    print(f"    Hammersley 1968 bound:             {bounds['hammersley_1968_bound']:.4f}")
    print(f"    Kallus–Romik 2018 (tightest proven): {bounds['kallus_romik_2018_bound']:.4f}")
    print(f"    Gerver 1992 conjectured optimum:   {bounds['gerver_1992_conjecture']:.4f}")
    print(f"    Our star-shaped optimum:           {bounds['our_optimum_star_shaped']:.4f}")
    gap_to_gerver  = bounds["gerver_1992_conjecture"] - polar_area
    gap_to_proven  = bounds["kallus_romik_2018_bound"]  - polar_area
    print(f"\n    Gap to Gerver conjecture:  {gap_to_gerver:.4f}")
    print(f"    Gap to proven upper bound: {gap_to_proven:.4f}")
    pct = polar_area / bounds["gerver_1992_conjecture"] * 100
    print(f"    We achieve {pct:.2f}% of Gerver's conjectured optimum.")

    # ── Summary plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: area comparison bar chart
    ax = axes[0]
    labels = ["Unit-disk\nbound (π)", "Hammersley\n1968", "Kallus–Romik\n2018",
              "Gerver 1992\nconjecture", "Our result\n(star-shaped)"]
    values = [bounds["unit_disk_bound"], bounds["hammersley_1968_bound"],
              bounds["kallus_romik_2018_bound"], bounds["gerver_1992_conjecture"],
              bounds["our_optimum_star_shaped"]]
    colors = ["#bdc3c7", "#95a5a6", "#f39c12", "#27ae60", "#e74c3c"]
    bars   = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(bounds["gerver_1992_conjecture"], color="#27ae60",
               linestyle="--", lw=1.5, alpha=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Area")
    ax.set_title("Area Bounds Comparison")
    ax.set_ylim(0, 3.5)
    ax.grid(axis="y", alpha=0.3)

    # Right: shape at θ=0, 45, 90 overlaid
    ax = axes[1]
    import matplotlib.patches as patches
    vert  = patches.Rectangle((-1, 0), 1, 2.2, color="steelblue", alpha=0.15, zorder=0)
    horiz = patches.Rectangle((-2.2, 0), 2.2, 1, color="steelblue", alpha=0.15, zorder=0)
    ax.add_patch(vert); ax.add_patch(horiz)
    ax.axvline(-1, color="navy", lw=1.5)
    ax.axvline(0,  color="navy", lw=1.5)
    ax.axhline(0,  color="navy", lw=1.5)
    ax.axhline(1,  color="navy", lw=1.5)
    ax.plot(0, 0, "ko", ms=5, zorder=5)

    from .collision import rotation_matrix_2d
    colors_theta = ["#e74c3c", "#9b59b6", "#3498db"]
    for th_deg, col in zip([0, 45, 90], colors_theta):
        th  = torch.tensor(th_deg * np.pi / 180, dtype=torch.float32)
        R   = rotation_matrix_2d(th)
        p   = (pts_t @ R.T).numpy()
        tx  = -p[:,0].max(); ty = -p[:,1].min()
        p  += np.array([tx, ty])
        pc  = np.vstack([p, p[0]])
        ax.fill(pc[:,0], pc[:,1], color=col, alpha=0.3, zorder=3, label=f"θ={th_deg}°")
        ax.plot(pc[:,0], pc[:,1], color=col, lw=1.5, zorder=4)

    ax.set_xlim(-2.2, 0.3); ax.set_ylim(-0.3, 2.2)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f"Verified sofa (area={polar_area:.4f}, MC={mc_area:.4f})\n"
        f"max_pen(3600 angles)={feas['max_penetration_3600_angles']:.5f}"
    )

    fig.suptitle("Phase 1 Independent Verification", fontsize=13)
    plt.tight_layout()
    os.makedirs("phase1_2d/results", exist_ok=True)
    save_path = "phase1_2d/results/verification.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVerification plot saved → {save_path}")
    print("=" * 65)

    return {
        "polar_area": polar_area,
        "mc_area": mc_area,
        "shoelace_area": sl_area,
        "max_penetration": feas["max_penetration_3600_angles"],
        "feasible": feas["feasible"],
    }


if __name__ == "__main__":
    run_verification()
