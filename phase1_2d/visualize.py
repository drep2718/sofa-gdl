"""
Visualization for Phase 1: animate the sofa moving around the corner.

Produces:
  1. Static plot: optimized shape + hallway + key geometry labels
  2. Animated GIF: sofa rotating θ = 0° → 90° through the corner

The hallway is drawn to infinite extent in both corridors so the viewer
can see clearly when the shape stays inside vs. violates the walls.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from .shape import SofaShape
from .collision import rotation_matrix_2d


# ── hallway geometry constants ─────────────────────────────────────────────
#   Vertical corridor:   x ∈ [-1, 0],  y ∈ [-3, +3]   (drawn finite)
#   Horizontal corridor: x ∈ [-3,  0], y ∈ [ 0,  1]
VERT_X  = (-1.0,  0.0)
VERT_Y  = (-3.0,  3.0)
HORIZ_X = (-3.0,  0.0)
HORIZ_Y = ( 0.0,  1.0)


def load_sofa() -> SofaShape:
    """Load saved optimized shape, or fall back to initial circle."""
    save_path = "phase1_2d/results/best_sofa.pt"
    sofa = SofaShape(n_ctrl=64, n_sample=512, radius=0.35)
    if os.path.exists(save_path):
        data = torch.load(save_path, weights_only=True)
        with torch.no_grad():
            sofa.raw_r.copy_(data["raw_r"])
            sofa.centroid.copy_(data["centroid"])
        print(f"Loaded saved sofa (area={data['area']:.4f})")
    else:
        print("No saved result — showing initial circle. Run optimize.py first.")
    return sofa


def draw_hallway(ax, alpha: float = 0.18):
    """
    Draw the L-shaped hallway as filled + outlined rectangles.

    The hallway interior is blue; walls are navy lines.
    We draw it large enough that the sofa is always visually contained.
    """
    # Filled interior
    vert  = patches.Rectangle(
        (VERT_X[0],  VERT_Y[0]),
        VERT_X[1]  - VERT_X[0],
        VERT_Y[1]  - VERT_Y[0],
        color="steelblue", alpha=alpha, zorder=0,
    )
    horiz = patches.Rectangle(
        (HORIZ_X[0], HORIZ_Y[0]),
        HORIZ_X[1] - HORIZ_X[0],
        HORIZ_Y[1] - HORIZ_Y[0],
        color="steelblue", alpha=alpha, zorder=0,
    )
    ax.add_patch(vert)
    ax.add_patch(horiz)

    # Walls (solid navy lines)
    wkw = dict(color="navy", lw=2.0, zorder=2)
    # Vertical corridor outer left wall
    ax.plot([VERT_X[0], VERT_X[0]], [VERT_Y[0], 0.0],   **wkw)
    # Vertical corridor outer left wall (above horizontal)
    ax.plot([VERT_X[0], VERT_X[0]], [1.0, VERT_Y[1]],   **wkw)
    # Vertical corridor inner right wall (below horizontal join)
    ax.plot([VERT_X[1], VERT_X[1]], [VERT_Y[0], 0.0],   **wkw)
    # Horizontal corridor bottom wall
    ax.plot([HORIZ_X[0], 0.0],      [HORIZ_Y[0], HORIZ_Y[0]], **wkw)
    # Horizontal corridor top wall
    ax.plot([HORIZ_X[0], VERT_X[0]], [HORIZ_Y[1], HORIZ_Y[1]], **wkw)
    # Inner corner vertical stub (x=0, y=0 to y=1)
    ax.plot([0.0, 0.0], [0.0, HORIZ_Y[1]], **wkw)
    # Outer corner connection (x=-1 to x=-3, y=1)
    ax.plot([VERT_X[0], HORIZ_X[0]], [HORIZ_Y[1], HORIZ_Y[1]], **wkw)

    # Corner label
    ax.plot(0, 0, "ko", ms=4, zorder=5)
    ax.annotate("inner corner\n(0,0)", xy=(0, 0), xytext=(0.1, -0.2),
                fontsize=7, color="navy")


def draw_sofa(ax, pts: np.ndarray, color: str = "tomato", alpha: float = 0.65, label: str = ""):
    closed = np.vstack([pts, pts[0]])
    ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=alpha, zorder=3, label=label)
    ax.plot(closed[:, 0], closed[:, 1], color="darkred", lw=1.2, zorder=4)


def _apply_canonical_translation(pts_rot: np.ndarray) -> np.ndarray:
    """
    Push rotated shape to hug inner corner: max_x → 0, min_y → 0.
    Mirrors the canonical translation used during optimization.
    """
    tx = -pts_rot[:, 0].max()
    ty = -pts_rot[:, 1].min()
    return pts_rot + np.array([tx, ty])


def plot_static(sofa: SofaShape, save_path: str = "phase1_2d/results/sofa_static.png"):
    """Static plot at θ = 0°, 45°, 90° with canonical translation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for i, theta_deg in enumerate([0, 45, 90]):
        ax = axes[i]
        draw_hallway(ax)

        theta = torch.tensor(theta_deg * np.pi / 180, dtype=torch.float32)
        R     = rotation_matrix_2d(theta)
        pts   = _apply_canonical_translation((sofa().detach() @ R.T).numpy())

        draw_sofa(ax, pts, label=f"θ={theta_deg}°")

        area = sofa.area().item()
        ax.set_xlim(-2.2, 0.4)
        ax.set_ylim(-0.4, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"θ = {theta_deg}°   area = {area:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Phase 1 — Moving Sofa (polar parameterization)   "
        f"area={sofa.area().item():.4f}   Gerver ref=2.2195",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved static plot → {save_path}")
    plt.show()


def animate_sofa(
    sofa:      SofaShape,
    n_frames:  int = 90,
    save_path: str = "phase1_2d/results/sofa_animation.gif",
):
    """
    Animate sofa rotating from θ=0° to θ=90° through the corner.

    The shape stays fixed (polar frame); we rotate its boundary points
    by θ at each frame to simulate the corner-turn motion.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    pts_base = sofa().detach()   # (T, 2) in sofa frame
    area     = sofa.area().item()
    thetas   = np.linspace(0, np.pi / 2, n_frames)

    def update(frame_idx):
        ax.clear()
        draw_hallway(ax, alpha=0.22)

        theta   = torch.tensor(thetas[frame_idx], dtype=torch.float32)
        R       = rotation_matrix_2d(theta)
        pts_rot = _apply_canonical_translation((pts_base @ R.T).numpy())
        draw_sofa(ax, pts_rot)

        ax.set_xlim(-2.2, 0.4)
        ax.set_ylim(-0.4, 2.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"Moving Sofa — θ = {np.degrees(thetas[frame_idx]):.1f}°   "
            f"area = {area:.4f}   Gerver = 2.2195"
        )
        ax.set_xlabel("x"); ax.set_ylabel("y")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer="pillow", fps=20)
    print(f"Saved animation → {save_path}")
    plt.show()


if __name__ == "__main__":
    sofa = load_sofa()
    plot_static(sofa)
    animate_sofa(sofa)
