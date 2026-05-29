"""
Phase 1 experiment sweep: compare shape representations and symmetry groups.

Runs 8 configurations covering:
  - Resolution (n_ctrl: 64 / 128 / 256)
  - Symmetry group (none vs. Z/2Z bilateral)
  - Basis (B-spline vs. Fourier)

Each config is optimised with identical hyperparameters so the comparison
is fair.  Results are saved to phase1_2d/results/experiment_results.txt
and a summary plot to phase1_2d/results/experiments_comparison.png.

Usage:
    python -m phase1_2d.experiments
"""

import os
import time
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Type

from .shape import SofaShape
from .shape_variants import (
    SofaShapeHighRes,
    SofaShapeSymmetric,
    SofaShapeFourier,
    SofaShapeFourierSym,
)
from .collision import constraint_penalty, max_penetration


# ── shared optimisation routine ───────────────────────────────────────────────

def run_optimization(
    sofa,
    n_angles:     int   = 91,
    n_outer:      int   = 30,
    n_inner:      int   = 600,
    lr:           float = 2e-3,
    lambda_init:  float = 5.0,
    lambda_scale: float = 2.0,
    lambda_max:   float = 500.0,
    verbose:      bool  = False,
):
    """Run ALM optimisation on any shape object with .forward(), .area() methods."""
    thetas = torch.linspace(0, np.pi / 2, n_angles, dtype=torch.float32)
    lam = lambda_init

    for outer in range(n_outer):
        optimizer = torch.optim.Adam(sofa.parameters(), lr=lr)
        for inner in range(n_inner):
            optimizer.zero_grad()
            loss = -sofa.area() + lam * constraint_penalty(sofa(), thetas)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            max_pen = max_penetration(sofa(), thetas)

        if verbose:
            print(f"  outer={outer:2d} area={sofa.area().item():.4f} "
                  f"max_pen={max_pen:.5f} λ={lam:.1f}")

        if max_pen > 1e-3:
            lam = min(lam * lambda_scale, lambda_max)

    with torch.no_grad():
        final_pen = max_penetration(sofa(), thetas)
    return sofa.area().item(), final_pen


# ── experiment definitions ────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name:       str
    label:      str          # short label for plot
    group:      str          # symmetry group description
    basis:      str          # B-spline / Fourier
    n_params:   int = 0      # filled in after model creation
    area:       float = 0.0
    max_pen:    float = 0.0
    seconds:    float = 0.0
    sofa:       object = field(default=None, repr=False)


def make_configs():
    """Define all experiment configurations."""
    configs = [
        # ── B-spline, no symmetry ─────────────────────────────────────────
        ExperimentConfig(
            name="bspline_64",
            label="B-spline\nn=64",
            group="trivial C1",
            basis="B-spline",
        ),
        ExperimentConfig(
            name="bspline_128",
            label="B-spline\nn=128",
            group="trivial C1",
            basis="B-spline",
        ),
        # bspline_256 skipped — too slow on CPU (use GPU for full sweep)
        # ExperimentConfig(name="bspline_256", ...)
        # ── B-spline + Z/2Z bilateral symmetry ───────────────────────────
        ExperimentConfig(
            name="bspline_sym_64",
            label="B-spline\nZ/2Z n=64",
            group="Z/2Z",
            basis="B-spline",
        ),
        ExperimentConfig(
            name="bspline_sym_128",
            label="B-spline\nZ/2Z n=128",
            group="Z/2Z",
            basis="B-spline",
        ),
        # ── Fourier, no symmetry ──────────────────────────────────────────
        ExperimentConfig(
            name="fourier_k32",
            label="Fourier\nK=32",
            group="trivial C1",
            basis="Fourier",
        ),
        # ── Fourier + Z/2Z (cosines-only) ─────────────────────────────────
        ExperimentConfig(
            name="fourier_sym_k32",
            label="Fourier\nZ/2Z K=32",
            group="Z/2Z",
            basis="Fourier (even)",
        ),
    ]
    return configs


def build_model(cfg: ExperimentConfig):
    if cfg.name == "bspline_64":
        return SofaShape(n_ctrl=64, n_sample=512)
    if cfg.name == "bspline_128":
        return SofaShapeHighRes(n_ctrl=128, n_sample=1024)
    if cfg.name == "bspline_256":
        return SofaShapeHighRes(n_ctrl=256, n_sample=2048)
    if cfg.name == "bspline_sym_64":
        return SofaShapeSymmetric(n_ctrl=64, n_sample=512)
    if cfg.name == "bspline_sym_128":
        return SofaShapeSymmetric(n_ctrl=128, n_sample=1024)
    if cfg.name == "fourier_k32":
        return SofaShapeFourier(n_modes=32, n_sample=512)
    if cfg.name == "fourier_k64":
        return SofaShapeFourier(n_modes=64, n_sample=1024)
    if cfg.name == "fourier_sym_k32":
        return SofaShapeFourierSym(n_modes=32, n_sample=512)
    if cfg.name == "fourier_sym_k64":
        return SofaShapeFourierSym(n_modes=64, n_sample=1024)
    raise ValueError(cfg.name)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── run all experiments ───────────────────────────────────────────────────────

def run_all(verbose: bool = True):
    configs = make_configs()
    os.makedirs("phase1_2d/results", exist_ok=True)
    gerver = 2.2195

    print(f"\n{'='*72}")
    print(f" Phase 1 Experiment Sweep — {len(configs)} configurations")
    print(f" Gerver reference area: {gerver}")
    print(f"{'='*72}\n")

    for i, cfg in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {cfg.name}", flush=True)
        sofa = build_model(cfg)
        cfg.n_params = count_params(sofa)
        print(f"         params={cfg.n_params}  basis={cfg.basis}  group={cfg.group}", flush=True)

        t0 = time.time()
        area, max_pen = run_optimization(
            sofa,
            n_angles=91,
            n_outer=20,     # was 30 — fewer outer loops to keep runtime ~3 min/config
            n_inner=400,    # was 600
            lr=2e-3,
            lambda_init=5.0,
            lambda_scale=2.0,
            lambda_max=500.0,
            verbose=verbose,
        )
        cfg.seconds = time.time() - t0
        cfg.area    = area
        cfg.max_pen = max_pen
        cfg.sofa    = sofa

        gap = gerver - area
        feasible = "✓" if max_pen < 1e-2 else "✗"
        print(f"         area={area:.4f}  gap={gap:.4f}  "
              f"max_pen={max_pen:.5f} {feasible}  "
              f"time={cfg.seconds:.0f}s\n", flush=True)

    return configs


# ── results table and plots ───────────────────────────────────────────────────

def print_table(configs):
    gerver = 2.2195
    print(f"\n{'─'*72}")
    print(f"{'Config':<22} {'Basis':<16} {'Group':<10} "
          f"{'Params':>7} {'Area':>7} {'Gap':>7} {'MaxPen':>9} {'Sec':>6}")
    print(f"{'─'*72}")
    for cfg in sorted(configs, key=lambda c: -c.area):
        print(f"{cfg.name:<22} {cfg.basis:<16} {cfg.group:<10} "
              f"{cfg.n_params:>7} {cfg.area:>7.4f} "
              f"{gerver-cfg.area:>7.4f} {cfg.max_pen:>9.5f} {cfg.seconds:>6.0f}")
    print(f"{'─'*72}")
    print(f"{'Gerver reference':>55} {gerver:.4f}")
    print(f"{'─'*72}\n")


def save_results_txt(configs, path="phase1_2d/results/experiment_results.txt"):
    gerver = 2.2195
    lines = []
    lines.append("Phase 1 Experiment Results")
    lines.append(f"Gerver reference: {gerver}")
    lines.append("")
    lines.append(f"{'Config':<22} {'Area':>7} {'Gap':>7} {'MaxPen':>9} {'Params':>7} {'Sec':>6}")
    lines.append("─" * 60)
    for cfg in sorted(configs, key=lambda c: -c.area):
        lines.append(f"{cfg.name:<22} {cfg.area:>7.4f} "
                     f"{gerver-cfg.area:>7.4f} {cfg.max_pen:>9.5f} "
                     f"{cfg.n_params:>7} {cfg.seconds:>6.0f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved → {path}")


def plot_comparison(configs, save_path="phase1_2d/results/experiments_comparison.png"):
    gerver = 2.2195
    configs_sorted = sorted(configs, key=lambda c: -c.area)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Bar chart: area comparison ────────────────────────────────────────
    ax = axes[0]
    names  = [c.label for c in configs_sorted]
    areas  = [c.area  for c in configs_sorted]
    colors = []
    for c in configs_sorted:
        if c.group == "Z/2Z" and c.basis.startswith("Fourier"):
            colors.append("#e74c3c")   # red: Fourier + symmetric
        elif c.group == "Z/2Z":
            colors.append("#e67e22")   # orange: B-spline + symmetric
        elif c.basis.startswith("Fourier"):
            colors.append("#3498db")   # blue: Fourier unsymmetric
        else:
            colors.append("#95a5a6")   # grey: B-spline unsymmetric

    bars = ax.barh(names, areas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(gerver, color="green", linestyle="--", lw=2, label=f"Gerver {gerver}")
    ax.set_xlabel("Area")
    ax.set_title("Area by Configuration")
    ax.legend(fontsize=9)
    ax.set_xlim(1.8, 2.35)
    for bar, area in zip(bars, areas):
        ax.text(area + 0.002, bar.get_y() + bar.get_height()/2,
                f"{area:.4f}", va="center", fontsize=8)

    # ── Scatter: params vs. area ──────────────────────────────────────────
    ax = axes[1]
    for c in configs:
        marker = "^" if c.basis.startswith("Fourier") else "o"
        color  = "#e74c3c" if c.group == "Z/2Z" else "#3498db"
        ax.scatter(c.n_params, c.area, s=120, marker=marker,
                   color=color, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(c.label.replace("\n", " "), (c.n_params, c.area),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(gerver, color="green", linestyle="--", lw=1.5, label=f"Gerver {gerver}")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Area")
    ax.set_title("Parameter efficiency")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=9, markeredgecolor="black", label="No symmetry"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=9, markeredgecolor="black", label="Z/2Z symmetric"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=9, markeredgecolor="black", label="B-spline"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markersize=9, markeredgecolor="black", label="Fourier"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Best shapes overlay ───────────────────────────────────────────────
    ax = axes[2]
    thetas_vis = torch.linspace(0, 2 * np.pi, 200)

    # Sort by area, draw top-3
    top3 = sorted(configs, key=lambda c: -c.area)[:3]
    palette = ["#e74c3c", "#3498db", "#2ecc71"]

    # Draw hallway
    import matplotlib.patches as patches
    vert  = patches.Rectangle((-1, 0), 1, 2.2, color="steelblue", alpha=0.15, zorder=0)
    horiz = patches.Rectangle((-2.2, 0), 2.2, 1, color="steelblue", alpha=0.15, zorder=0)
    ax.add_patch(vert); ax.add_patch(horiz)
    ax.axvline(-1, color="navy", lw=1.5)
    ax.axvline(0,  color="navy", lw=1.5)
    ax.axhline(0,  color="navy", lw=1.5)
    ax.axhline(1,  color="navy", lw=1.5, xmax=((-1+2.2)/2.6))
    ax.plot(0, 0, "ko", ms=4)

    for cfg, color in zip(top3, palette):
        with torch.no_grad():
            pts = cfg.sofa().numpy()
        # Apply canonical translation at θ=0
        tx = -pts[:, 0].max(); ty = -pts[:, 1].min()
        pts = pts + np.array([tx, ty])
        closed = np.vstack([pts, pts[0]])
        ax.fill(closed[:,0], closed[:,1], alpha=0.35, color=color,
                label=f"{cfg.label.replace(chr(10),' ')} ({cfg.area:.4f})", zorder=3)
        ax.plot(closed[:,0], closed[:,1], color=color, lw=1.5, zorder=4)

    ax.set_xlim(-2.2, 0.3); ax.set_ylim(-0.3, 2.2)
    ax.set_aspect("equal")
    ax.set_title("Top-3 shapes (θ=0°, canonical position)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Phase 1 Experiment Sweep — Shape Parameterisation & Symmetry Groups\n"
        f"(Gerver reference: {gerver}  |  Green dashed line)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved → {save_path}")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configs = run_all(verbose=False)
    print_table(configs)
    save_results_txt(configs)
    plot_comparison(configs)
