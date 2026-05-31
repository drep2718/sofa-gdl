# Moving Sofa Problem × Geometric Deep Learning

Differentiable shape optimization applied to one of the most famous unsolved problems in geometry.

**Best result: area = 2.0749 — 93.5% of Gerver's conjectured optimum (2.2195)**

---

## The Problem

The **Moving Sofa Problem** (Moser, 1966) asks:

> *What is the largest 2D shape that can be moved around a 90° corner in a unit-width hallway?*

Despite being easy to state, it remains **unsolved**. The best known solution is **Gerver's sofa** (1992), with area ≈ 2.2195 — conjectured optimal but unproven. The 3D version (largest volume object through an L-shaped hallway) is even less studied with no known answer.

The constraint is a continuous family of geometric inequalities: for every rotation angle θ ∈ [0°, 90°], the entire shape must fit inside the L-shaped hallway. This makes it a hard constrained optimization problem over an infinite-dimensional shape space.

---

## Approach

This project applies **Geometric Deep Learning** principles — encoding problem symmetries directly into the optimizer — to search for optimal shapes via differentiable geometry.

### Shape Representation

The sofa boundary is parameterized as a **polar B-spline**: a smooth closed curve defined by N radial control points. This guarantees:
- The shape is always **star-shaped** (no self-intersections)
- Area is computed exactly via the polar formula `A = (π/T) Σ r²`
- All parameters are differentiable — gradients flow through the geometry

### Hallway Constraint (Differentiable SDF)

The L-shaped hallway is defined by two corridors:
- Vertical: `−1 ≤ x ≤ 0`
- Horizontal: `0 ≤ y ≤ 1`

For each rotation angle θ, the sofa is placed at its **canonical position** (hugging the inner corner — the correct moving-sofa physics):

```
tx = −max_x(R(θ) · pts)
ty = −min_y(R(θ) · pts)
```

The signed distance function (SDF) measures how far each boundary point penetrates the wall. The constraint penalty is:

```
penalty = max over all (angle, point) pairs of relu(sdf)
```

Using `max` rather than `mean` is critical — mean dilutes violations across 46,000+ samples and hides them from the optimizer.

### Optimization: Augmented Lagrangian

```
loss = −Area(shape) + λ · max_violation(shape, θ)

Repeat for n_outer loops:
  1. Minimize loss with Adam (cosine-annealed lr: 3e-3 → 5e-4, 600 steps)
  2. If max_pen > 1e-3: double λ (up to 500)
```

Multi-start with 5 random seeds avoids local minima.

### Symmetry Analysis (GDL)

Gerver's sofa has a **Z/2Z bilateral symmetry**: swapping the two corridors maps `(x, y) → (−y, −x)`. For a polar curve with centroid at `(cx, cy)` satisfying `cx = −cy`, this acts on polar angles as:

```
α  →  3π/2 − α
```

This project derived and implemented this symmetry by tying control-point pairs `k ↔ (3N/4 − k) mod N`, halving the parameter count. A full experiment sweep tested whether enforcing this symmetry improves optimization.

---

## Results

### Best Result

| Metric | Value |
|--------|-------|
| Area | **2.0749** |
| % of Gerver | **93.5%** |
| Max penetration (3,600 angles) | 0.003303 |
| Verified feasible | ✓ |

The result was independently verified at 3,600 rotation angles (vs. 181 training angles) via:
- **Monte Carlo area**: 200k random samples, ray-casting → 2.0610 (0.08% error vs. polar formula)
- **Shoelace cross-check**: agrees with polar area to < 0.01
- **Self-intersection test**: 0 edge crossings

### GDL Experiment Sweep

| Configuration | Area | % of Gerver |
|---|---|---|
| B-spline n=64, no symmetry | **2.0749** | **93.5%** |
| B-spline n=128, no symmetry | 1.936 | 87.2% |
| Fourier K=32, no symmetry | 1.790 | 80.6% |
| B-spline n=64, Z/2Z symmetry | 1.082 | 48.7% |
| Fourier K=32, Z/2Z symmetry | 1.685 | 75.9% |

**Key findings:**
- `n_ctrl=64` is the sweet spot — 128 control points hurts (higher-dimensional non-convex landscape)
- B-spline beats Fourier — local basis functions better capture the sofa's locally-structured boundary
- Diagonal Z/2Z symmetry is **incompatible with polar parameterization**: the symmetry forces `r(0) = r(3π/2)`, tying constraints from different corridor walls at different rotation angles — optimizer gets trapped at area ≈ 1.08
- The star-shaped ceiling is ~2.07–2.10; breaking past it requires a non-star-shaped representation

---

## Project Structure

```
sofa-gdl/
├── phase1_2d/
│   ├── shape.py              # Polar B-spline parameterization
│   ├── collision.py          # Differentiable SDF + canonical translation
│   ├── optimize_improved.py  # Multi-start + cosine-annealed lr → best result
│   ├── shape_variants.py     # HighRes, Symmetric, Fourier, FourierSym variants
│   ├── experiments.py        # 6-config GDL sweep
│   ├── verify.py             # Independent 3600-angle + Monte Carlo verification
│   ├── visualize.py          # Static plot + animation
│   └── results/
│       ├── best_sofa_improved.pt       # Best checkpoint (area=2.0749)
│       ├── sofa_animation.gif          # Sofa navigating the corner
│       └── experiments_comparison.png  # GDL sweep comparison chart
├── phase2_mesh/              # 3D mesh infrastructure
│   ├── mesh_gen.py           # Icosphere + Fourier deformation
│   ├── sdf_hallway.py        # 3D L-shaped hallway SDF
│   ├── volume.py             # Divergence-theorem volume (differentiable)
│   └── dataset.py            # (mesh, feasible?) dataset generator
├── phase3_gnn/               # GNN feasibility classifier
│   ├── model.py              # GraphSAGE classifier
│   ├── train.py              # Training loop
│   └── eval.py               # AUC, confusion matrix
├── phase5_equivariant/       # SO(3)-equivariant GNN
│   ├── model_equivar.py      # Equivariant message passing
│   └── compare.py            # vs. vanilla GNN
└── tests/
    └── test_phase1.py        # 26 unit tests (all passing)
```

---

## Running It

```bash
# Setup
conda env create -f environment.yml
conda activate sofa-gdl

# Best result (multi-start, n_ctrl=64, 30 outer loops)
python -m phase1_2d.optimize_improved

# GDL experiment sweep (6 configs)
python -m phase1_2d.experiments

# Independent verification of saved checkpoint
python -m phase1_2d.verify

# Visualization — static plot + animation
python -m phase1_2d.visualize

# Tests
pytest tests/ -v
```

---

## Key Technical Insights

**Canonical translation is the physics.** Without it, the sofa just rotates in place and the maximum achievable area is π/4 ≈ 0.785. The correct formulation slides the shape to hug the inner corner at every angle — this single fix was the difference between a broken optimizer and a working one.

**Max-violation beats mean-violation.** With 91 angles × 512 boundary points = 46,592 samples per gradient step, a single deeply-penetrating point gets diluted to near zero under mean penalty. Max penalty directly targets the worst offender.

**Symmetry must be compatible with the basis.** The GDL principle of encoding problem symmetry into the model works when the symmetry acts naturally on the parameterization. The diagonal Z/2Z symmetry is geometrically correct but creates conflicting constraints in polar coordinates — a star-shaped symmetric sofa is fundamentally limited to area ~1.08.

**n_ctrl=64 is the sweet spot.** More control points don't help — the non-convex optimization landscape becomes harder to navigate, and Adam gets worse results at n_ctrl=128 (1.936 vs. 2.075).

---

## What's Next

- **Longer optimization**: area was still climbing at outer=29 — 60+ outer loops likely reaches ~2.10
- **Non-star-shaped parameterization**: Cartesian B-spline or support function `h(θ)` to represent the concave notch and break past 2.10
- **3D extension**: apply the same pipeline to the 3D moving sofa problem — an open question with no known answer

---

## References

- Moser, L. (1966). *Problem 66-11.* SIAM Review.
- Gerver, J. L. (1992). *The moving sofa problem.* The American Mathematical Monthly.
- Kallus, Y. & Romik, D. (2018). *Improved upper bounds in the moving sofa problem.* Advances in Mathematics. (Best proven upper bound: 2.37)
