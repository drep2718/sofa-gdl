"""
Phase 4: 3D shape optimization — maximize volume subject to feasibility.

Problem:
    maximize   Volume(V)
    subject to GNN_feasibility(V, F, normals) >= 0.95

Algorithm: Augmented Lagrangian Method on mesh vertex positions.

    L(V, λ) = -Volume(V) + λ * max(0, 0.95 - GNN(V))²

Outer loop: increase λ when constraint is violated.
Inner loop: Adam gradient descent on L w.r.t. V (vertices are the free variables).
After each gradient step: apply Laplacian smoothing to keep mesh valid.

The GNN gradient ∂GNN/∂V is the "shape gradient of feasibility" — it tells
us which vertices to move to make the shape more feasible without ever
running the expensive SDF checker.

After optimization: verify each candidate with the true SDF checker, not the GNN.

Run:
    python -m phase4_optim.optimize_3d \
        --checkpoint phase3_gnn/checkpoints/best_model.pt \
        --n_restarts 50 \
        --out phase4_optim/results/
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from phase2_mesh.mesh_gen import random_mesh, compute_vertex_normals, icosphere, scale_mesh
from phase2_mesh.volume import mesh_volume_abs, laplacian_smoothness
from phase2_mesh.sdf_hallway import max_penetration_3d, is_feasible
from phase3_gnn.model import MeshFeasibilityGNN
from phase3_gnn.train import get_device


N_ANGLES = 91
FEASIBILITY_THRESHOLD = 0.95


def load_gnn(checkpoint_path: str, device: torch.device) -> MeshFeasibilityGNN:
    model = MeshFeasibilityGNN(in_channels=6, hidden=128)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_edge_index(F: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build undirected edge index from face list."""
    edges = torch.cat([
        F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
        F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]],
    ], dim=0).T.contiguous()
    return edges.to(device)


def gnn_feasibility(
    V: torch.Tensor,
    F: torch.Tensor,
    edge_index: torch.Tensor,
    model: MeshFeasibilityGNN,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute P(feasible) for a single mesh — differentiable w.r.t. V.

    We recompute normals inside this function so gradients flow through
    the normal computation into V. This means ∂GNN/∂V includes both the
    direct effect of vertex positions AND their effect via surface normals.
    """
    # Recompute normals differentiably
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (M, 3)

    normals = torch.zeros_like(V)
    for i in range(3):
        normals.scatter_add_(0, F[:, i:i+1].expand(-1, 3), face_normals)
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-12)

    x = torch.cat([V, normals], dim=1).to(device)          # (N, 6)
    batch_vec = torch.zeros(V.shape[0], dtype=torch.long, device=device)

    # Run in train mode so gradients flow (eval mode disables dropout,
    # but we still need gradients for optimization)
    return model(x, edge_index, batch_vec)  # scalar in [0,1]


def optimize_single(
    V_init: torch.Tensor,
    F: torch.Tensor,
    model: MeshFeasibilityGNN,
    device: torch.device,
    n_outer: int = 15,
    n_inner: int = 200,
    lr: float = 5e-4,
    lambda_init: float = 1.0,
    lambda_scale: float = 5.0,
    smooth_weight: float = 1e-3,
    verbose: bool = False,
) -> dict:
    """
    Run augmented Lagrangian optimization for one random initialization.

    Returns a dict with the best V found and its stats.
    """
    V = nn.Parameter(V_init.clone().to(device))
    F_dev = F.to(device)
    edge_index = build_edge_index(F, device)
    lam = lambda_init

    best_vol = 0.0
    best_V = V_init.clone()

    for outer in range(n_outer):
        optimizer = torch.optim.Adam([V], lr=lr)

        for inner in range(n_inner):
            optimizer.zero_grad()

            vol = mesh_volume_abs(V, F_dev)
            feas = gnn_feasibility(V, F_dev, edge_index, model, device)

            # Augmented Lagrangian: penalize infeasibility
            constraint_viol = torch.relu(FEASIBILITY_THRESHOLD - feas)
            smooth_pen = laplacian_smoothness(V, F_dev)

            loss = -vol + lam * constraint_viol.pow(2) + smooth_weight * smooth_pen
            loss.backward()
            optimizer.step()

        # After inner loop: check true feasibility with SDF
        with torch.no_grad():
            vol = mesh_volume_abs(V, F_dev).item()
            feas_prob = gnn_feasibility(V, F_dev, edge_index, model, device).item()

            thetas = torch.linspace(0, np.pi / 2, N_ANGLES)
            V_cpu = V.detach().cpu()
            max_pen = max_penetration_3d(V_cpu, thetas)
            truly_feasible = max_pen < 1e-3

        if verbose:
            print(f"  outer={outer} vol={vol:.4f} gnn={feas_prob:.3f} "
                  f"max_pen={max_pen:.4f} feasible={truly_feasible} λ={lam:.1f}")

        if feas_prob >= FEASIBILITY_THRESHOLD and vol > best_vol:
            best_vol = vol
            best_V = V.detach().cpu().clone()

        if feas_prob < FEASIBILITY_THRESHOLD:
            lam *= lambda_scale

    return {
        "V": best_V,
        "F": F.cpu(),
        "volume": best_vol,
        "max_pen": max_penetration_3d(best_V, torch.linspace(0, np.pi / 2, N_ANGLES)),
        "feasible": best_vol > 0.0,
    }


def run_optimization(
    checkpoint_path: str,
    n_restarts: int = 50,
    out_dir: str = "phase4_optim/results",
    n_outer: int = 15,
    n_inner: int = 200,
    verbose: bool = True,
):
    device = get_device()
    print(f"Device: {device}")

    model = load_gnn(checkpoint_path, device)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    thetas = torch.linspace(0, np.pi / 2, N_ANGLES)

    # Keep a fixed icosphere topology; only optimize vertex positions
    V_base, F = icosphere(subdivisions=3)

    iterator = range(n_restarts)
    if verbose:
        iterator = tqdm(iterator, desc="Restarts")

    for i in iterator:
        np.random.seed(i * 17)
        amplitude = np.random.uniform(0.05, 0.4)
        from phase2_mesh.mesh_gen import deform_mesh
        V_init, _ = deform_mesh(V_base.clone(), F, amplitude=amplitude, seed=i)
        V_init, _ = scale_mesh(V_init, F, target_scale=0.42)

        result = optimize_single(
            V_init, F, model, device,
            n_outer=n_outer, n_inner=n_inner, verbose=False,
        )
        result["seed"] = i
        results.append(result)

    # Sort by volume descending, take top 5 truly feasible
    feasible_results = [r for r in results if r["feasible"] and r["max_pen"] < 1e-3]
    feasible_results.sort(key=lambda r: r["volume"], reverse=True)
    top5 = feasible_results[:5]

    print(f"\nFound {len(feasible_results)} truly feasible shapes (out of {n_restarts} restarts)")

    for rank, r in enumerate(top5):
        print(f"  #{rank+1}: volume={r['volume']:.5f}  max_pen={r['max_pen']:.6f}  seed={r['seed']}")
        save_obj(r["V"], r["F"], os.path.join(out_dir, f"best_{rank+1}.obj"))

    torch.save(results, os.path.join(out_dir, "all_results.pt"))
    print(f"\nSaved .obj files to {out_dir}/")
    return top5


def save_obj(V: torch.Tensor, F: torch.Tensor, path: str):
    """Save mesh as Wavefront .obj file (viewable in Blender, MeshLab, etc.)."""
    V_np = V.numpy()
    F_np = F.numpy()
    with open(path, "w") as f:
        for v in V_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in F_np:
            # OBJ uses 1-indexed vertices
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str,
                        default="phase3_gnn/checkpoints/best_model.pt")
    parser.add_argument("--n_restarts",  type=int, default=50)
    parser.add_argument("--out",         type=str, default="phase4_optim/results")
    parser.add_argument("--n_outer",     type=int, default=15)
    parser.add_argument("--n_inner",     type=int, default=200)
    args = parser.parse_args()

    run_optimization(
        checkpoint_path=args.checkpoint,
        n_restarts=args.n_restarts,
        out_dir=args.out,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
    )
