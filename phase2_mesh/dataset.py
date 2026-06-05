"""
Dataset generation for the GNN feasibility classifier (Phase 3).

Each sample is a mesh with a binary label: can this shape navigate the 3D corner?

Generation strategy:
  - Vary amplitude: low amplitude → near-sphere, small → usually feasible
                    high amplitude → very deformed → usually infeasible
  - Vary scale: meshes scaled to radius ∈ [0.2, 0.55] — larger ones hit walls
  - This naturally produces a ~50/50 feasible/infeasible split

Each sample stored as a .pt file containing:
  {
    'V':      (N, 3) float32 vertices,
    'F':      (M, 3) int64  faces,
    'normals':(N, 3) float32 per-vertex normals,
    'label':  int   (1=feasible, 0=infeasible),
    'volume': float,
    'max_pen': float,
  }

Run from project root:
    python -m phase2_mesh.dataset --n 10000 --out dataset/
"""

import argparse
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from .mesh_gen import random_mesh, compute_vertex_normals, icosphere, deform_mesh
from .sdf_hallway import is_feasible, max_penetration_3d
from .volume import mesh_volume_abs


N_ANGLES = 91  # rotation angles: 0° to 90° in 1° steps


def generate_one(seed: int, scale_range=(0.2, 0.55)) -> dict:
    """Generate and label one random mesh."""
    rng = np.random.RandomState(seed)
    amplitude = rng.uniform(0.05, 0.50)
    target_scale = rng.uniform(*scale_range)

    V, F = random_mesh(subdivisions=3, amplitude=amplitude, seed=seed)

    # Override scale from random_mesh's default
    from .mesh_gen import scale_mesh
    V, F = scale_mesh(V, F, target_scale=target_scale)

    normals = compute_vertex_normals(V, F)
    volume = mesh_volume_abs(V, F).item()

    thetas = torch.linspace(0, np.pi / 2, N_ANGLES)
    max_pen = max_penetration_3d(V, thetas)
    label = int(max_pen < 1e-3)

    return {
        "V": V,
        "F": F,
        "normals": normals,
        "label": label,
        "volume": volume,
        "max_pen": max_pen,
    }


def generate_dataset(
    n: int = 10_000,
    out_dir: str = "dataset",
    start_seed: int = 0,
    verbose: bool = True,
) -> None:
    """
    Generate n labeled meshes and save to out_dir.

    Each mesh saved as dataset/mesh_{i:06d}.pt.
    Also saves a manifest: dataset/manifest.pt with labels, volumes, seeds.
    """
    os.makedirs(out_dir, exist_ok=True)

    labels = []
    volumes = []
    max_pens = []

    iterator = range(start_seed, start_seed + n)
    if verbose:
        iterator = tqdm(iterator, desc="Generating meshes")

    n_feasible = 0
    for i, seed in enumerate(iterator):
        sample = generate_one(seed)

        save_path = os.path.join(out_dir, f"mesh_{i:06d}.pt")
        torch.save(sample, save_path)

        labels.append(sample["label"])
        volumes.append(sample["volume"])
        max_pens.append(sample["max_pen"])
        n_feasible += sample["label"]

        if verbose and (i + 1) % 500 == 0:
            frac = n_feasible / (i + 1)
            tqdm.write(f"  {i+1}/{n} | feasible: {n_feasible} ({frac:.1%})")

    manifest = {
        "n": n,
        "labels": torch.tensor(labels, dtype=torch.long),
        "volumes": torch.tensor(volumes, dtype=torch.float32),
        "max_pens": torch.tensor(max_pens, dtype=torch.float32),
        "seeds": torch.arange(start_seed, start_seed + n),
    }
    torch.save(manifest, os.path.join(out_dir, "manifest.pt"))

    if verbose:
        n_feas = sum(labels)
        print(f"\nDataset saved to {out_dir}/")
        print(f"  Total:     {n}")
        print(f"  Feasible:  {n_feas} ({n_feas/n:.1%})")
        print(f"  Infeasible:{n - n_feas} ({(n-n_feas)/n:.1%})")
        print(f"  Avg volume:{np.mean(volumes):.4f}")


class SofaDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping the generated mesh files.

    Returns per-sample dicts with keys:
        x:     (N, 6) float — vertex positions + normals concatenated
        edge_index: (2, E) long — mesh edges for graph convolution
        y:     scalar long — feasibility label
        volume: scalar float

    The edge_index is built from the face list: each triangle edge
    a→b and b→a added (undirected graph).
    """

    def __init__(self, dataset_dir: str, indices=None):
        self.dir = dataset_dir
        manifest = torch.load(os.path.join(dataset_dir, "manifest.pt"), weights_only=True)
        self.n = manifest["n"]
        self.labels = manifest["labels"]
        self.volumes = manifest["volumes"]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(self.n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = os.path.join(self.dir, f"mesh_{real_idx:06d}.pt")
        sample = torch.load(path, weights_only=True)

        V = sample["V"]               # (N, 3)
        normals = sample["normals"]   # (N, 3)
        F = sample["F"]               # (M, 3)

        # Node features: position + normals
        x = torch.cat([V, normals], dim=1)  # (N, 6)

        # Build undirected edge index from faces
        edges = torch.cat([
            F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
            F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]],
        ], dim=0).T.contiguous()  # (2, 6M)

        return {
            "x": x,
            "edge_index": edges,
            "y": sample["label"],
            "volume": sample["volume"],
        }


def train_val_test_split(dataset_dir: str, train=0.8, val=0.1):
    """
    Return SofaDataset instances for train/val/test splits.

    Uses a fixed random seed for reproducible splits.
    """
    manifest = torch.load(os.path.join(dataset_dir, "manifest.pt"), weights_only=True)
    n = manifest["n"]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)

    n_train = int(n * train)
    n_val = int(n * val)

    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:n_train + n_val].tolist()
    test_idx = perm[n_train + n_val:].tolist()

    return (
        SofaDataset(dataset_dir, train_idx),
        SofaDataset(dataset_dir, val_idx),
        SofaDataset(dataset_dir, test_idx),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate moving sofa mesh dataset")
    parser.add_argument("--n", type=int, default=10_000, help="Number of meshes")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    args = parser.parse_args()

    t0 = time.time()
    generate_dataset(n=args.n, out_dir=args.out, start_seed=args.seed)
    print(f"Total time: {time.time() - t0:.1f}s")
