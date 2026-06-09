"""
Training loop for the mesh feasibility GNN.

The batch collation is custom because meshes have variable vertex/edge counts.
We can't use PyTorch's default DataLoader collate_fn — instead we manually
offset edge indices when batching multiple graphs into one large disconnected graph.

This "batch-as-single-graph" trick is standard in graph ML:
  graph 0: vertices 0..N0-1,   edges all < N0
  graph 1: vertices N0..N0+N1-1, edges all in [N0, N0+N1)
  ...
All vertex features and edge indices stacked → one big graph.
The `batch` tensor records which graph each vertex belongs to for pooling.

Run from project root:
    python -m phase3_gnn.train --dataset dataset/ --epochs 100 --out phase3_gnn/checkpoints/
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .model import MeshFeasibilityGNN
from phase2_mesh.dataset import SofaDataset, train_val_test_split


def collate_graphs(batch):
    """
    Custom collate: merge a list of variable-size graphs into one big graph.

    Each sample has keys: x (N,6), edge_index (2,E), y (scalar), volume (scalar).
    We stack all x's, offset edge indices by cumulative vertex count, build batch tensor.
    """
    xs, edges, ys, vols = [], [], [], []
    offset = 0

    for sample in batch:
        n = sample["x"].shape[0]
        xs.append(sample["x"])
        edges.append(sample["edge_index"] + offset)  # offset vertex indices
        ys.append(sample["y"])
        vols.append(sample["volume"])
        offset += n

    # Batch tensor: vertex i belongs to graph g if vertices[sum(n_prev)..sum(n_prev)+n_g]
    batch_tensor = torch.cat([
        torch.full((s["x"].shape[0],), i, dtype=torch.long)
        for i, s in enumerate(batch)
    ])

    return {
        "x":          torch.cat(xs, dim=0),              # (N_total, 6)
        "edge_index": torch.cat(edges, dim=1),            # (2, E_total)
        "y":          torch.tensor(ys, dtype=torch.long), # (B,)
        "volume":     torch.tensor(vols, dtype=torch.float32),
        "batch":      batch_tensor,                       # (N_total,)
    }


def get_device() -> torch.device:
    """Pick MPS (Apple Silicon), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for batch in loader:
        x          = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec  = batch["batch"].to(device)
        y          = batch["y"].to(device).float()

        optimizer.zero_grad()
        prob = model(x, edge_index, batch_vec)  # (B,)
        loss = criterion(prob, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        pred = (prob > 0.5).long()
        n_correct += (pred == batch["y"].to(device)).sum().item()
        n_total += len(y)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for batch in loader:
        x          = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec  = batch["batch"].to(device)
        y          = batch["y"].to(device).float()

        prob = model(x, edge_index, batch_vec)
        loss = criterion(prob, y)

        total_loss += loss.item() * len(y)
        pred = (prob > 0.5).long()
        n_correct += (pred == batch["y"].to(device)).sum().item()
        n_total += len(y)

        all_probs.append(prob.cpu())
        all_labels.append(batch["y"].cpu())

    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return total_loss / n_total, n_correct / n_total, probs, labels


def train(
    dataset_dir: str = "dataset",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    hidden: int = 128,
    out_dir: str = "phase3_gnn/checkpoints",
    verbose: bool = True,
):
    device = get_device()
    if verbose:
        print(f"Device: {device}")

    # Data
    train_ds, val_ds, test_ds = train_val_test_split(dataset_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_graphs, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_graphs, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_graphs, num_workers=0)

    if verbose:
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model     = MeshFeasibilityGNN(in_channels=6, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    os.makedirs(out_dir, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        if verbose and epoch % 10 == 0:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"best_val={best_val_acc:.3f} | {dt:.1f}s"
            )

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), weights_only=True))
    test_loss, test_acc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)

    torch.save(history, os.path.join(out_dir, "history.pt"))
    torch.save({"probs": test_probs, "labels": test_labels}, os.path.join(out_dir, "test_results.pt"))

    if verbose:
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Checkpoints saved to {out_dir}/")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--out",     type=str, default="phase3_gnn/checkpoints")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        out_dir=args.out,
    )
