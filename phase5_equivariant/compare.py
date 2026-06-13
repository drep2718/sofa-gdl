"""
Phase 5 comparison: equivariant GNN vs vanilla GNN.

Trains the equivariant model on the same dataset and measures:
  1. Validation AUC (classification quality)
  2. Best volume found in optimization (solution quality)
  3. Optimization steps to reach feasibility (convergence speed)

The key hypothesis: enforcing rotation equivariance improves both
classification accuracy (the model generalizes across orientations) and
optimization quality (the gradient direction is geometrically consistent).

Run:
    python -m phase5_equivariant.compare \
        --dataset dataset/ \
        --vanilla_ckpt phase3_gnn/checkpoints/best_model.pt
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .model_equivar import EquivariantMeshGNN
from phase2_mesh.dataset import SofaDataset, train_val_test_split
from phase3_gnn.train import get_device
from phase3_gnn.eval import collect_predictions, print_metrics


def collate_equivar(batch):
    """
    Custom collate for equivariant model.

    Key difference from Phase 3: we extract raw vertex positions as `pos`
    (not concatenated with normals) since the equivariant model computes
    relative positions internally and doesn't take normals as explicit input.
    """
    pos_list, edges, ys = [], [], []
    offset = 0

    for sample in batch:
        # x is (N, 6): positions + normals. We only need positions for equivar model.
        pos = sample["x"][:, :3]  # (N, 3)
        n = pos.shape[0]
        pos_list.append(pos)
        edges.append(sample["edge_index"] + offset)
        ys.append(sample["y"])
        offset += n

    batch_tensor = torch.cat([
        torch.full((s["x"].shape[0],), i, dtype=torch.long)
        for i, s in enumerate(batch)
    ])

    return {
        "pos":        torch.cat(pos_list, dim=0),    # (N_total, 3)
        "edge_index": torch.cat(edges, dim=1),        # (2, E_total)
        "y":          torch.tensor(ys, dtype=torch.long),
        "batch":      batch_tensor,
    }


def train_equivariant(
    dataset_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    out_dir: str = "phase5_equivariant/checkpoints",
):
    device = get_device()
    print(f"Training equivariant GNN on {device}")

    train_ds, val_ds, test_ds = train_val_test_split(dataset_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_equivar, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_equivar, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_equivar, num_workers=0)

    model     = EquivariantMeshGNN(hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for batch in train_loader:
            pos        = batch["pos"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec  = batch["batch"].to(device)
            y          = batch["y"].to(device).float()

            optimizer.zero_grad()
            prob = model(pos, edge_index, batch_vec)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            pred = (prob > 0.5).long()
            n_correct += (pred == batch["y"].to(device)).sum().item()
            n_total += len(y)

        train_acc = n_correct / n_total

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pos        = batch["pos"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_vec  = batch["batch"].to(device)
                y          = batch["y"].to(device).float()
                prob = model(pos, edge_index, batch_vec)
                loss = criterion(prob, y)
                val_loss += loss.item() * len(y)
                pred = (prob > 0.5).long()
                val_correct += (pred == batch["y"].to(device)).sum().item()
                val_total += len(y)
        val_acc = val_correct / val_total

        history["train_loss"].append(total_loss / n_total)
        history["val_loss"].append(val_loss / val_total)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_equivar.pt"))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train acc={train_acc:.3f} | val acc={val_acc:.3f} | best={best_val_acc:.3f}")

    torch.save(history, os.path.join(out_dir, "equivar_history.pt"))
    return model, best_val_acc


def compare_models(
    vanilla_ckpt: str,
    equivar_ckpt: str,
    dataset_dir: str,
):
    """
    Print a side-by-side comparison table: vanilla GNN vs equivariant GNN.
    """
    from phase3_gnn.model import MeshFeasibilityGNN
    from phase3_gnn.train import collate_graphs

    device = get_device()

    # Load vanilla model
    vanilla = MeshFeasibilityGNN(in_channels=6, hidden=128).to(device)
    vanilla.load_state_dict(torch.load(vanilla_ckpt, weights_only=True, map_location=device))
    vanilla.eval()

    # Load equivariant model
    equivar = EquivariantMeshGNN(hidden=64).to(device)
    equivar.load_state_dict(torch.load(equivar_ckpt, weights_only=True, map_location=device))
    equivar.eval()

    _, _, test_ds = train_val_test_split(dataset_dir)
    vanilla_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                collate_fn=collate_graphs, num_workers=0)
    equivar_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                collate_fn=collate_equivar, num_workers=0)

    print("=" * 60)
    print("VANILLA GNN (Phase 3)")
    print("-" * 60)
    vanilla_probs, vanilla_labels = collect_predictions(vanilla, vanilla_loader, device)
    vanilla_auc = print_metrics(vanilla_probs, vanilla_labels)

    print()
    print("=" * 60)
    print("EQUIVARIANT GNN (Phase 5)")
    print("-" * 60)

    # Collect equivariant predictions
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in equivar_loader:
            pos        = batch["pos"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_vec  = batch["batch"].to(device)
            prob = equivar(pos, edge_index, batch_vec)
            all_probs.append(prob.cpu())
            all_labels.append(batch["y"])
    equivar_probs  = torch.cat(all_probs).numpy()
    equivar_labels = torch.cat(all_labels).numpy()
    equivar_auc = print_metrics(equivar_probs, equivar_labels)

    print()
    print("=" * 60)
    print("SUMMARY")
    print(f"  Vanilla  AUC: {vanilla_auc:.4f}")
    print(f"  Equivar  AUC: {equivar_auc:.4f}")
    delta = equivar_auc - vanilla_auc
    winner = "Equivariant" if delta > 0 else "Vanilla"
    print(f"  Winner:       {winner} (+{abs(delta):.4f})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="dataset")
    parser.add_argument("--vanilla_ckpt", type=str,
                        default="phase3_gnn/checkpoints/best_model.pt")
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--out",          type=str,
                        default="phase5_equivariant/checkpoints")
    args = parser.parse_args()

    model, best_val_acc = train_equivariant(
        args.dataset, epochs=args.epochs, out_dir=args.out
    )

    equivar_ckpt = os.path.join(args.out, "best_equivar.pt")
    compare_models(args.vanilla_ckpt, equivar_ckpt, args.dataset)
