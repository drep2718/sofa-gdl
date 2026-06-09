"""
Evaluation utilities for the trained GNN.

Produces:
  1. AUC-ROC score
  2. Confusion matrix (TP/FP/TN/FN)
  3. Precision-recall curve
  4. Calibration check: does P(feasible) correlate with true SDF margin?

Run from project root after training:
    python -m phase3_gnn.eval --checkpoint phase3_gnn/checkpoints/best_model.pt \
                               --dataset dataset/
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_recall_curve,
    RocCurveDisplay, ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from .model import MeshFeasibilityGNN
from .train import collate_graphs, get_device
from phase2_mesh.dataset import train_val_test_split


def load_model(checkpoint_path: str, device: torch.device) -> MeshFeasibilityGNN:
    model = MeshFeasibilityGNN(in_channels=6, hidden=128)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on all batches, return stacked probs and labels."""
    all_probs, all_labels = [], []
    for batch in loader:
        x          = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        batch_vec  = batch["batch"].to(device)
        prob = model(x, edge_index, batch_vec)
        all_probs.append(prob.cpu())
        all_labels.append(batch["y"])
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def print_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    preds = (probs > threshold).astype(int)
    auc   = roc_auc_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion matrix (threshold={threshold}):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    return auc


def plot_evaluation(
    probs: np.ndarray,
    labels: np.ndarray,
    history_path: str = None,
    save_dir: str = "phase3_gnn/results",
):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, fig)

    # ── 1. ROC curve ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    fpr = []
    tpr = []
    thresholds = np.linspace(0, 1, 100)
    for t in thresholds:
        preds = (probs > t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        fpr.append(fp / (fp + tn + 1e-8))
        tpr.append(tp / (tp + fn + 1e-8))
    auc = roc_auc_score(labels, probs)
    ax1.plot(fpr, tpr, color="steelblue", lw=2)
    ax1.plot([0,1], [0,1], "k--", lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title(f"ROC (AUC={auc:.3f})")
    ax1.grid(True, alpha=0.3)

    # ── 2. Precision-Recall curve ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)
    ax2.plot(rec_vals, prec_vals, color="tomato", lw=2)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall")
    ax2.grid(True, alpha=0.3)

    # ── 3. Confusion matrix heatmap ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    im = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(["Pred 0","Pred 1"])
    ax3.set_yticklabels(["True 0","True 1"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=12)
    ax3.set_title("Confusion Matrix")

    # ── 4. Training history (if available) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    if history_path and os.path.exists(history_path):
        history = torch.load(history_path, weights_only=True)
        ax4.plot(history["train_acc"], label="train acc", color="steelblue")
        ax4.plot(history["val_acc"],   label="val acc",   color="tomato")
        ax4.set_xlabel("Epoch"); ax4.set_ylabel("Accuracy")
        ax4.set_title("Training Curves")
        ax4.legend(); ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No history\navailable", ha="center", va="center")
        ax4.set_title("Training Curves")

    plt.tight_layout()
    path = os.path.join(save_dir, "evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation plot to {path}")
    plt.show()


def evaluate_model(
    checkpoint_path: str,
    dataset_dir: str,
    save_dir: str = "phase3_gnn/results",
):
    device = get_device()
    model = load_model(checkpoint_path, device)

    _, _, test_ds = train_val_test_split(dataset_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=collate_graphs, num_workers=0)

    print(f"Evaluating on {len(test_ds)} test samples...")
    probs, labels = collect_predictions(model, test_loader, device)

    print_metrics(probs, labels)

    history_path = os.path.join(os.path.dirname(checkpoint_path), "history.pt")
    plot_evaluation(probs, labels, history_path, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="phase3_gnn/checkpoints/best_model.pt")
    parser.add_argument("--dataset",    type=str, default="dataset")
    parser.add_argument("--out",        type=str, default="phase3_gnn/results")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.dataset, args.out)
