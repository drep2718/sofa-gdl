"""
Mesh GNN feasibility classifier.

Architecture: simplified DiffusionNet-style graph network using
PyTorch Geometric's SAGEConv (GraphSAGE convolution).

Why SAGEConv over vanilla GCN?
  - GCN requires normalized adjacency (varies by graph), making batching awkward
  - SAGEConv uses mean aggregation + separate self/neighbor transforms — works
    cleanly across meshes with different vertex counts
  - DiffusionNet proper uses Laplacian eigenvectors (expensive to compute);
    our simplified version uses SAGEConv which captures multi-hop diffusion
    via message passing depth

Input per vertex: [x, y, z, nx, ny, nz] — position + surface normal (6 features)
Output: scalar P(feasible) ∈ [0, 1]

Message passing flow:
  (N, 6) → Conv1 → (N, 128) → Conv2 → (N, 128) → Conv3 → (N, 64)
         → global {mean, max} pool → (128,) → MLP → (1,) → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEConv(nn.Module):
    """
    Hand-rolled GraphSAGE convolution (avoids PyG dependency for now).

    h_v = ReLU( W_self * h_v + W_neigh * mean_{u ∈ N(v)} h_u + b )

    We can swap this for PyG's SAGEConv later without changing the architecture.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_self  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: torch.Tensor,           # (N, in_channels)
        edge_index: torch.Tensor,  # (2, E) — [src, dst]
    ) -> torch.Tensor:             # (N, out_channels)
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Aggregate: mean of neighbor features for each vertex
        neigh_sum   = torch.zeros(N, x.shape[1], dtype=x.dtype, device=x.device)
        neigh_count = torch.zeros(N, 1,           dtype=x.dtype, device=x.device)

        neigh_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
        neigh_count.scatter_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, dtype=x.dtype, device=x.device))

        neigh_mean = neigh_sum / (neigh_count + 1e-8)  # (N, in_channels)

        return F.relu(self.W_self(x) + self.W_neigh(neigh_mean) + self.bias)


class MeshFeasibilityGNN(nn.Module):
    """
    3-layer graph network that predicts P(shape is feasible for corner navigation).

    After 3 rounds of message passing, we pool all vertex representations into
    a single graph-level vector, then classify with an MLP.

    The {mean, max} global pooling captures both the average geometry AND the
    most extreme features (worst-case protruding vertices).
    """

    def __init__(
        self,
        in_channels: int = 6,    # xyz + normals
        hidden: int = 128,
        out_hidden: int = 64,
    ):
        super().__init__()

        self.conv1 = GraphSAGEConv(in_channels, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)

        self.conv2 = GraphSAGEConv(hidden, hidden)
        self.bn2   = nn.BatchNorm1d(hidden)

        self.conv3 = GraphSAGEConv(hidden, out_hidden)
        self.bn3   = nn.BatchNorm1d(out_hidden)

        # After {mean, max} pool: 2 * out_hidden features
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,            # (N_total, 6) — all vertices in batch
        edge_index: torch.Tensor,   # (2, E_total) — all edges in batch
        batch: torch.Tensor,        # (N_total,) — which graph each vertex belongs to
    ) -> torch.Tensor:              # (B,) — P(feasible) per graph
        # Message passing
        x = self.bn1(self.conv1(x, edge_index))
        x = self.bn2(self.conv2(x, edge_index))
        x = self.bn3(self.conv3(x, edge_index))

        # Global pooling: mean and max per graph
        B = batch.max().item() + 1
        h_mean = torch.zeros(B, x.shape[1], dtype=x.dtype, device=x.device)
        h_max  = torch.full((B, x.shape[1]), -1e9, dtype=x.dtype, device=x.device)

        h_mean.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.shape[1]), x)
        h_max.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, x.shape[1]), x, reduce="amax")

        # Normalize mean by count
        counts = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(len(batch), 1, dtype=x.dtype, device=x.device))
        h_mean = h_mean / (counts + 1e-8)

        h = torch.cat([h_mean, h_max], dim=1)  # (B, 2*out_hidden)
        logit = self.mlp(h).squeeze(-1)         # (B,)
        return torch.sigmoid(logit)             # (B,) ∈ [0, 1]

    def predict_single(
        self,
        V: torch.Tensor,           # (N, 3) — single mesh vertices
        normals: torch.Tensor,     # (N, 3) — per-vertex normals
        F: torch.Tensor,           # (M, 3) — face indices
        device: torch.device = None,
    ) -> float:
        """
        Predict feasibility probability for a single mesh.
        Convenience method used in Phase 4 optimization.
        """
        if device is None:
            device = next(self.parameters()).device

        x = torch.cat([V, normals], dim=1).to(device)  # (N, 6)

        edges = torch.cat([
            F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
            F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]],
        ], dim=0).T.to(device)  # (2, 6M)

        batch = torch.zeros(V.shape[0], dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            prob = self.forward(x, edges, batch)
        return prob.item()
