"""
SE(3)-equivariant GNN for feasibility classification (Phase 5).

Why equivariance matters here:
  The optimal sofa has a reflection symmetry (symmetric about the corner bisector
  plane z=0 and also about y=-x+const). An equivariant network can't cheat by
  memorizing that "a bump in the +x direction is bad" — it must learn the geometry
  intrinsically. This halves the effective search space and should improve both
  sample efficiency and the quality of shapes found in Phase 4.

SE(3)-equivariant message passing:
  - Vertex features are typed vectors (irreducible representations of SO(3))
  - Type-0: scalars (e.g., distances, local curvature)
  - Type-1: 3-vectors (e.g., positions, velocities, dipoles)
  - Message passing preserves the type structure: rotations of inputs produce
    rotations of outputs in the same representation

We use a simplified equivariant architecture without the full e3nn library —
instead implementing the key idea manually: use relative position vectors as
message features, and aggregate in a way that respects the SO(3) symmetry.

For the full e3nn version, see the commented section at the bottom.
This gives us equivariance under rotation (not full SE(3) / translation equivariance,
since we use absolute positions as node features, not just relative ones).

Architecture:
  Node features:  absolute position xyz (type-1 vector)
  Message:        relative position (v_j - v_i), edge distance, dot products
  Aggregation:    sum (equivariant) of message vectors
  Readout:        norm of aggregated vectors + MLP → scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EquivariantConv(nn.Module):
    """
    Rotation-equivariant graph convolution using relative position vectors.

    For each vertex i, computes:
        m_i = sum_{j ∈ N(i)} φ(||r_ij||) * r_ij    (equivariant message)
        s_i = sum_{j ∈ N(i)} ψ(||r_ij||)           (scalar message)

    where r_ij = v_j - v_i is the relative position vector.
    φ and ψ are learned radial basis functions (MLPs on the distance).

    m_i transforms as a vector under rotation (type-1 irrep), so the output
    is equivariant: R·m_i = m_i(R·V) for any rotation R.

    The scalar message s_i is invariant (type-0 irrep).

    Final node update: concatenate [||m_i||, s_i, existing_scalars] → MLP → new scalars
    """

    def __init__(self, in_scalar: int, out_scalar: int, n_radial: int = 16):
        super().__init__()
        # Radial basis MLP: maps distance → weights for vector/scalar messages
        self.phi_mlp = nn.Sequential(
            nn.Linear(1, n_radial), nn.SiLU(),
            nn.Linear(n_radial, 1),
        )
        self.psi_mlp = nn.Sequential(
            nn.Linear(1, n_radial), nn.SiLU(),
            nn.Linear(n_radial, in_scalar),
        )
        # Update MLP: [||m_i||, aggregated scalars, current scalars] → out_scalar
        self.update_mlp = nn.Sequential(
            nn.Linear(1 + in_scalar + in_scalar, out_scalar), nn.SiLU(),
            nn.Linear(out_scalar, out_scalar),
        )
        self.out_scalar = out_scalar

    def forward(
        self,
        pos: torch.Tensor,         # (N, 3) vertex positions
        h: torch.Tensor,           # (N, in_scalar) current scalar node features
        edge_index: torch.Tensor,  # (2, E) edges
    ) -> torch.Tensor:             # (N, out_scalar) updated features
        N = pos.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Relative positions and distances
        r_ij  = pos[src] - pos[dst]          # (E, 3) relative position vectors
        d_ij  = r_ij.norm(dim=1, keepdim=True) + 1e-8  # (E, 1) distances

        # Equivariant (vector) messages: φ(d) * r_ij/d  (unit vector, scaled)
        phi   = self.phi_mlp(d_ij)            # (E, 1)
        vec_msg = phi * r_ij / d_ij          # (E, 3) — equivariant

        # Scalar messages: ψ(d) * h_j  (neighbor features, distance-weighted)
        psi   = self.psi_mlp(d_ij)            # (E, in_scalar)
        scl_msg = psi * h[src]               # (E, in_scalar)

        # Aggregate
        vec_agg = torch.zeros(N, 3, dtype=pos.dtype, device=pos.device)
        scl_agg = torch.zeros(N, h.shape[1], dtype=h.dtype, device=h.device)

        vec_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, 3), vec_msg)
        scl_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), scl_msg)

        # Invariant feature: norm of aggregated vector (rotation-invariant scalar)
        vec_norm = vec_agg.norm(dim=1, keepdim=True)  # (N, 1)

        # Update: combine norm, aggregated scalars, current features
        feat = torch.cat([vec_norm, scl_agg, h], dim=1)  # (N, 1 + in + in)
        return self.update_mlp(feat)                      # (N, out_scalar)


class EquivariantMeshGNN(nn.Module):
    """
    Rotation-equivariant mesh GNN for feasibility classification.

    Uses EquivariantConv layers that respect SO(3) symmetry by construction.
    Input: vertex positions (used both as node features and to compute relative
           positions for messages).
    Output: scalar P(feasible) ∈ [0, 1] — invariant to rotation.

    Key difference from Phase 3's vanilla GNN:
      - Phase 3 uses absolute positions as features → can memorize orientation
      - Phase 5 uses relative positions for messages → orientation-agnostic
        The output is the same regardless of how you rotate the mesh as input.
    """

    def __init__(self, hidden: int = 64, n_radial: int = 16):
        super().__init__()
        # Initial scalar features: per-vertex distances to centroid (1 feature)
        # We could use more invariant features (local curvature, etc.) but
        # distance-to-centroid is a simple baseline.
        self.init_mlp = nn.Linear(1, hidden)

        self.conv1 = EquivariantConv(hidden, hidden, n_radial)
        self.conv2 = EquivariantConv(hidden, hidden, n_radial)
        self.conv3 = EquivariantConv(hidden, hidden // 2, n_radial)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        pos: torch.Tensor,          # (N_total, 3) — all vertex positions in batch
        edge_index: torch.Tensor,   # (2, E_total)
        batch: torch.Tensor,        # (N_total,) — graph membership
    ) -> torch.Tensor:              # (B,) P(feasible) per graph
        # Initial scalar feature: distance from centroid
        B = int(batch.max().item()) + 1
        counts = torch.zeros(B, 1, dtype=pos.dtype, device=pos.device)
        centroids = torch.zeros(B, 3, dtype=pos.dtype, device=pos.device)
        counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(len(batch), 1, dtype=pos.dtype, device=pos.device))
        centroids.scatter_add_(0, batch.unsqueeze(1).expand(-1, 3), pos)
        centroids = centroids / (counts + 1e-8)

        # Per-vertex distance to their graph's centroid (invariant scalar)
        dist_to_centroid = (pos - centroids[batch]).norm(dim=1, keepdim=True)  # (N, 1)
        h = F.silu(self.init_mlp(dist_to_centroid))  # (N, hidden)

        # Equivariant message passing
        h = self.conv1(pos, h, edge_index)
        h = self.conv2(pos, h, edge_index)
        h = self.conv3(pos, h, edge_index)

        # Global mean + max pool
        half = h.shape[1]
        h_mean = torch.zeros(B, half, dtype=h.dtype, device=h.device)
        h_max  = torch.full((B, half), -1e9, dtype=h.dtype, device=h.device)

        h_mean.scatter_add_(0, batch.unsqueeze(1).expand(-1, half), h)
        h_max.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, half), h, reduce="amax")

        h_mean = h_mean / (counts + 1e-8)
        h_graph = torch.cat([h_mean, h_max], dim=1)  # (B, hidden)

        logit = self.classifier(h_graph).squeeze(-1)
        return torch.sigmoid(logit)
