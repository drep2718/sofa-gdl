"""
3D mesh generation via icosphere deformation.

Strategy: start from a geodesic sphere (icosphere), apply smooth random
displacement fields to get varied shapes. The displacement is smooth because
we use a sum of low-frequency spherical harmonics — this prevents jagged meshes
that would cause SDF artifacts and make the GNN harder to train.

Why icosphere? It has roughly uniform vertex distribution, which makes
graph convolutions well-conditioned. A cube-subdivided sphere has clustering
at corners, which biases learning.

Key types:
  vertices  V: (N, 3) float32 — 3D positions of mesh vertices
  faces     F: (M, 3) int64   — triangles, each row is 3 vertex indices
"""

import torch
import numpy as np
from typing import Tuple


def icosphere(subdivisions: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a geodesic icosphere by subdividing an icosahedron.

    subdivisions=0 → 12 vertices, 20 faces
    subdivisions=1 → 42 vertices, 80 faces
    subdivisions=2 → 162 vertices, 320 faces
    subdivisions=3 → 642 vertices, 1280 faces  ← default (good balance)

    Returns:
        V: (N, 3) vertex positions on the unit sphere
        F: (M, 3) face indices
    """
    # Golden ratio icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int64)

    midpoint_cache = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i,j), max(i,j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        idx = len(verts)
        midpoint_cache[key] = idx
        # Can't append to numpy array in loop — use list accumulation
        return idx

    # Rebuild with list to allow appending
    verts_list = list(verts)

    def midpoint_v2(i: int, j: int) -> int:
        key = (min(i,j), max(i,j))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (np.array(verts_list[i]) + np.array(verts_list[j])) / 2
        mid /= np.linalg.norm(mid)
        idx = len(verts_list)
        verts_list.append(mid)
        midpoint_cache[key] = idx
        return idx

    faces_list = list(faces)

    for _ in range(subdivisions):
        new_faces = []
        for tri in faces_list:
            a, b, c = tri
            ab = midpoint_v2(a, b)
            bc = midpoint_v2(b, c)
            ca = midpoint_v2(c, a)
            new_faces += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        faces_list = new_faces

    V = torch.tensor(np.array(verts_list), dtype=torch.float32)
    F = torch.tensor(np.array(faces_list), dtype=torch.long)
    return V, F


def deform_mesh(
    V: torch.Tensor,
    F: torch.Tensor,
    amplitude: float = 0.3,
    n_modes: int = 8,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a smooth random displacement to mesh vertices.

    The displacement is a sum of low-frequency sinusoidal modes in spherical
    coordinates. Using low modes keeps the mesh smooth and avoids high-frequency
    spikes that would make the SDF checker produce false positives.

    Args:
        V:         (N, 3) vertex positions (should be on/near unit sphere)
        F:         (M, 3) face indices (unchanged by deformation)
        amplitude: max displacement magnitude (relative to unit radius)
        n_modes:   number of random Fourier modes to combine
        seed:      random seed for reproducibility

    Returns:
        V_deformed: (N, 3) displaced vertex positions
        F:          (M, 3) same faces
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    V_np = V.numpy()
    N = V_np.shape[0]

    # Convert to spherical coords: (r, theta, phi)
    r = np.linalg.norm(V_np, axis=1, keepdims=True)  # (N,1) — should be ~1
    theta = np.arccos(np.clip(V_np[:, 2:3] / r, -1, 1))  # polar angle
    phi = np.arctan2(V_np[:, 1:2], V_np[:, 0:1])          # azimuthal angle

    # Sum of low-frequency modes
    displacement = np.zeros((N, 1), dtype=np.float64)
    for _ in range(n_modes):
        freq_theta = rng.randint(1, 4)
        freq_phi   = rng.randint(0, 4)
        phase      = rng.uniform(0, 2 * np.pi)
        coeff      = rng.uniform(-1, 1)
        displacement += coeff * np.sin(freq_theta * theta + phase) * np.cos(freq_phi * phi)

    # Normalize displacement to [-1, 1] and scale by amplitude
    max_abs = np.abs(displacement).max() + 1e-8
    displacement = displacement / max_abs * amplitude

    # Displace radially: new position = (r + displacement) * unit_normal
    unit_normals = V_np / (r + 1e-8)
    V_deformed = unit_normals * (r + displacement)

    return torch.tensor(V_deformed, dtype=torch.float32), F


def scale_mesh(
    V: torch.Tensor,
    F: torch.Tensor,
    target_scale: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly scale mesh to fit within the hallway (width=1 corridors).

    After scaling, the mesh radius ≈ target_scale so it nominally fits
    in a corridor of width 1 — though corner navigation may still fail.
    """
    current_radius = V.norm(dim=1).max().item()
    scale = target_scale / (current_radius + 1e-8)
    return V * scale, F


def random_mesh(
    subdivisions: int = 3,
    amplitude: float = None,
    seed: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single random deformed mesh ready for SDF checking.

    amplitude is sampled uniformly from [0.05, 0.45] if not provided,
    giving a range from nearly-sphere to highly-deformed blobs.

    Returns:
        V: (N, 3) vertex positions
        F: (M, 3) face indices
    """
    if seed is not None:
        np.random.seed(seed)
    if amplitude is None:
        amplitude = np.random.uniform(0.05, 0.45)

    V, F = icosphere(subdivisions)
    V, F = deform_mesh(V, F, amplitude=amplitude, n_modes=8)
    V, F = scale_mesh(V, F, target_scale=0.45)
    return V, F


def compute_vertex_normals(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Compute per-vertex normals by area-weighted averaging of face normals.

    These are used as GNN input features alongside vertex positions.

    Returns:
        normals: (N, 3) unit normals per vertex
    """
    v0 = V[F[:, 0]]  # (M, 3)
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (M, 3)
    face_areas   = face_normals.norm(dim=1, keepdim=True) + 1e-12
    face_normals_unit = face_normals / face_areas         # (M, 3) unit normals

    # Scatter face normals to vertices (area-weighted — face_normals before normalization)
    normals = torch.zeros_like(V)
    for i in range(3):
        normals.scatter_add_(0, F[:, i:i+1].expand(-1, 3), face_normals)

    norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return normals / norms
