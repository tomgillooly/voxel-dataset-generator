"""Boundary patch discretization for voxel grid surfaces.

Divides the 6 faces of a voxel grid bounding box into patches for
computing light transport transfer matrices.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class BoundaryPatch:
    """A single boundary patch on the voxel grid surface.

    Attributes:
        patch_id: Global patch index (0 to n_patches-1)
        face_id: Face index (0-5: +X, -X, +Y, -Y, +Z, -Z)
        local_idx: Index within the face
        center: World space center position (3,)
        normal: Outward-pointing unit normal (3,)
        area: Patch area in world units
        u_range: (u_min, u_max) in face-local coordinates
        v_range: (v_min, v_max) in face-local coordinates
    """
    patch_id: int
    face_id: int
    local_idx: int
    center: np.ndarray
    normal: np.ndarray
    area: float
    u_range: Tuple[float, float]
    v_range: Tuple[float, float]


# Face definitions: (axis, sign, u_axis, v_axis)
# axis: which axis the face is perpendicular to (0=X, 1=Y, 2=Z)
# sign: +1 for positive face, -1 for negative face
# u_axis, v_axis: which axes define the local UV coordinates
FACE_DEFINITIONS = [
    (0, +1, 1, 2),  # Face 0: +X, Y=u, Z=v
    (0, -1, 1, 2),  # Face 1: -X, Y=u, Z=v
    (1, +1, 0, 2),  # Face 2: +Y, X=u, Z=v
    (1, -1, 0, 2),  # Face 3: -Y, X=u, Z=v
    (2, +1, 0, 1),  # Face 4: +Z, X=u, Y=v
    (2, -1, 0, 1),  # Face 5: -Z, X=u, Y=v
]

# Face normals (outward-pointing)
FACE_NORMALS = np.array([
    [+1, 0, 0],   # +X
    [-1, 0, 0],   # -X
    [0, +1, 0],   # +Y
    [0, -1, 0],   # -Y
    [0, 0, +1],   # +Z
    [0, 0, -1],   # -Z
], dtype=np.float32)


def generate_boundary_patches(
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    patches_per_face: int = 8
) -> List[BoundaryPatch]:
    """Generate boundary patches on voxel grid surface.

    Creates patches_per_face x patches_per_face patches on each of 6 faces.
    Total patches = 6 * patches_per_face^2 = 384 for patches_per_face=8.

    Patch ordering:
    - Face 0 (+X): patches 0 to patches_per_face^2 - 1
    - Face 1 (-X): next patches_per_face^2 patches
    - ... and so on

    Within each face, patches are ordered row-major (v-major) in face-local UV.

    Args:
        grid_min: Minimum corner of bounding box (3,)
        grid_max: Maximum corner of bounding box (3,)
        patches_per_face: Number of patches per axis per face

    Returns:
        List of BoundaryPatch objects, length 6 * patches_per_face^2
    """
    grid_min = np.asarray(grid_min, dtype=np.float32)
    grid_max = np.asarray(grid_max, dtype=np.float32)
    grid_size = grid_max - grid_min

    patches = []
    global_idx = 0

    for face_id, (axis, sign, u_axis, v_axis) in enumerate(FACE_DEFINITIONS):
        # Face position along normal axis
        face_pos = grid_max[axis] if sign > 0 else grid_min[axis]

        # UV extent on this face
        u_min, u_max = grid_min[u_axis], grid_max[u_axis]
        v_min, v_max = grid_min[v_axis], grid_max[v_axis]

        u_step = (u_max - u_min) / patches_per_face
        v_step = (v_max - v_min) / patches_per_face
        area = u_step * v_step

        # Normal for this face
        normal = FACE_NORMALS[face_id].copy()

        for vi in range(patches_per_face):
            for ui in range(patches_per_face):
                # Patch center in UV space
                u_center = u_min + (ui + 0.5) * u_step
                v_center = v_min + (vi + 0.5) * v_step

                # World space center
                center = np.zeros(3, dtype=np.float32)
                center[axis] = face_pos
                center[u_axis] = u_center
                center[v_axis] = v_center

                patch = BoundaryPatch(
                    patch_id=global_idx,
                    face_id=face_id,
                    local_idx=vi * patches_per_face + ui,
                    center=center,
                    normal=normal,
                    area=area,
                    u_range=(u_min + ui * u_step, u_min + (ui + 1) * u_step),
                    v_range=(v_min + vi * v_step, v_min + (vi + 1) * v_step),
                )
                patches.append(patch)
                global_idx += 1

    return patches


def patches_to_arrays(patches: List[BoundaryPatch]) -> Dict[str, np.ndarray]:
    """Convert patch list to numpy arrays for kernel use.

    Args:
        patches: List of BoundaryPatch objects

    Returns:
        Dictionary with:
        - 'centers': (n_patches, 3) float32
        - 'normals': (n_patches, 3) float32
        - 'areas': (n_patches,) float32
        - 'face_ids': (n_patches,) int32
    """
    n_patches = len(patches)

    centers = np.zeros((n_patches, 3), dtype=np.float32)
    normals = np.zeros((n_patches, 3), dtype=np.float32)
    areas = np.zeros(n_patches, dtype=np.float32)
    face_ids = np.zeros(n_patches, dtype=np.int32)

    for i, patch in enumerate(patches):
        centers[i] = patch.center
        normals[i] = patch.normal
        areas[i] = patch.area
        face_ids[i] = patch.face_id

    return {
        'centers': centers,
        'normals': normals,
        'areas': areas,
        'face_ids': face_ids,
    }


def get_patch_info(
    patch_idx: int,
    patches_per_face: int = 8
) -> Tuple[int, int, int]:
    """Get face and local indices from global patch index.

    Args:
        patch_idx: Global patch index
        patches_per_face: Patches per axis per face

    Returns:
        Tuple of (face_id, patch_u, patch_v)
    """
    patches_per_face_sq = patches_per_face * patches_per_face
    face_id = patch_idx // patches_per_face_sq
    local_idx = patch_idx % patches_per_face_sq
    patch_v = local_idx // patches_per_face
    patch_u = local_idx % patches_per_face
    return face_id, patch_u, patch_v


def get_opposite_face(face_id: int) -> int:
    """Get the opposite face index.

    Args:
        face_id: Face index (0-5)

    Returns:
        Opposite face index
    """
    # +X <-> -X, +Y <-> -Y, +Z <-> -Z
    return face_id ^ 1  # XOR with 1 flips the sign bit


def sample_point_on_patch(
    patch: BoundaryPatch,
    u: float = 0.5,
    v: float = 0.5
) -> np.ndarray:
    """Sample a point on a patch surface.

    Args:
        patch: Boundary patch
        u: U coordinate in [0, 1]
        v: V coordinate in [0, 1]

    Returns:
        World space position (3,)
    """
    u_world = patch.u_range[0] + u * (patch.u_range[1] - patch.u_range[0])
    v_world = patch.v_range[0] + v * (patch.v_range[1] - patch.v_range[0])

    face_def = FACE_DEFINITIONS[patch.face_id]
    axis, sign, u_axis, v_axis = face_def

    point = np.zeros(3, dtype=np.float32)
    point[axis] = patch.center[axis]  # On the face
    point[u_axis] = u_world
    point[v_axis] = v_world

    return point


def find_exit_patch(
    point: np.ndarray,
    direction: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    patches_per_face: int = 8
) -> int:
    """Find which patch a ray exits through.

    Args:
        point: Exit point on bounding box surface (3,)
        direction: Ray direction (for determining which face)
        grid_min: Minimum corner of bounding box
        grid_max: Maximum corner of bounding box
        patches_per_face: Patches per axis per face

    Returns:
        Patch index (0 to n_patches-1)
    """
    # Determine which face based on which coordinate is at boundary
    eps = 1e-6
    grid_size = grid_max - grid_min

    # Find exit face
    exit_face = -1
    if abs(point[0] - grid_max[0]) < eps:
        exit_face = 0  # +X
    elif abs(point[0] - grid_min[0]) < eps:
        exit_face = 1  # -X
    elif abs(point[1] - grid_max[1]) < eps:
        exit_face = 2  # +Y
    elif abs(point[1] - grid_min[1]) < eps:
        exit_face = 3  # -Y
    elif abs(point[2] - grid_max[2]) < eps:
        exit_face = 4  # +Z
    elif abs(point[2] - grid_min[2]) < eps:
        exit_face = 5  # -Z
    else:
        # Point not on boundary, use ray direction to determine
        # which face would be hit
        t_to_faces = np.inf * np.ones(6)
        inv_dir = np.where(np.abs(direction) > 1e-8, 1.0 / direction, np.inf)

        # +X, -X
        t_to_faces[0] = (grid_max[0] - point[0]) * inv_dir[0]
        t_to_faces[1] = (grid_min[0] - point[0]) * inv_dir[0]
        # +Y, -Y
        t_to_faces[2] = (grid_max[1] - point[1]) * inv_dir[1]
        t_to_faces[3] = (grid_min[1] - point[1]) * inv_dir[1]
        # +Z, -Z
        t_to_faces[4] = (grid_max[2] - point[2]) * inv_dir[2]
        t_to_faces[5] = (grid_min[2] - point[2]) * inv_dir[2]

        # Only consider faces we're moving towards (t > 0)
        t_to_faces = np.where(t_to_faces > 0, t_to_faces, np.inf)
        exit_face = int(np.argmin(t_to_faces))

    # Get UV axes for this face
    face_def = FACE_DEFINITIONS[exit_face]
    _, _, u_axis, v_axis = face_def

    # Compute UV coordinates
    normalized = (point - grid_min) / grid_size
    u = np.clip(normalized[u_axis], 0, 1 - 1e-6)
    v = np.clip(normalized[v_axis], 0, 1 - 1e-6)

    patch_u = int(u * patches_per_face)
    patch_v = int(v * patches_per_face)

    patch_u = min(patch_u, patches_per_face - 1)
    patch_v = min(patch_v, patches_per_face - 1)

    # Compute global patch index
    patches_per_face_sq = patches_per_face * patches_per_face
    return exit_face * patches_per_face_sq + patch_v * patches_per_face + patch_u


def compute_patch_solid_angle(
    patch: BoundaryPatch,
    from_point: np.ndarray
) -> float:
    """Compute solid angle of patch as seen from a point.

    Args:
        patch: Boundary patch
        from_point: Observer position (3,)

    Returns:
        Solid angle in steradians
    """
    # Vector from point to patch center
    to_patch = patch.center - from_point
    distance_sq = np.sum(to_patch ** 2)
    distance = np.sqrt(distance_sq)

    if distance < 1e-8:
        return 2 * np.pi  # At the patch, see hemisphere

    # Direction to patch
    direction = to_patch / distance

    # Cosine of angle between direction and patch normal
    cos_angle = -np.dot(direction, patch.normal)

    if cos_angle <= 0:
        return 0.0  # Looking at back of patch

    # Solid angle = A * cos(theta) / r^2
    return patch.area * cos_angle / distance_sq
