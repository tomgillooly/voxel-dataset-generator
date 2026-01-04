"""Metal kernel source strings for voxel ray tracing.

All Metal code is stored as Python string constants for use with MLX's
mx.fast.metal_kernel() JIT compilation system.
"""

# =============================================================================
# Common Header - Shared structures and utilities
# =============================================================================

COMMON_HEADER = """
#include <metal_stdlib>
using namespace metal;

// Constants
constant float PI = 3.14159265358979323846f;
constant float INV_PI = 0.31830988618379067154f;
constant float TWO_PI = 6.28318530717958647692f;
constant float MAX_T = 1e20f;

// PCG random number generator
// Reference: https://jcgt.org/published/0009/03/02/paper.pdf
inline uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand_float(thread uint* state) {
    *state = pcg_hash(*state);
    return float(*state) / float(0xFFFFFFFFu);
}

inline float2 rand_float2(thread uint* state) {
    return float2(rand_float(state), rand_float(state));
}

// Utility functions
inline float3 safe_normalize(float3 v) {
    float len = length(v);
    return len > 1e-8f ? v / len : float3(0, 0, 1);
}

// Build orthonormal basis from a direction vector
inline void build_orthonormal_basis(float3 n, thread float3* tangent, thread float3* bitangent) {
    float3 up = abs(n.y) < 0.999f ? float3(0, 1, 0) : float3(1, 0, 0);
    *tangent = normalize(cross(up, n));
    *bitangent = cross(n, *tangent);
}

// Transform local direction to world space given basis
inline float3 local_to_world(float3 local_dir, float3 tangent, float3 bitangent, float3 normal) {
    return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal;
}
"""

# =============================================================================
# Spherical Harmonics Header and Kernel
# =============================================================================

SH_HEADER = """
// Spherical Harmonics Order 2 (9 coefficients)
// Indices: Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
//
// Reference: Green, "Spherical Harmonic Lighting: The Gritty Details", GDC 2003

// SH normalization constants
constant float SH_C0 = 0.282094791773878f;   // 1 / (2 * sqrt(pi))
constant float SH_C1 = 0.488602511902920f;   // sqrt(3 / (4 * pi))
constant float SH_C2_0 = 1.092548430592079f; // sqrt(15 / (4 * pi))
constant float SH_C2_1 = 0.315391565252520f; // sqrt(5 / (16 * pi))
constant float SH_C2_2 = 0.546274215296040f; // sqrt(15 / (16 * pi))

// Evaluate all 9 SH basis functions for a direction
inline void eval_sh_basis(float3 dir, thread float* coeffs) {
    float x = dir.x, y = dir.y, z = dir.z;

    // l=0 (1 coefficient)
    coeffs[0] = SH_C0;

    // l=1 (3 coefficients)
    coeffs[1] = SH_C1 * y;
    coeffs[2] = SH_C1 * z;
    coeffs[3] = SH_C1 * x;

    // l=2 (5 coefficients)
    coeffs[4] = SH_C2_0 * x * y;           // Y_2^-2
    coeffs[5] = SH_C2_0 * y * z;           // Y_2^-1
    coeffs[6] = SH_C2_1 * (3.0f * z * z - 1.0f);  // Y_2^0
    coeffs[7] = SH_C2_0 * x * z;           // Y_2^1
    coeffs[8] = SH_C2_2 * (x * x - y * y); // Y_2^2
}

// Evaluate a specific SH basis function (for importance sampling)
inline float eval_sh_basis_single(int idx, float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;

    switch (idx) {
        case 0: return SH_C0;
        case 1: return SH_C1 * y;
        case 2: return SH_C1 * z;
        case 3: return SH_C1 * x;
        case 4: return SH_C2_0 * x * y;
        case 5: return SH_C2_0 * y * z;
        case 6: return SH_C2_1 * (3.0f * z * z - 1.0f);
        case 7: return SH_C2_0 * x * z;
        case 8: return SH_C2_2 * (x * x - y * y);
        default: return 0.0f;
    }
}

// Sample direction from cosine-weighted hemisphere
inline float3 sample_cosine_hemisphere(float2 u) {
    float phi = TWO_PI * u.x;
    float cos_theta = sqrt(u.y);
    float sin_theta = sqrt(1.0f - u.y);

    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Sample direction uniformly on sphere
inline float3 sample_uniform_sphere(float2 u) {
    float z = 1.0f - 2.0f * u.x;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = TWO_PI * u.y;

    return float3(r * cos(phi), r * sin(phi), z);
}
"""

SH_KERNEL_SOURCE = """
// Kernel to evaluate SH basis for multiple directions
[[kernel]]
void eval_sh_kernel(
    device const float3* directions [[buffer(0)]],
    device float* sh_coeffs [[buffer(1)]],
    uint idx [[thread_position_in_grid]],
    constant uint& num_directions [[buffer(2)]]
) {
    if (idx >= num_directions) return;

    float3 dir = directions[idx];
    float coeffs[9];
    eval_sh_basis(dir, coeffs);

    // Write to output (9 coefficients per direction)
    for (int i = 0; i < 9; i++) {
        sh_coeffs[idx * 9 + i] = coeffs[i];
    }
}

// Kernel to project radiance values onto SH basis
[[kernel]]
void project_sh_kernel(
    device const float3* directions [[buffer(0)]],
    device const float* radiances [[buffer(1)]],
    device atomic_float* sh_coeffs [[buffer(2)]],
    uint idx [[thread_position_in_grid]],
    constant uint& num_samples [[buffer(3)]]
) {
    if (idx >= num_samples) return;

    float3 dir = directions[idx];
    float radiance = radiances[idx];

    float coeffs[9];
    eval_sh_basis(dir, coeffs);

    // Atomic accumulation (for parallel projection)
    float weight = 4.0f * PI / float(num_samples);  // Monte Carlo normalization
    for (int i = 0; i < 9; i++) {
        atomic_fetch_add_explicit(&sh_coeffs[i], radiance * coeffs[i] * weight, memory_order_relaxed);
    }
}
"""

# =============================================================================
# DDA Traversal Header and Kernel
# =============================================================================

DDA_HEADER = """
// DDA traversal result
struct DDAResult {
    float distance;      // Total distance through occupied voxels
    float t_exit;        // Ray parameter at exit point
    float3 exit_point;   // World space exit position
    int exit_face;       // Which face exited through (0-5: +X,-X,+Y,-Y,+Z,-Z)
    bool hit;            // Whether ray intersected the grid at all
};

// Voxel lookup with Z-Y-X row-major ordering (matching NumPy C-order)
inline bool is_voxel_occupied(
    device const uint8_t* voxels,
    int x, int y, int z,
    int3 resolution
) {
    if (x < 0 || x >= resolution.x ||
        y < 0 || y >= resolution.y ||
        z < 0 || z >= resolution.z) {
        return false;
    }
    int idx = z * (resolution.y * resolution.x) + y * resolution.x + x;
    return voxels[idx] != 0;
}

// DDA voxel traversal algorithm
// Adapted from optix_raytracer/cuda/voxel_programs.cu
inline DDAResult trace_dda(
    float3 ray_origin,
    float3 ray_direction,
    device const uint8_t* voxels,
    float3 grid_min,
    float3 grid_max,
    int3 resolution,
    float voxel_size
) {
    DDAResult result;
    result.distance = 0.0f;
    result.hit = false;
    result.exit_face = -1;

    // Normalize direction
    ray_direction = safe_normalize(ray_direction);

    // Compute inverse direction for slab intersection
    float3 inv_dir = float3(
        abs(ray_direction.x) > 1e-8f ? 1.0f / ray_direction.x : MAX_T,
        abs(ray_direction.y) > 1e-8f ? 1.0f / ray_direction.y : MAX_T,
        abs(ray_direction.z) > 1e-8f ? 1.0f / ray_direction.z : MAX_T
    );

    // Ray-box intersection (slab method)
    float3 t0 = (grid_min - ray_origin) * inv_dir;
    float3 t1 = (grid_max - ray_origin) * inv_dir;

    // Ensure t0 < t1
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);

    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);

    // No intersection
    if (t_enter > t_exit || t_exit < 0.0f) {
        return result;
    }

    result.hit = true;
    t_enter = max(t_enter, 0.0f);
    result.t_exit = t_exit;
    result.exit_point = ray_origin + ray_direction * t_exit;

    // Determine exit face
    if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
        result.exit_face = ray_direction.x > 0 ? 0 : 1;  // +X or -X
    } else if (tmax.y <= tmax.z) {
        result.exit_face = ray_direction.y > 0 ? 2 : 3;  // +Y or -Y
    } else {
        result.exit_face = ray_direction.z > 0 ? 4 : 5;  // +Z or -Z
    }

    // Entry point (slightly inside to avoid boundary issues)
    float3 entry_point = ray_origin + ray_direction * (t_enter + 1e-5f);

    // Convert to voxel coordinates
    float3 voxel_f = (entry_point - grid_min) / voxel_size;
    int3 voxel = int3(floor(voxel_f));

    // Clamp to grid bounds
    voxel = clamp(voxel, int3(0), resolution - 1);

    // Step direction
    int3 step = int3(
        ray_direction.x > 0 ? 1 : (ray_direction.x < 0 ? -1 : 0),
        ray_direction.y > 0 ? 1 : (ray_direction.y < 0 ? -1 : 0),
        ray_direction.z > 0 ? 1 : (ray_direction.z < 0 ? -1 : 0)
    );

    // tDelta: distance along ray to cross one voxel
    float3 tDelta = float3(
        step.x != 0 ? abs(voxel_size * inv_dir.x) : MAX_T,
        step.y != 0 ? abs(voxel_size * inv_dir.y) : MAX_T,
        step.z != 0 ? abs(voxel_size * inv_dir.z) : MAX_T
    );

    // tMax: t value at next voxel boundary
    float3 voxel_corner = grid_min + float3(voxel) * voxel_size;
    float3 tMax;
    tMax.x = step.x != 0 ?
        ((step.x > 0 ? voxel_corner.x + voxel_size : voxel_corner.x) - ray_origin.x) * inv_dir.x
        : MAX_T;
    tMax.y = step.y != 0 ?
        ((step.y > 0 ? voxel_corner.y + voxel_size : voxel_corner.y) - ray_origin.y) * inv_dir.y
        : MAX_T;
    tMax.z = step.z != 0 ?
        ((step.z > 0 ? voxel_corner.z + voxel_size : voxel_corner.z) - ray_origin.z) * inv_dir.z
        : MAX_T;

    // DDA traversal loop
    float accumulated = 0.0f;
    float current_t = t_enter;
    int max_steps = resolution.x + resolution.y + resolution.z;

    for (int i = 0; i < max_steps; i++) {
        // Bounds check
        if (any(voxel < 0) || any(voxel >= resolution)) {
            break;
        }

        // Find next t value
        float next_t = min(min(tMax.x, tMax.y), tMax.z);
        next_t = min(next_t, t_exit);

        // Accumulate distance if occupied
        if (is_voxel_occupied(voxels, voxel.x, voxel.y, voxel.z, resolution)) {
            accumulated += next_t - current_t;
        }

        // Step to next voxel
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            tMax.x += tDelta.x;
            voxel.x += step.x;
        } else if (tMax.y < tMax.z) {
            tMax.y += tDelta.y;
            voxel.y += step.y;
        } else {
            tMax.z += tDelta.z;
            voxel.z += step.z;
        }

        current_t = next_t;
        if (next_t >= t_exit) break;
    }

    result.distance = accumulated;
    return result;
}
"""

DDA_KERNEL_SOURCE = """
// Simple ray tracing kernel - returns distance through occupied voxels
[[kernel]]
void trace_rays_kernel(
    device const float* ray_origins [[buffer(0)]],
    device const float* ray_directions [[buffer(1)]],
    device const uint8_t* voxels [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant float3& grid_min [[buffer(4)]],
    constant float3& grid_max [[buffer(5)]],
    constant int3& resolution [[buffer(6)]],
    constant float& voxel_size [[buffer(7)]],
    constant uint& num_rays [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_rays) return;

    float3 origin = float3(
        ray_origins[idx * 3 + 0],
        ray_origins[idx * 3 + 1],
        ray_origins[idx * 3 + 2]
    );
    float3 direction = float3(
        ray_directions[idx * 3 + 0],
        ray_directions[idx * 3 + 1],
        ray_directions[idx * 3 + 2]
    );

    DDAResult result = trace_dda(origin, direction, voxels,
                                  grid_min, grid_max, resolution, voxel_size);

    distances[idx] = result.distance;
}

// Extended ray tracing kernel - returns distance and exit info
[[kernel]]
void trace_rays_extended_kernel(
    device const float* ray_origins [[buffer(0)]],
    device const float* ray_directions [[buffer(1)]],
    device const uint8_t* voxels [[buffer(2)]],
    device float* distances [[buffer(3)]],
    device float* exit_points [[buffer(4)]],
    device int* exit_faces [[buffer(5)]],
    device uint8_t* hits [[buffer(6)]],
    constant float3& grid_min [[buffer(7)]],
    constant float3& grid_max [[buffer(8)]],
    constant int3& resolution [[buffer(9)]],
    constant float& voxel_size [[buffer(10)]],
    constant uint& num_rays [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_rays) return;

    float3 origin = float3(
        ray_origins[idx * 3 + 0],
        ray_origins[idx * 3 + 1],
        ray_origins[idx * 3 + 2]
    );
    float3 direction = float3(
        ray_directions[idx * 3 + 0],
        ray_directions[idx * 3 + 1],
        ray_directions[idx * 3 + 2]
    );

    DDAResult result = trace_dda(origin, direction, voxels,
                                  grid_min, grid_max, resolution, voxel_size);

    distances[idx] = result.distance;
    exit_points[idx * 3 + 0] = result.exit_point.x;
    exit_points[idx * 3 + 1] = result.exit_point.y;
    exit_points[idx * 3 + 2] = result.exit_point.z;
    exit_faces[idx] = result.exit_face;
    hits[idx] = result.hit ? 1 : 0;
}
"""

# =============================================================================
# Monte Carlo Scattering Header and Kernel
# =============================================================================

MC_HEADER = """
// Henyey-Greenstein phase function sampling
// Reference: PBRT, Section 15.2.3
inline float3 sample_henyey_greenstein(float g, float2 u) {
    float cos_theta;

    if (abs(g) < 1e-4f) {
        // Isotropic case
        cos_theta = 1.0f - 2.0f * u.x;
    } else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * u.x);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }

    cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = TWO_PI * u.y;

    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Compute which boundary patch a point belongs to
inline int compute_exit_patch(
    float3 point,
    int exit_face,
    float3 grid_min,
    float3 grid_max,
    int patches_per_face
) {
    // Face ordering: +X(0), -X(1), +Y(2), -Y(3), +Z(4), -Z(5)
    // Each face has patches_per_face^2 patches

    float3 grid_size = grid_max - grid_min;
    float3 normalized = (point - grid_min) / grid_size;  // [0, 1]

    int patch_u, patch_v;

    switch (exit_face) {
        case 0: case 1:  // +X, -X: use Y, Z
            patch_u = clamp(int(normalized.y * patches_per_face), 0, patches_per_face - 1);
            patch_v = clamp(int(normalized.z * patches_per_face), 0, patches_per_face - 1);
            break;
        case 2: case 3:  // +Y, -Y: use X, Z
            patch_u = clamp(int(normalized.x * patches_per_face), 0, patches_per_face - 1);
            patch_v = clamp(int(normalized.z * patches_per_face), 0, patches_per_face - 1);
            break;
        case 4: case 5:  // +Z, -Z: use X, Y
            patch_u = clamp(int(normalized.x * patches_per_face), 0, patches_per_face - 1);
            patch_v = clamp(int(normalized.y * patches_per_face), 0, patches_per_face - 1);
            break;
        default:
            return 0;
    }

    int patch_idx = patch_v * patches_per_face + patch_u;
    return exit_face * patches_per_face * patches_per_face + patch_idx;
}

// Get patch center and normal
inline void get_patch_geometry(
    int patch_idx,
    int patches_per_face,
    float3 grid_min,
    float3 grid_max,
    thread float3* center,
    thread float3* normal
) {
    int patches_per_face_sq = patches_per_face * patches_per_face;
    int face = patch_idx / patches_per_face_sq;
    int local_idx = patch_idx % patches_per_face_sq;
    int patch_v = local_idx / patches_per_face;
    int patch_u = local_idx % patches_per_face;

    float3 grid_size = grid_max - grid_min;
    float patch_size_u, patch_size_v;
    float u_center, v_center;

    // Face normals (outward)
    const float3 face_normals[6] = {
        float3(1, 0, 0),   // +X
        float3(-1, 0, 0),  // -X
        float3(0, 1, 0),   // +Y
        float3(0, -1, 0),  // -Y
        float3(0, 0, 1),   // +Z
        float3(0, 0, -1)   // -Z
    };

    *normal = face_normals[face];
    *center = (grid_min + grid_max) * 0.5f;  // Start at center

    switch (face) {
        case 0:  // +X
            patch_size_u = grid_size.y / patches_per_face;
            patch_size_v = grid_size.z / patches_per_face;
            center->x = grid_max.x;
            center->y = grid_min.y + (patch_u + 0.5f) * patch_size_u;
            center->z = grid_min.z + (patch_v + 0.5f) * patch_size_v;
            break;
        case 1:  // -X
            patch_size_u = grid_size.y / patches_per_face;
            patch_size_v = grid_size.z / patches_per_face;
            center->x = grid_min.x;
            center->y = grid_min.y + (patch_u + 0.5f) * patch_size_u;
            center->z = grid_min.z + (patch_v + 0.5f) * patch_size_v;
            break;
        case 2:  // +Y
            patch_size_u = grid_size.x / patches_per_face;
            patch_size_v = grid_size.z / patches_per_face;
            center->y = grid_max.y;
            center->x = grid_min.x + (patch_u + 0.5f) * patch_size_u;
            center->z = grid_min.z + (patch_v + 0.5f) * patch_size_v;
            break;
        case 3:  // -Y
            patch_size_u = grid_size.x / patches_per_face;
            patch_size_v = grid_size.z / patches_per_face;
            center->y = grid_min.y;
            center->x = grid_min.x + (patch_u + 0.5f) * patch_size_u;
            center->z = grid_min.z + (patch_v + 0.5f) * patch_size_v;
            break;
        case 4:  // +Z
            patch_size_u = grid_size.x / patches_per_face;
            patch_size_v = grid_size.y / patches_per_face;
            center->z = grid_max.z;
            center->x = grid_min.x + (patch_u + 0.5f) * patch_size_u;
            center->y = grid_min.y + (patch_v + 0.5f) * patch_size_v;
            break;
        case 5:  // -Z
            patch_size_u = grid_size.x / patches_per_face;
            patch_size_v = grid_size.y / patches_per_face;
            center->z = grid_min.z;
            center->x = grid_min.x + (patch_u + 0.5f) * patch_size_u;
            center->y = grid_min.y + (patch_v + 0.5f) * patch_size_v;
            break;
    }
}
"""

MC_KERNEL_SOURCE = """
// Monte Carlo subsurface scattering kernel
// Traces rays from an incident patch and accumulates exitant SH coefficients
[[kernel]]
void mc_scatter_kernel(
    device const uint8_t* voxels [[buffer(0)]],
    device atomic_float* output_sh [[buffer(1)]],  // (n_patches, n_sh)
    constant float3& grid_min [[buffer(2)]],
    constant float3& grid_max [[buffer(3)]],
    constant int3& resolution [[buffer(4)]],
    constant float& voxel_size [[buffer(5)]],
    constant float& sigma_s [[buffer(6)]],
    constant float& sigma_a [[buffer(7)]],
    constant float& g [[buffer(8)]],
    constant int& incident_patch [[buffer(9)]],
    constant int& incident_sh_idx [[buffer(10)]],
    constant int& patches_per_face [[buffer(11)]],
    constant int& n_sh [[buffer(12)]],
    constant uint& num_samples [[buffer(13)]],
    constant uint& seed [[buffer(14)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_samples) return;

    // Initialize RNG with unique seed per sample
    uint rng_state = pcg_hash(idx ^ (seed * 0x9E3779B9u));

    // Get incident patch geometry
    float3 patch_center, patch_normal;
    get_patch_geometry(incident_patch, patches_per_face, grid_min, grid_max,
                       &patch_center, &patch_normal);

    // Sample incident direction (cosine-weighted hemisphere, inward)
    float2 u = rand_float2(&rng_state);
    float3 local_dir = sample_cosine_hemisphere(u);

    // Build basis for incident patch (normal points outward, we want inward)
    float3 tangent, bitangent;
    float3 inward_normal = -patch_normal;
    build_orthonormal_basis(inward_normal, &tangent, &bitangent);

    float3 direction = local_to_world(local_dir, tangent, bitangent, inward_normal);

    // Compute SH weight for this direction
    float sh_weight = eval_sh_basis_single(incident_sh_idx, direction);

    // Start ray slightly inside the volume
    float3 origin = patch_center + direction * 1e-4f;

    // Optical properties
    float sigma_t = sigma_s + sigma_a;
    float albedo = sigma_s / sigma_t;
    float mfp = 1.0f / sigma_t;

    // Path tracing
    float weight = sh_weight;
    const int MAX_BOUNCES = 256;
    const float RUSSIAN_ROULETTE_THRESHOLD = 0.01f;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        // Sample free-flight distance (exponential distribution)
        float t_scatter = -log(max(rand_float(&rng_state), 1e-10f)) * mfp;

        // Trace through voxel grid
        DDAResult dda = trace_dda(origin, direction, voxels,
                                   grid_min, grid_max, resolution, voxel_size);

        if (!dda.hit) {
            // Ray missed the volume entirely (shouldn't happen from inside)
            break;
        }

        if (t_scatter < dda.distance) {
            // Scattering event inside volume
            origin = origin + direction * t_scatter;

            // Apply albedo (survival probability)
            weight *= albedo;

            // Sample new direction using Henyey-Greenstein
            float2 xi = rand_float2(&rng_state);
            float3 local_scatter = sample_henyey_greenstein(g, xi);

            // Transform to world space
            build_orthonormal_basis(direction, &tangent, &bitangent);
            direction = local_to_world(local_scatter, tangent, bitangent, direction);

        } else {
            // Ray exits volume - record contribution
            int exit_patch = compute_exit_patch(dda.exit_point, dda.exit_face,
                                                 grid_min, grid_max, patches_per_face);

            // Cosine term for exitant direction
            float3 exit_normal;
            float3 exit_center;
            get_patch_geometry(exit_patch, patches_per_face, grid_min, grid_max,
                               &exit_center, &exit_normal);

            float cos_theta = abs(dot(direction, exit_normal));

            // Project exitant direction to SH basis
            float sh_coeffs[9];
            eval_sh_basis(direction, sh_coeffs);

            // Atomic add to output (weighted contribution)
            for (int i = 0; i < n_sh; i++) {
                int out_idx = exit_patch * n_sh + i;
                atomic_fetch_add_explicit(&output_sh[out_idx],
                                          weight * cos_theta * sh_coeffs[i],
                                          memory_order_relaxed);
            }
            break;
        }

        // Russian roulette for path termination
        if (weight < RUSSIAN_ROULETTE_THRESHOLD && bounce > 3) {
            if (rand_float(&rng_state) > weight / RUSSIAN_ROULETTE_THRESHOLD) {
                break;
            }
            weight = RUSSIAN_ROULETTE_THRESHOLD;
        }
    }
}

// Batch version: process multiple incident conditions in parallel
[[kernel]]
void mc_scatter_batch_kernel(
    device const uint8_t* voxels [[buffer(0)]],
    device atomic_float* output_sh [[buffer(1)]],  // (n_conditions, n_patches, n_sh)
    constant float3& grid_min [[buffer(2)]],
    constant float3& grid_max [[buffer(3)]],
    constant int3& resolution [[buffer(4)]],
    constant float& voxel_size [[buffer(5)]],
    constant float& sigma_s [[buffer(6)]],
    constant float& sigma_a [[buffer(7)]],
    constant float& g [[buffer(8)]],
    device const int* incident_patches [[buffer(9)]],
    device const int* incident_sh_indices [[buffer(10)]],
    constant int& patches_per_face [[buffer(11)]],
    constant int& n_sh [[buffer(12)]],
    constant int& n_patches [[buffer(13)]],
    constant uint& samples_per_condition [[buffer(14)]],
    constant uint& seed [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]  // (sample_idx, condition_idx)
) {
    uint sample_idx = gid.x;
    uint condition_idx = gid.y;

    if (sample_idx >= samples_per_condition) return;

    int incident_patch = incident_patches[condition_idx];
    int incident_sh_idx = incident_sh_indices[condition_idx];

    // Initialize RNG
    uint rng_state = pcg_hash(sample_idx ^ (condition_idx * 0x85EBCA6Bu) ^ (seed * 0x9E3779B9u));

    // Get incident patch geometry
    float3 patch_center, patch_normal;
    get_patch_geometry(incident_patch, patches_per_face, grid_min, grid_max,
                       &patch_center, &patch_normal);

    // Sample incident direction
    float2 u = rand_float2(&rng_state);
    float3 local_dir = sample_cosine_hemisphere(u);

    float3 tangent, bitangent;
    float3 inward_normal = -patch_normal;
    build_orthonormal_basis(inward_normal, &tangent, &bitangent);

    float3 direction = local_to_world(local_dir, tangent, bitangent, inward_normal);
    float sh_weight = eval_sh_basis_single(incident_sh_idx, direction);

    float3 origin = patch_center + direction * 1e-4f;

    // Optical properties
    float sigma_t = sigma_s + sigma_a;
    float albedo = sigma_s / sigma_t;
    float mfp = 1.0f / sigma_t;

    // Path tracing
    float weight = sh_weight;
    const int MAX_BOUNCES = 256;
    const float RUSSIAN_ROULETTE_THRESHOLD = 0.01f;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        float t_scatter = -log(max(rand_float(&rng_state), 1e-10f)) * mfp;

        DDAResult dda = trace_dda(origin, direction, voxels,
                                   grid_min, grid_max, resolution, voxel_size);

        if (!dda.hit) break;

        if (t_scatter < dda.distance) {
            origin = origin + direction * t_scatter;
            weight *= albedo;

            float2 xi = rand_float2(&rng_state);
            float3 local_scatter = sample_henyey_greenstein(g, xi);

            build_orthonormal_basis(direction, &tangent, &bitangent);
            direction = local_to_world(local_scatter, tangent, bitangent, direction);

        } else {
            int exit_patch = compute_exit_patch(dda.exit_point, dda.exit_face,
                                                 grid_min, grid_max, patches_per_face);

            float3 exit_normal, exit_center;
            get_patch_geometry(exit_patch, patches_per_face, grid_min, grid_max,
                               &exit_center, &exit_normal);

            float cos_theta = abs(dot(direction, exit_normal));

            float sh_coeffs[9];
            eval_sh_basis(direction, sh_coeffs);

            // Output index: condition_idx * (n_patches * n_sh) + exit_patch * n_sh + i
            int base_idx = condition_idx * n_patches * n_sh + exit_patch * n_sh;
            for (int i = 0; i < n_sh; i++) {
                atomic_fetch_add_explicit(&output_sh[base_idx + i],
                                          weight * cos_theta * sh_coeffs[i],
                                          memory_order_relaxed);
            }
            break;
        }

        if (weight < RUSSIAN_ROULETTE_THRESHOLD && bounce > 3) {
            if (rand_float(&rng_state) > weight / RUSSIAN_ROULETTE_THRESHOLD) {
                break;
            }
            weight = RUSSIAN_ROULETTE_THRESHOLD;
        }
    }
}
"""
