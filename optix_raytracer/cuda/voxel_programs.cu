#include <optix.h>
#include <optix_device.h>
#include "voxel_common.h"

// Launch parameters in constant memory
extern "C" { __constant__ LaunchParams params; }

/**
 * DDA (Digital Differential Analyzer) voxel traversal algorithm.
 * Efficiently steps through voxels along a ray and accumulates distance
 * traveled through occupied voxels.
 */
__device__ float trace_voxel_grid(const float3& ray_origin,
                                   const float3& ray_direction,
                                   const VoxelGridParams& grid) {
    // Compute entry and exit points with grid bounding box
    float3 inv_dir = make_float3(1.0f / ray_direction.x,
                                  1.0f / ray_direction.y,
                                  1.0f / ray_direction.z);

    // Compute intersections with bounding box
    float3 t0 = make_float3(
        (grid.grid_min.x - ray_origin.x) * inv_dir.x,
        (grid.grid_min.y - ray_origin.y) * inv_dir.y,
        (grid.grid_min.z - ray_origin.z) * inv_dir.z
    );

    float3 t1 = make_float3(
        (grid.grid_max.x - ray_origin.x) * inv_dir.x,
        (grid.grid_max.y - ray_origin.y) * inv_dir.y,
        (grid.grid_max.z - ray_origin.z) * inv_dir.z
    );

    // Ensure t0 < t1
    if (t0.x > t1.x) { float tmp = t0.x; t0.x = t1.x; t1.x = tmp; }
    if (t0.y > t1.y) { float tmp = t0.y; t0.y = t1.y; t1.y = tmp; }
    if (t0.z > t1.z) { float tmp = t0.z; t0.z = t1.z; t1.z = tmp; }

    // Compute entry and exit t values
    float t_enter = fmaxf(fmaxf(t0.x, t0.y), t0.z);
    float t_exit = fminf(fminf(t1.x, t1.y), t1.z);

    // Check if ray intersects grid
    if (t_enter > t_exit || t_exit < 0.0f) {
        return 0.0f;  // No intersection
    }

    // Ensure we start from a valid point
    t_enter = fmaxf(t_enter, 0.0f);

    // Entry point
    float3 entry_point = ray_origin + ray_direction * (t_enter + 1e-5f);

    // Convert to voxel coordinates
    int3 voxel = world_to_voxel(entry_point, grid.grid_min, grid.voxel_size);

    // Clamp to grid bounds
    voxel.x = max(0, min(voxel.x, grid.resolution.x - 1));
    voxel.y = max(0, min(voxel.y, grid.resolution.y - 1));
    voxel.z = max(0, min(voxel.z, grid.resolution.z - 1));

    // Step direction
    int3 step = make_int3(
        ray_direction.x > 0 ? 1 : (ray_direction.x < 0 ? -1 : 0),
        ray_direction.y > 0 ? 1 : (ray_direction.y < 0 ? -1 : 0),
        ray_direction.z > 0 ? 1 : (ray_direction.z < 0 ? -1 : 0)
    );

    // tDelta: distance along ray to cross one voxel boundary
    float3 tDelta = make_float3(
        fabsf(grid.voxel_size.x * inv_dir.x),
        fabsf(grid.voxel_size.y * inv_dir.y),
        fabsf(grid.voxel_size.z * inv_dir.z)
    );

    // tMax: t value at next voxel boundary for each axis
    float3 voxel_corner = make_float3(
        grid.grid_min.x + voxel.x * grid.voxel_size.x,
        grid.grid_min.y + voxel.y * grid.voxel_size.y,
        grid.grid_min.z + voxel.z * grid.voxel_size.z
    );

    float3 tMax;
    tMax.x = (step.x > 0) ?
        ((voxel_corner.x + grid.voxel_size.x - ray_origin.x) * inv_dir.x) :
        ((voxel_corner.x - ray_origin.x) * inv_dir.x);
    tMax.y = (step.y > 0) ?
        ((voxel_corner.y + grid.voxel_size.y - ray_origin.y) * inv_dir.y) :
        ((voxel_corner.y - ray_origin.y) * inv_dir.y);
    tMax.z = (step.z > 0) ?
        ((voxel_corner.z + grid.voxel_size.z - ray_origin.z) * inv_dir.z) :
        ((voxel_corner.z - ray_origin.z) * inv_dir.z);

    // DDA traversal
    float accumulated_distance = 0.0f;
    float current_t = t_enter;

    const int max_steps = grid.resolution.x + grid.resolution.y + grid.resolution.z;
    int steps = 0;

    while (steps < max_steps) {
        // Check if we're still inside the grid
        if (voxel.x < 0 || voxel.x >= grid.resolution.x ||
            voxel.y < 0 || voxel.y >= grid.resolution.y ||
            voxel.z < 0 || voxel.z >= grid.resolution.z) {
            break;
        }

        // Determine next t value (when we'll exit this voxel)
        float next_t;
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            next_t = tMax.x;
        } else if (tMax.y < tMax.z) {
            next_t = tMax.y;
        } else {
            next_t = tMax.z;
        }

        // Clamp to grid exit
        if (next_t > t_exit) {
            next_t = t_exit;
        }

        // Check if CURRENT voxel is occupied and accumulate distance through it
        if (is_voxel_occupied(voxel.x, voxel.y, voxel.z, grid)) {
            float segment_length = next_t - current_t;
            accumulated_distance += segment_length;
        }

        // Move to next voxel
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

        if (next_t >= t_exit) {
            break;
        }

        steps++;
    }

    return accumulated_distance;
}

/**
 * Ray generation program - launches rays and writes results
 */
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Compute linear index for this ray
    int ray_idx = idx.y * dim.x + idx.x;

    if (ray_idx >= params.num_rays) {
        return;
    }

    // Get ray origin and direction
    float3 ray_origin = params.ray_origins[ray_idx];
    float3 ray_direction = params.ray_directions[ray_idx];

    // Normalize direction (in case it wasn't already)
    ray_direction = normalize(ray_direction);

    // Trace through voxel grid using DDA
    float distance = trace_voxel_grid(ray_origin, ray_direction, params.grid);

    // Write result
    params.output_distances[ray_idx] = distance;
}

/**
 * Miss program - called when ray doesn't hit anything
 */
extern "C" __global__ void __miss__ms() {
    // For voxel gracing, we handle everything in raygen
    // This shouldn't be called in our current setup
    RayPayload* payload = reinterpret_cast<RayPayload*>(
        optixGetPayload_0()
    );
    payload->accumulated_distance = 0.0f;
    payload->hit = false;
}

/**
 * Closest hit program - called when ray hits geometry
 */
extern "C" __global__ void __closesthit__ch() {
    // For voxel tracing, we handle everything in raygen
    // This is here for compatibility but shouldn't be used
    RayPayload* payload = reinterpret_cast<RayPayload*>(
        optixGetPayload_0()
    );
    payload->hit = true;
}

/**
 * Intersection program for custom primitives (optional)
 * This could be used for more complex voxel representations
 */
extern "C" __global__ void __intersection__is() {
    // Custom intersection logic would go here
    // For now, we use DDA in raygen instead
}
