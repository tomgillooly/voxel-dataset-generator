#pragma once

#include <optix.h>
#include <cuda_runtime.h>

// Voxel grid parameters stored in constant memory
struct VoxelGridParams {
    unsigned char* voxel_data;  // 3D grid of occupancy (0 or 1)
    int3 resolution;             // Grid dimensions (x, y, z)
    float3 voxel_size;          // Size of each voxel
    float3 grid_min;            // Minimum corner of grid in world space
    float3 grid_max;            // Maximum corner of grid in world space
    OptixTraversableHandle handle;
};

// Ray payload for distance accumulation
struct RayPayload {
    float accumulated_distance;  // Total distance through occupied voxels
    bool hit;                    // Whether ray hit the grid
};

// Launch parameters passed to OptiX programs
struct LaunchParams {
    // Ray data
    float3* ray_origins;      // Array of ray origins [num_rays]
    float3* ray_directions;   // Array of ray directions [num_rays]
    float* output_distances;  // Output accumulated distances [num_rays]

    // Grid parameters
    VoxelGridParams grid;

    // Launch dimensions
    int num_rays;
    int image_width;
    int image_height;
};

// Voxel intersection attributes
struct VoxelIntersectionAttribs {
    float3 entry_point;  // Where ray enters voxel
    float3 exit_point;   // Where ray exits voxel
    float t_entry;       // Ray parameter at entry
    float t_exit;        // Ray parameter at exit
};

// Constants
constexpr float RAY_TMIN = 1e-4f;
constexpr float RAY_TMAX = 1e16f;

// Ray type indices
enum { RAY_TYPE_RADIANCE = 0, RAY_TYPE_COUNT };

// Helper functions (implemented in device code)
#ifdef __CUDACC__

// Convert 3D voxel coordinates to linear index
// NumPy arrays are stored in C-order (row-major): [z][y][x]
// So linear index is: z * (res_y * res_x) + y * res_x + x
__device__ __forceinline__
int voxel_index(int x, int y, int z, const int3& resolution) {
    return z * (resolution.y * resolution.x) + y * resolution.x + x;
}

// Check if a point is inside the grid bounds
__device__ __forceinline__
bool is_inside_grid(const float3& point, const float3& grid_min, const float3& grid_max) {
    return point.x >= grid_min.x && point.x <= grid_max.x &&
           point.y >= grid_min.y && point.y <= grid_max.y &&
           point.z >= grid_min.z && point.z <= grid_max.z;
}

// Convert world space position to voxel coordinates
__device__ __forceinline__
int3 world_to_voxel(const float3& world_pos, const float3& grid_min, const float3& voxel_size) {
    return make_int3(
        static_cast<int>((world_pos.x - grid_min.x) / voxel_size.x),
        static_cast<int>((world_pos.y - grid_min.y) / voxel_size.y),
        static_cast<int>((world_pos.z - grid_min.z) / voxel_size.z)
    );
}

// Check if voxel is occupied
__device__ __forceinline__
bool is_voxel_occupied(int x, int y, int z, const VoxelGridParams& grid) {
    if (x < 0 || x >= grid.resolution.x ||
        y < 0 || y >= grid.resolution.y ||
        z < 0 || z >= grid.resolution.z) {
        return false;
    }
    int idx = voxel_index(x, y, z, grid.resolution);
    return grid.voxel_data[idx] != 0;
}

// Vector operations
__device__ __forceinline__
float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__
float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__
float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__
float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__
float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__
float3 normalize(const float3& v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

#endif // __CUDACC__
