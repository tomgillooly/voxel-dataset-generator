#pragma once

#include "optix_setup.h"
#include "voxel_common.h"
#include <vector>
#include <memory>

/**
 * High-level interface for voxel ray tracing.
 * Manages voxel grid data and provides ray tracing functionality.
 */
class VoxelRayTracer {
public:
    /**
     * Constructor
     * @param voxel_data Flattened voxel grid (row-major, Z-Y-X order)
     * @param resolution Grid dimensions (x, y, z)
     * @param voxel_size Physical size of each voxel (default: 1.0)
     */
    VoxelRayTracer(const std::vector<unsigned char>& voxel_data,
                   int res_x, int res_y, int res_z,
                   float voxel_size = 1.0f);

    ~VoxelRayTracer();

    /**
     * Trace rays through the voxel grid and accumulate distances.
     * @param ray_origins Array of ray origins [num_rays * 3]
     * @param ray_directions Array of ray directions [num_rays * 3] (should be normalized)
     * @param num_rays Number of rays to trace
     * @return Vector of accumulated distances (one per ray)
     */
    std::vector<float> traceRays(const std::vector<float>& ray_origins,
                                 const std::vector<float>& ray_directions,
                                 int num_rays);

    /**
     * Update the voxel grid without recreating the entire tracer.
     * @param voxel_data New voxel grid data
     * @param resolution New grid dimensions
     */
    void updateVoxelGrid(const std::vector<unsigned char>& voxel_data,
                        int res_x, int res_y, int res_z);

    /**
     * Get grid information
     */
    void getGridInfo(int& res_x, int& res_y, int& res_z,
                    float& voxel_size) const;

    /**
     * Check if tracer is ready
     */
    bool isReady() const { return m_initialized; }

private:
    // Initialize OptiX and upload voxel data
    void initialize();

    // Upload voxel data to GPU
    void uploadVoxelData();

    // Create acceleration structure (AABB for grid bounding box)
    void buildAccelerationStructure();

    // Free GPU resources
    void cleanup();

    // OptiX setup
    std::unique_ptr<OptixSetup> m_optix_setup;

    // Voxel grid data (CPU)
    std::vector<unsigned char> m_voxel_data_host;
    int m_res_x, m_res_y, m_res_z;
    float m_voxel_size;

    // GPU memory
    unsigned char* m_voxel_data_device = nullptr;

    // Acceleration structure
    OptixTraversableHandle m_gas_handle = 0;
    void* m_gas_output_buffer = nullptr;

    // Launch parameters
    LaunchParams m_params;
    void* m_params_device = nullptr;

    bool m_initialized = false;
};
