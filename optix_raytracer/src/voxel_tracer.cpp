#include "voxel_tracer.h"
#include <iostream>
#include <cstring>
#include <algorithm>

// PTX directory is defined by CMake
#ifndef PTX_DIR
#define PTX_DIR "./ptx"
#endif

VoxelRayTracer::VoxelRayTracer(const std::vector<unsigned char>& voxel_data,
                               int res_x, int res_y, int res_z,
                               float voxel_size)
    : m_voxel_data_host(voxel_data),
      m_res_x(res_x),
      m_res_y(res_y),
      m_res_z(res_z),
      m_voxel_size(voxel_size) {

    if (voxel_data.size() != static_cast<size_t>(res_x * res_y * res_z)) {
        throw std::runtime_error("Voxel data size doesn't match resolution");
    }

    initialize();
}

VoxelRayTracer::~VoxelRayTracer() {
    cleanup();
}

void VoxelRayTracer::initialize() {
    try {
        // Create OptiX setup
        m_optix_setup = std::make_unique<OptixSetup>();

        if (!m_optix_setup->initialize()) {
            throw std::runtime_error("Failed to initialize OptiX");
        }

        // Create pipeline
        std::string ptx_dir = PTX_DIR;
        if (!m_optix_setup->createPipeline(ptx_dir)) {
            throw std::runtime_error("Failed to create OptiX pipeline");
        }

        // Upload voxel data to GPU
        uploadVoxelData();

        // Build acceleration structure
        buildAccelerationStructure();

        // Setup launch parameters
        m_params.grid.resolution = make_int3(m_res_x, m_res_y, m_res_z);
        m_params.grid.voxel_size = make_float3(m_voxel_size, m_voxel_size, m_voxel_size);

        // Compute grid bounds (centered at origin)
        float half_width = (m_res_x * m_voxel_size) / 2.0f;
        float half_height = (m_res_y * m_voxel_size) / 2.0f;
        float half_depth = (m_res_z * m_voxel_size) / 2.0f;

        m_params.grid.grid_min = make_float3(-half_width, -half_height, -half_depth);
        m_params.grid.grid_max = make_float3(half_width, half_height, half_depth);

        m_params.grid.voxel_data = m_voxel_data_device;
        m_params.grid.handle = m_gas_handle;

        // Allocate device memory for params
        CUDA_CHECK(cudaMalloc(&m_params_device, sizeof(LaunchParams)));

        m_initialized = true;

    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

void VoxelRayTracer::uploadVoxelData() {
    size_t data_size = m_voxel_data_host.size();

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&m_voxel_data_device, data_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(m_voxel_data_device,
                          m_voxel_data_host.data(),
                          data_size,
                          cudaMemcpyHostToDevice));
}

void VoxelRayTracer::buildAccelerationStructure() {
    // For voxel grids, we create a simple AABB encompassing the entire grid
    // The actual traversal is done via DDA in the shader

    OptixAabb aabb;
    float half_width = (m_res_x * m_voxel_size) / 2.0f;
    float half_height = (m_res_y * m_voxel_size) / 2.0f;
    float half_depth = (m_res_z * m_voxel_size) / 2.0f;

    aabb.minX = -half_width;
    aabb.minY = -half_height;
    aabb.minZ = -half_depth;
    aabb.maxX = half_width;
    aabb.maxY = half_height;
    aabb.maxZ = half_depth;

    // Upload AABB to device
    void* d_aabb;
    CUDA_CHECK(cudaMalloc(&d_aabb, sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(d_aabb, &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));

    // Build input for acceleration structure
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    OptixBuildInputCustomPrimitiveArray& custom_prim_array = build_input.customPrimitiveArray;
    custom_prim_array.aabbBuffers = reinterpret_cast<CUdeviceptr*>(&d_aabb);
    custom_prim_array.numPrimitives = 1;

    uint32_t build_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    custom_prim_array.flags = build_flags;
    custom_prim_array.numSbtRecords = 1;

    // Acceleration structure options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_optix_setup->getContext(),
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));

    // Allocate temporary buffers
    void* d_temp_buffer;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(&m_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes));

    // Build acceleration structure
    OPTIX_CHECK(optixAccelBuild(
        m_optix_setup->getContext(),
        0,  // CUDA stream
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(d_temp_buffer),
        gas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(m_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes,
        &m_gas_handle,
        nullptr,
        0
    ));

    // Clean up temporary buffers
    CUDA_CHECK(cudaFree(d_temp_buffer));
    CUDA_CHECK(cudaFree(d_aabb));
}

std::vector<float> VoxelRayTracer::traceRays(const std::vector<float>& ray_origins,
                                              const std::vector<float>& ray_directions,
                                              int num_rays) {
    if (!m_initialized) {
        throw std::runtime_error("VoxelRayTracer not initialized");
    }

    if (ray_origins.size() != num_rays * 3 || ray_directions.size() != num_rays * 3) {
        throw std::runtime_error("Invalid ray data size");
    }

    // Allocate device memory for rays and output
    float3* d_origins;
    float3* d_directions;
    float* d_distances;

    CUDA_CHECK(cudaMalloc(&d_origins, num_rays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_directions, num_rays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_rays * sizeof(float)));

    // Copy ray data to device
    CUDA_CHECK(cudaMemcpy(d_origins, ray_origins.data(),
                          num_rays * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_directions, ray_directions.data(),
                          num_rays * sizeof(float3), cudaMemcpyHostToDevice));

    // Setup launch parameters
    m_params.ray_origins = d_origins;
    m_params.ray_directions = d_directions;
    m_params.output_distances = d_distances;
    m_params.num_rays = num_rays;

    // Compute image dimensions for 2D launch
    int width = static_cast<int>(std::ceil(std::sqrt(num_rays)));
    int height = (num_rays + width - 1) / width;
    m_params.image_width = width;
    m_params.image_height = height;

    // Copy params to device
    CUDA_CHECK(cudaMemcpy(m_params_device, &m_params,
                          sizeof(LaunchParams), cudaMemcpyHostToDevice));

    // Launch OptiX
    OPTIX_CHECK(optixLaunch(
        m_optix_setup->getPipeline(),
        0,  // CUDA stream
        reinterpret_cast<CUdeviceptr>(m_params_device),
        sizeof(LaunchParams),
        &m_optix_setup->getSBT(),
        width,
        height,
        1  // depth
    ));

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<float> distances(num_rays);
    CUDA_CHECK(cudaMemcpy(distances.data(), d_distances,
                          num_rays * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_origins));
    CUDA_CHECK(cudaFree(d_directions));
    CUDA_CHECK(cudaFree(d_distances));

    return distances;
}

void VoxelRayTracer::updateVoxelGrid(const std::vector<unsigned char>& voxel_data,
                                     int res_x, int res_y, int res_z) {
    if (voxel_data.size() != static_cast<size_t>(res_x * res_y * res_z)) {
        throw std::runtime_error("Voxel data size doesn't match resolution");
    }

    // Update host data
    m_voxel_data_host = voxel_data;
    m_res_x = res_x;
    m_res_y = res_y;
    m_res_z = res_z;

    // Free old device data
    if (m_voxel_data_device) {
        CUDA_CHECK(cudaFree(m_voxel_data_device));
        m_voxel_data_device = nullptr;
    }

    if (m_gas_output_buffer) {
        CUDA_CHECK(cudaFree(m_gas_output_buffer));
        m_gas_output_buffer = nullptr;
    }

    // Re-upload and rebuild
    uploadVoxelData();
    buildAccelerationStructure();

    // Update params
    m_params.grid.resolution = make_int3(m_res_x, m_res_y, m_res_z);
    m_params.grid.voxel_data = m_voxel_data_device;
    m_params.grid.handle = m_gas_handle;

    float half_width = (m_res_x * m_voxel_size) / 2.0f;
    float half_height = (m_res_y * m_voxel_size) / 2.0f;
    float half_depth = (m_res_z * m_voxel_size) / 2.0f;

    m_params.grid.grid_min = make_float3(-half_width, -half_height, -half_depth);
    m_params.grid.grid_max = make_float3(half_width, half_height, half_depth);
}

void VoxelRayTracer::getGridInfo(int& res_x, int& res_y, int& res_z,
                                 float& voxel_size) const {
    res_x = m_res_x;
    res_y = m_res_y;
    res_z = m_res_z;
    voxel_size = m_voxel_size;
}

void VoxelRayTracer::cleanup() {
    if (m_voxel_data_device) {
        cudaFree(m_voxel_data_device);
        m_voxel_data_device = nullptr;
    }

    if (m_gas_output_buffer) {
        cudaFree(m_gas_output_buffer);
        m_gas_output_buffer = nullptr;
    }

    if (m_params_device) {
        cudaFree(m_params_device);
        m_params_device = nullptr;
    }

    m_initialized = false;
}
