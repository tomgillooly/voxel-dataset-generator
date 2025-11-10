#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

/**
 * OptiX context manager for voxel ray tracing.
 * Handles initialization, pipeline creation, and cleanup.
 */
class OptixSetup {
public:
    OptixSetup();
    ~OptixSetup();

    // Initialize OptiX context
    bool initialize();

    // Create ray tracing pipeline for voxel grids
    bool createPipeline(const std::string& ptx_dir);

    // Get OptiX context
    OptixDeviceContext getContext() const { return m_context; }

    // Get pipeline
    OptixPipeline getPipeline() const { return m_pipeline; }

    // Get shader binding table
    OptixShaderBindingTable getSBT() const { return m_sbt; }

    // Check if initialized
    bool isInitialized() const { return m_initialized; }

private:
    // Callback for OptiX log messages
    static void contextLogCallback(unsigned int level, const char* tag,
                                   const char* message, void* cbdata);

    // Create modules from PTX
    bool createModules(const std::string& ptx_file);

    // Create program groups
    bool createProgramGroups();

    // Link pipeline
    bool linkPipeline();

    // Build shader binding table
    bool buildSBT();

    // Cleanup resources
    void cleanup();

    OptixDeviceContext m_context = nullptr;
    OptixPipeline m_pipeline = nullptr;
    OptixModule m_module = nullptr;

    // Program groups
    OptixProgramGroup m_raygen_prog_group = nullptr;
    OptixProgramGroup m_miss_prog_group = nullptr;
    OptixProgramGroup m_hitgroup_prog_group = nullptr;

    // Shader binding table
    OptixShaderBindingTable m_sbt = {};

    // Device memory for SBT records
    void* m_raygen_record = nullptr;
    void* m_miss_record = nullptr;
    void* m_hitgroup_record = nullptr;

    bool m_initialized = false;
    CUcontext m_cuda_context;
};

// Helper function to check OptiX errors
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::string error_msg = std::string("OptiX call failed: ") +       \
                                   optixGetErrorName(res) + " (" +             \
                                   optixGetErrorString(res) + ")";             \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if (error != cudaSuccess) {                                            \
            std::string error_msg = std::string("CUDA call failed: ") +        \
                                   cudaGetErrorName(error) + " (" +            \
                                   cudaGetErrorString(error) + ")";            \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)

// Helper function to check CUDA driver errors
#define CUDA_DRIVER_CHECK(call)                                                \
    do {                                                                       \
        CUresult res = call;                                                   \
        if (res != CUDA_SUCCESS) {                                             \
            const char* error_name;                                            \
            const char* error_string;                                          \
            cuGetErrorName(res, &error_name);                                  \
            cuGetErrorString(res, &error_string);                              \
            std::string error_msg = std::string("CUDA driver call failed: ") + \
                                   error_name + " (" + error_string + ")";     \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)
