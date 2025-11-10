#include "optix_setup.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

OptixSetup::OptixSetup() : m_cuda_context(nullptr) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&m_cuda_context));
}

OptixSetup::~OptixSetup() {
    cleanup();
}

void OptixSetup::contextLogCallback(unsigned int level, const char* tag,
                                    const char* message, void* cbdata) {
    std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
}

bool OptixSetup::initialize() {
    if (m_initialized) {
        return true;
    }

    try {
        // Initialize OptiX
        OPTIX_CHECK(optixInit());

        // Create OptiX device context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptixSetup::contextLogCallback;
        options.logCallbackLevel = 4;  // Maximum verbosity

        OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, &options, &m_context));

        m_initialized = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize OptiX: " << e.what() << std::endl;
        return false;
    }
}

bool OptixSetup::createPipeline(const std::string& ptx_dir) {
    if (!m_initialized) {
        std::cerr << "OptiX not initialized" << std::endl;
        return false;
    }

    try {
        std::string ptx_file = ptx_dir + "/voxel_programs.ptx";

        // Create modules
        if (!createModules(ptx_file)) {
            return false;
        }

        // Create program groups
        if (!createProgramGroups()) {
            return false;
        }

        // Link pipeline
        if (!linkPipeline()) {
            return false;
        }

        // Build SBT
        if (!buildSBT()) {
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create pipeline: " << e.what() << std::endl;
        return false;
    }
}

bool OptixSetup::createModules(const std::string& ptx_file) {
    // Read PTX file
    std::ifstream ptx_stream(ptx_file);
    if (!ptx_stream.is_open()) {
        std::cerr << "Failed to open PTX file: " << ptx_file << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << ptx_stream.rdbuf();
    std::string ptx_code = buffer.str();

    // Module compile options
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    // Pipeline compile options
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;  // For RayPayload
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    // Create module
    char log[2048];
    size_t log_size = sizeof(log);

    // Use optixModuleCreate for OptiX 8.0+, optixModuleCreateFromPTX for OptiX 7.x
    #if OPTIX_VERSION >= 80000
    OPTIX_CHECK(optixModuleCreate(
        m_context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &log_size,
        &m_module
    ));
    #else
    OPTIX_CHECK(optixModuleCreateFromPTX(
        m_context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &log_size,
        &m_module
    ));
    #endif

    if (log_size > 1) {
        std::cout << "Module creation log:\n" << log << std::endl;
    }

    return true;
}

bool OptixSetup::createProgramGroups() {
    OptixProgramGroupOptions program_group_options = {};
    char log[2048];
    size_t log_size;

    // Ray generation program group
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = m_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &raygen_prog_group_desc,
        1,
        &program_group_options,
        log,
        &log_size,
        &m_raygen_prog_group
    ));

    // Miss program group
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = m_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &miss_prog_group_desc,
        1,
        &program_group_options,
        log,
        &log_size,
        &m_miss_prog_group
    ));

    // Hit group
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        m_context,
        &hitgroup_prog_group_desc,
        1,
        &program_group_options,
        log,
        &log_size,
        &m_hitgroup_prog_group
    ));

    return true;
}

bool OptixSetup::linkPipeline() {
    OptixProgramGroup program_groups[] = {
        m_raygen_prog_group,
        m_miss_prog_group,
        m_hitgroup_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    // debugLevel was removed in OptiX 8.0
    #if OPTIX_VERSION < 80000
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #endif

    char log[2048];
    size_t log_size = sizeof(log);

    // Pipeline creation API changed in OptiX 8.0
    #if OPTIX_VERSION >= 80000
    // OptiX 8.0+ requires pipeline compile options
    OptixPipelineCompileOptions pipeline_compile_options_for_link = {};
    pipeline_compile_options_for_link.usesMotionBlur = false;
    pipeline_compile_options_for_link.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_for_link.numPayloadValues = 2;
    pipeline_compile_options_for_link.numAttributeValues = 2;
    pipeline_compile_options_for_link.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_for_link.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixPipelineCreate(
        m_context,
        &pipeline_compile_options_for_link,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &log_size,
        &m_pipeline
    ));
    #else
    // OptiX 7.x
    OPTIX_CHECK(optixPipelineCreate(
        m_context,
        nullptr,  // No compile options needed here
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &log_size,
        &m_pipeline
    ));
    #endif

    if (log_size > 1) {
        std::cout << "Pipeline creation log:\n" << log << std::endl;
    }

    return true;
}

bool OptixSetup::buildSBT() {
    // Raygen record
    struct RaygenRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    RaygenRecord raygen_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &raygen_record));

    CUDA_CHECK(cudaMalloc(&m_raygen_record, sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(m_raygen_record, &raygen_record, sizeof(RaygenRecord),
                          cudaMemcpyHostToDevice));

    // Miss record
    struct MissRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    MissRecord miss_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group, &miss_record));

    CUDA_CHECK(cudaMalloc(&m_miss_record, sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(m_miss_record, &miss_record, sizeof(MissRecord),
                          cudaMemcpyHostToDevice));

    // Hit group record
    struct HitgroupRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    HitgroupRecord hitgroup_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_prog_group, &hitgroup_record));

    CUDA_CHECK(cudaMalloc(&m_hitgroup_record, sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(m_hitgroup_record, &hitgroup_record, sizeof(HitgroupRecord),
                          cudaMemcpyHostToDevice));

    // Build SBT
    m_sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(m_raygen_record);
    m_sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(m_miss_record);
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_hitgroup_record);
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = 1;

    return true;
}

void OptixSetup::cleanup() {
    if (m_raygen_record) cudaFree(m_raygen_record);
    if (m_miss_record) cudaFree(m_miss_record);
    if (m_hitgroup_record) cudaFree(m_hitgroup_record);

    if (m_pipeline) optixPipelineDestroy(m_pipeline);
    if (m_raygen_prog_group) optixProgramGroupDestroy(m_raygen_prog_group);
    if (m_miss_prog_group) optixProgramGroupDestroy(m_miss_prog_group);
    if (m_hitgroup_prog_group) optixProgramGroupDestroy(m_hitgroup_prog_group);
    if (m_module) optixModuleDestroy(m_module);
    if (m_context) optixDeviceContextDestroy(m_context);

    m_initialized = false;
}
