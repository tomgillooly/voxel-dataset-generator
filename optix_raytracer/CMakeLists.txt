cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(optix_voxel_tracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# OptiX SDK
if(NOT DEFINED OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
endif()

if(NOT OptiX_INSTALL_DIR)
    message(FATAL_ERROR "OptiX_INSTALL_DIR not set. Please set it to your OptiX SDK installation directory.")
endif()

set(OptiX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/include)
if(NOT EXISTS ${OptiX_INCLUDE_DIR}/optix.h)
    message(FATAL_ERROR "OptiX headers not found at ${OptiX_INCLUDE_DIR}")
endif()

message(STATUS "Found OptiX: ${OptiX_INCLUDE_DIR}")
message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")

# pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OptiX_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # Turing, Ampere, Ada Lovelace

# Generate PTX from CUDA files
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/ptx)
file(MAKE_DIRECTORY ${CUDA_GENERATED_OUTPUT_DIR})

# Custom command to compile CUDA to PTX
set(CUDA_PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/voxel_programs.cu
)

foreach(cuda_file ${CUDA_PROGRAMS})
    get_filename_component(cuda_file_base ${cuda_file} NAME_WE)
    set(ptx_file ${CUDA_GENERATED_OUTPUT_DIR}/${cuda_file_base}.ptx)

    add_custom_command(
        OUTPUT ${ptx_file}
        COMMAND ${CMAKE_CUDA_COMPILER}
                --ptx
                -O3
                --use_fast_math
                -I${OptiX_INCLUDE_DIR}
                -I${CUDA_INCLUDE_DIRS}
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                --gpu-architecture=compute_75
                -o ${ptx_file}
                ${cuda_file}
        DEPENDS ${cuda_file}
        COMMENT "Compiling ${cuda_file} to PTX"
    )

    list(APPEND PTX_FILES ${ptx_file})
endforeach()

add_custom_target(ptx_target ALL DEPENDS ${PTX_FILES})

# Core library sources
set(CORE_SOURCES
    src/voxel_tracer.cpp
    src/optix_setup.cpp
)

set(CORE_HEADERS
    include/voxel_tracer.h
    include/optix_setup.h
    include/voxel_common.h
)

# Core library
add_library(optix_voxel_tracer_core SHARED ${CORE_SOURCES} ${CORE_HEADERS})
target_link_libraries(optix_voxel_tracer_core
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)
target_compile_definitions(optix_voxel_tracer_core PRIVATE
    PTX_DIR="${CUDA_GENERATED_OUTPUT_DIR}"
)
add_dependencies(optix_voxel_tracer_core ptx_target)

# Python module
pybind11_add_module(optix_voxel_tracer
    python/bindings.cpp
)

target_link_libraries(optix_voxel_tracer PRIVATE
    optix_voxel_tracer_core
    ${CUDA_LIBRARIES}
)

add_dependencies(optix_voxel_tracer ptx_target)

# Install
install(TARGETS optix_voxel_tracer_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS optix_voxel_tracer
    LIBRARY DESTINATION ${Python_SITEARCH}
)

install(DIRECTORY ${CUDA_GENERATED_OUTPUT_DIR}/
    DESTINATION share/optix_voxel_tracer/ptx
    FILES_MATCHING PATTERN "*.ptx"
)

# Copy PTX files to build directory for development
add_custom_command(TARGET optix_voxel_tracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CUDA_GENERATED_OUTPUT_DIR}
    $<TARGET_FILE_DIR:optix_voxel_tracer>/ptx
)

# Print configuration summary
message(STATUS "=== Configuration Summary ===")
message(STATUS "OptiX SDK: ${OptiX_INSTALL_DIR}")
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "PTX output: ${CUDA_GENERATED_OUTPUT_DIR}")
message(STATUS "============================")
