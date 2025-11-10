# Architecture Overview

This document explains the technical architecture of the OptiX Voxel Ray Tracer.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  optix_voxel_tracer.VoxelRayTracer                  │   │
│  │  - NumPy array interface                            │   │
│  │  - trace_rays(origins, directions) → distances      │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ pybind11
┌──────────────────────▼──────────────────────────────────────┐
│                    C++ Host Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  VoxelRayTracer                                     │   │
│  │  - Manages voxel grid on GPU                        │   │
│  │  - Orchestrates ray launches                        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  OptixSetup                                         │   │
│  │  - OptiX context management                         │   │
│  │  - Pipeline creation                                │   │
│  │  - Shader binding table                             │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │ OptiX API
┌──────────────────────▼──────────────────────────────────────┐
│                  OptiX Runtime Layer                        │
│  - Ray scheduling and dispatch                              │
│  - Traversal and intersection                               │
│  - Program invocation                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ PTX execution
┌──────────────────────▼──────────────────────────────────────┐
│                 CUDA Device Programs                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  __raygen__rg                                       │   │
│  │  - Launches one ray per thread                      │   │
│  │  - Calls DDA traversal                              │   │
│  │  - Writes output distances                          │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  trace_voxel_grid (DDA algorithm)                   │   │
│  │  - Efficient voxel grid traversal                   │   │
│  │  - Distance accumulation through occupied voxels    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Python Bindings (`python/bindings.cpp`)

**Purpose**: Provide Pythonic interface using NumPy arrays

**Key Classes**:
- `PyVoxelRayTracer`: Wraps C++ `VoxelRayTracer` with NumPy array conversions

**Features**:
- Automatic array shape validation
- Zero-copy data transfer where possible
- Maintains NumPy memory layout (Z, Y, X)

### 2. C++ Host Code

#### VoxelRayTracer (`src/voxel_tracer.cpp`)

**Purpose**: Main interface for ray tracing operations

**Responsibilities**:
- Upload voxel grids to GPU memory
- Build OptiX acceleration structures
- Manage ray launch parameters
- Coordinate ray tracing execution

**Key Methods**:
- `traceRays()`: Main entry point for ray tracing
- `uploadVoxelData()`: Transfer voxel grid to GPU
- `buildAccelerationStructure()`: Create bounding box GAS

#### OptixSetup (`src/optix_setup.cpp`)

**Purpose**: Manage OptiX context and pipeline

**Responsibilities**:
- Initialize OptiX device context
- Load PTX modules
- Create program groups
- Link pipeline
- Build shader binding table (SBT)

**Pipeline Configuration**:
- Ray generation program: `__raygen__rg`
- Miss program: `__miss__ms`
- Closest hit program: `__closesthit__ch`

### 3. CUDA Device Code (`cuda/voxel_programs.cu`)

#### Ray Generation Program

```cuda
__global__ void __raygen__rg()
```

**Purpose**: Entry point for each ray

**Algorithm**:
1. Get ray index from launch dimensions
2. Load ray origin and direction
3. Call DDA traversal
4. Write result to output buffer

#### DDA Traversal Algorithm

```cuda
__device__ float trace_voxel_grid(...)
```

**Purpose**: Efficiently traverse voxel grid and accumulate distance

**Algorithm** (Digital Differential Analyzer):
1. **Ray-Box Intersection**: Find entry/exit points with grid bounds
2. **Initialize DDA**:
   - Current voxel position
   - Step direction (+1, 0, -1 per axis)
   - tMax: t-value to next boundary per axis
   - tDelta: t-increment per voxel per axis
3. **Traverse Loop**:
   - Check current voxel occupancy
   - If occupied: accumulate segment length
   - Step to next voxel along shortest tMax
   - Update tMax
4. **Return** total accumulated distance

**Key Properties**:
- O(n) complexity where n = number of voxels traversed
- No ray-triangle intersections needed
- Handles arbitrary ray directions
- Early exit when leaving grid

## Memory Layout

### Voxel Grid Storage

**Host (CPU)**:
```
std::vector<uint8_t> (row-major, X-fastest)
Index: x + res_x * (y + res_y * z)
```

**Device (GPU)**:
```
uint8_t* (linear buffer, same layout)
Accessed via: voxel_index(x, y, z, resolution)
```

**Python NumPy**:
```
ndarray shape: (Z, Y, X)
Memory order: C-contiguous (row-major)
```

### Launch Parameters

```cuda
struct LaunchParams {
    float3* ray_origins;        // Device pointer
    float3* ray_directions;     // Device pointer
    float* output_distances;    // Device pointer
    VoxelGridParams grid;       // Inline struct
    int num_rays;
}
```

Stored in OptiX launch parameters (constant memory during kernel execution).

## Ray Tracing Pipeline

### OptiX Pipeline Stages

1. **Ray Generation**: Generate rays, invoke traversal
2. **Traversal**: (Handled by OptiX, we use AABB for grid bounds)
3. **Intersection**: (Not used - we do DDA instead)
4. **Any-Hit**: (Not used)
5. **Closest-Hit**: (Not used - everything in raygen)
6. **Miss**: Return 0 distance

**Note**: We use OptiX primarily for its ray scheduling and GPU dispatch. The actual voxel traversal is custom DDA code.

### Acceleration Structure

We create a single **AABB** (Axis-Aligned Bounding Box) encompassing the entire voxel grid:

```cpp
OptixAabb aabb = {
    .minX = -half_width,
    .minY = -half_height,
    .minZ = -half_depth,
    .maxX = half_width,
    .maxY = half_height,
    .maxZ = half_depth
};
```

This serves as a coarse culling structure. Rays that miss the AABB are immediately rejected.

## Data Flow

### Trace Rays Operation

```
Python
  ↓ numpy arrays
C++ Host (VoxelRayTracer::traceRays)
  ↓ cudaMemcpy to device
GPU Device Memory
  ↓ optixLaunch
CUDA Kernels (__raygen__rg)
  ↓ DDA traversal
  ↓ accumulate distances
GPU Device Memory (output_distances)
  ↓ cudaMemcpy to host
C++ Host
  ↓ std::vector
Python (numpy array)
```

### Memory Transfers

**Upload** (once per voxel grid):
- Voxel data: CPU → GPU (cudaMemcpy)

**Per trace**:
- Ray origins: CPU → GPU
- Ray directions: CPU → GPU
- Distances: GPU → CPU

## Performance Characteristics

### Time Complexity

- **DDA Traversal**: O(n) per ray, where n = voxels traversed
- **Launch Overhead**: O(1) amortized across many rays
- **Memory Transfer**: O(r) where r = number of rays

### Space Complexity

- **Voxel Grid**: O(w × h × d) bytes
- **Ray Data**: O(r × 12) bytes (3 floats × 2 per ray)
- **Output**: O(r × 4) bytes
- **Acceleration Structure**: O(1) (single AABB)

### Optimization Strategies

1. **GPU Occupancy**: Launch dimensions chosen to saturate GPU
2. **Memory Coalescing**: Rays organized for coalesced memory access
3. **Branch Divergence**: Minimal branching in DDA loop
4. **Shared Memory**: Not used (memory-bound workload)

## Build System

### CMake Configuration

1. Find OptiX SDK
2. Find CUDA Toolkit
3. Configure PTX compilation
4. Build C++ library
5. Build Python module with pybind11

### PTX Generation

CUDA source → nvcc → PTX → Embedded in binary

PTX files are loaded at runtime by OptiX for JIT compilation to SASS (GPU assembly).

## Extension Points

### Custom Intersection Programs

Replace DDA with:
- Sparse voxel octrees (SVO)
- Mip-mapped voxels
- Hierarchical grids

### Custom Shading

Accumulate other properties:
- Color
- Normals
- Material IDs

### Multi-Level Tracing

Use hierarchical voxel data from the main dataset pipeline to trace at multiple resolutions.

## Dependencies

- **OptiX 7.x**: Ray tracing API
- **CUDA 11.x**: GPU computing
- **pybind11**: Python bindings
- **CMake**: Build system
- **NumPy**: Python arrays (runtime)

## References

- OptiX Programming Guide: https://raytracing-docs.nvidia.com/optix7/guide/
- DDA Algorithm: Amanatides & Woo (1987) "A Fast Voxel Traversal Algorithm"
- PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
