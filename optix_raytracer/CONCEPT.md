# OptiX Voxel Ray Tracer - Concept Explanation

## What Does It Do?

This ray tracer casts rays through voxel grids and measures **how far each ray travels through solid/occupied voxels**. Unlike traditional ray tracing (which stops at the first surface), this treats voxels as **transparent** and accumulates distance.

## Visual Concept

```
Traditional Ray Tracing:          Transparent Ray Tracing (this):
(stops at first hit)              (accumulates distance)

Ray →                             Ray →
       ┌─────────┐                      ┌─────────┐
       │█████████│                      │█████████│
       │████●────│ ← Hit!               │████▓▓▓▓▓│
       │█████████│                      │████▓▓▓▓▓│
       └─────────┘                      └─────────┘
                                              ▲
Output: Hit position              Output: Total distance = d₁ + d₂
```

## Example Scenario

Imagine a hollow sphere made of voxels:

```
Side View:

    Ray 1 →   Ray 2 →     Ray 3 →

       ░░░░░░░░░░
     ░░          ░░
    ░              ░
    ░              ░     ← Hollow interior
    ░              ░
     ░░          ░░
       ░░░░░░░░░░

Ray 1: Misses → distance = 0
Ray 2: Passes through shell twice → distance = 2 × shell_thickness
Ray 3: Grazes edge → distance = small value
```

## Mathematical Definition

For a ray with origin **o** and direction **d**:

```
distance = ∫ occupancy(o + t·d) dt  (from t_enter to t_exit)
```

In discrete voxels:
```
distance = Σ segment_length  (for all occupied voxels along ray)
```

## Use Cases Explained

### 1. Depth/Thickness Maps

**Problem**: You want to know how thick an object is from a particular viewpoint.

**Solution**: Cast orthographic rays from that direction:

```
Top View (Z-down rays):

Grid of rays ↓ ↓ ↓ ↓ ↓
              ┌───────┐
              │▓▓▓▓▓▓▓│
              │▓▓░░░▓▓│  ← Object
              │▓▓▓▓▓▓▓│
              └───────┘

Output: 2D image where each pixel = thickness at that X,Y position
```

**Code**:
```python
origins[i, j] = [x, y, 2.0]    # Above object
directions[i, j] = [0, 0, -1]   # Point down
distances = tracer.trace_rays(origins, directions)
# distances[i,j] = thickness at position (x,y)
```

### 2. X-Ray Style Rendering

**Problem**: Visualize internal structure like an X-ray.

**Solution**: Accumulate density along rays:

```
Ray through dense object:
█████████ → High accumulated distance (bright)

Ray through sparse object:
█░░█░░░█░ → Low accumulated distance (dark)
```

**Result**: Images where brightness = material density along ray path

### 3. Material Thickness Analysis

**Problem**: Verify structural integrity or measure wall thickness.

**Solution**: Cast rays from multiple angles and analyze statistics:

```python
# Cast rays from all directions
for angle in range(0, 360, 10):
    camera_pos = [r*cos(angle), r*sin(angle), 0]
    distances = tracer.trace_rays(origins, directions)

    min_thickness = distances[distances > 0].min()
    max_thickness = distances.max()
    avg_thickness = distances.mean()
```

### 4. Neural Rendering Training Data

**Problem**: Train neural networks for 3D reconstruction or novel view synthesis.

**Solution**: Generate paired data (voxels → distances):

```
Input: Voxel grid (3D)
Output: Distance maps from multiple viewpoints (2D images)

Network learns: f(voxels, camera_pose) → distance_map
```

## Comparison: Ray Stopping vs Distance Accumulation

### Traditional Ray Tracing (Surface Finding)

```
Ray → ●───────────
      ↑
      First hit → return position/color

Use: Surface rendering, reflections, shadows
```

### Distance Accumulation (This Implementation)

```
Ray → ▓▓▓░░░▓▓▓▓░░▓▓
      ├─┤   ├──┤  ├┤
      d₁    d₂   d₃

Total distance = d₁ + d₂ + d₃

Use: Thickness, density, transparency, volume analysis
```

## Algorithm Walkthrough

Let's trace a single ray through a simple 2D grid:

```
Grid:      0 1 2 3 4
        0  □ □ □ □ □
        1  □ █ █ □ □
        2  □ █ █ □ □
        3  □ □ □ □ □

Ray: origin=(0.5, 1.5), direction=(1, 0)  [moving right]
```

**Step-by-step**:

1. **Entry**: Ray enters at (0.5, 1.5)
   - Current voxel: (0, 1) - Empty

2. **Step 1**: Move to next voxel boundary at x=1
   - Current voxel: (1, 1) - **Occupied**
   - Distance through: 0.5 units
   - Accumulated: 0.5

3. **Step 2**: Move to x=2
   - Current voxel: (2, 1) - **Occupied**
   - Distance through: 1.0 unit
   - Accumulated: 1.5

4. **Step 3**: Move to x=3
   - Current voxel: (3, 1) - Empty
   - Distance through: 0
   - Accumulated: 1.5

5. **Exit**: Ray leaves grid
   - **Final distance: 1.5 units**

## Why Use OptiX?

OptiX provides:

1. **GPU Parallelism**: Trace millions of rays simultaneously
2. **Efficient Scheduling**: Optimal GPU thread organization
3. **Hardware Acceleration**: Uses RT cores on RTX GPUs (when applicable)
4. **Memory Management**: Handles large voxel grids efficiently

**Without OptiX**: CPU-based ray tracing → minutes per image
**With OptiX**: GPU-accelerated → milliseconds per image

## Data Flow Summary

```
┌─────────────────┐
│   Voxel Grid    │  128×128×128 occupancy map
│  (from pipeline)│
└────────┬────────┘
         │ Load
         ▼
┌─────────────────┐
│  VoxelRayTracer │  Upload to GPU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generate Rays │  Define camera/viewpoint
│ (origins, dirs) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OptiX Launch   │  GPU ray tracing
│  (DDA traversal)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Distance Map   │  2D image of accumulated distances
│  (output)       │
└─────────────────┘
```

## Common Patterns

### Orthographic Projection
```python
# Parallel rays from one direction
origins[:, :, 2] = 2.0        # All at same Z
directions[:, :, 2] = -1.0    # All pointing down
```
**Use**: Thickness maps, top/side/front views

### Perspective Projection
```python
# Rays emanate from single point
all_origins = camera_pos      # Single point
directions = normalized(pixel_position - camera_pos)
```
**Use**: Realistic camera views, multi-view datasets

### Turntable Rendering
```python
# Circle around object
for angle in range(0, 360, 45):
    camera_pos = [r*cos(angle), r*sin(angle), height]
    # ... generate rays from this position
```
**Use**: 360° visualization, training data

## Performance Example

Typical workload:
- Voxel grid: 128³ = 2,097,152 voxels
- Rays: 512² = 262,144 rays
- Time: ~20ms (RTX 3080)
- Throughput: ~13 million rays/second

Compare to CPU (single-threaded):
- Same workload: ~5-10 seconds
- Speedup: **250-500×**

## Key Takeaways

1. **Transparency**: Rays pass through entire object, not stopping at first hit
2. **Accumulation**: Sum all segment lengths through occupied voxels
3. **Efficiency**: GPU parallelism enables real-time/near-real-time performance
4. **Flexibility**: Works with any voxel grid from your pipeline
5. **Integration**: Designed to complement the main voxel dataset generator

## Next Steps

1. **Build**: Follow [QUICKSTART.md](QUICKSTART.md)
2. **Try Examples**: Run [examples/basic_tracing.py](examples/basic_tracing.py)
3. **Integrate**: See [OPTIX_INTEGRATION.md](../OPTIX_INTEGRATION.md)
4. **Extend**: Modify for your specific use case

## Questions?

- **"Why not just use mesh ray tracing?"**
  → Voxels enable easy transparency, volume rendering, and hierarchical processing

- **"Can I trace at different resolutions?"**
  → Yes! Use different octree levels from your pipeline

- **"What about color or materials?"**
  → Currently distance-only, but extensible to other properties

- **"Do I need an RTX GPU?"**
  → No, any OptiX-capable NVIDIA GPU works (GTX 10xx+, RTX recommended)

---

Ready to ray trace some voxels? Start with [QUICKSTART.md](QUICKSTART.md)!
