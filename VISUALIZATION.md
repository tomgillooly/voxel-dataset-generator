# Visualization Guide

This project includes a Jupyter notebook for visualizing the generated voxel datasets.

## Installation

The visualization dependencies are **optional** and kept separate from the core package to keep it lightweight.

### Install visualization dependencies:

```bash
# Install with visualization support
uv sync --extra viz

# Or if you're using pip
pip install -e ".[viz]"
```

This will install:
- `matplotlib` - 2D plotting
- `plotly` - Interactive 3D visualizations
- `ipywidgets` - Interactive notebook widgets
- `jupyter` - Jupyter notebook environment

## Usage

### Start Jupyter

```bash
# Using uv
uv run jupyter notebook

# Or using jupyter directly
jupyter notebook
```

Then open `visualize_dataset.ipynb` in your browser.

### What's Included

The notebook provides:

1. **Dataset Overview** - Load and inspect dataset metadata

2. **Interactive Hierarchical Explorer** 🎮 (All 3D!)
   - **Object Selector** - Dropdown to choose which object to visualize
   - **Level Slider** - Navigate through hierarchy levels (0-5)
   - **View Modes**:
     - *Octree 3D* (Default) - All 8 octants as interactive 3D plots in a 2x4 grid
     - *Single 3D* - Single rotating 3D scatter plot with Plotly
   - **Visualization Styles** 🆕:
     - *Auto* (Default) - Automatically chooses best style for grid size
     - *Cubes* - Solid cubes (great for small grids ≤32³)
     - *Isosurface* - Smooth surface (better for large grids)
   - **Real-time Updates** - Click "Update View" to refresh visualization
   - Shows occupancy statistics and sub-volume counts per level
   - Each octant is a fully interactive 3D plot - rotate, zoom, pan!

3. **Deduplication Analysis**
   - Charts showing unique vs total sub-volumes per level
   - Deduplication ratio visualization
   - Identify most commonly reused sub-volumes

## Without Installing Visualization Dependencies

If you want to analyze the data without the visualization packages:

```bash
# Just sync the base dependencies
uv sync

# Use the command-line analysis tools
uv run python examples/analyze_dataset.py dataset/
```

## Alternative: Separate Virtual Environment

If you prefer complete separation, create a separate environment for visualization:

```bash
# Create a new environment
python -m venv viz_env
source viz_env/bin/activate  # or `viz_env\Scripts\activate` on Windows

# Install package with viz extras
pip install -e ".[viz]"

# Run Jupyter
jupyter notebook
```

## Using the Interactive Explorer

### Navigation Workflow

1. **Select an Object** - Use the dropdown to choose which object to visualize
2. **Choose a Level** - Slide from 0 (top-level 128³) to 5 (smallest 4³)
3. **Pick a View Mode**:
   - **Octree 3D** (Default) - All 8 octants as interactive 3D plots in a grid
   - **Single 3D** - Single large 3D plot
4. **Click "Update View"** - Refresh the visualization

### Exploring the Hierarchy

**Level 0** (Top Level):
- Shows the full 128³ voxel grid as a single 3D plot
- Best for understanding overall object shape
- Rotate, zoom, and pan to explore

**Levels 1-5** (Sub-volumes):
- **Octree 3D** (Default): Shows all 8 octants in a 2x4 grid
  - Each octant is a fully interactive 3D plot
  - Rotate/zoom/pan each octant independently
  - Empty octants show "EMPTY" text
  - Each displays position coordinates and occupancy status
- **Single 3D**: Shows first non-empty sub-volume in detail

### Visualization Styles

The notebook now supports multiple rendering styles for different scenarios:

#### Auto Mode (Default)
Automatically selects the best visualization style based on grid size:
- **Grids ≤32³ with <5000 voxels**: Uses cubes
- **Larger grids**: Uses isosurface
- Optimizes both quality and performance

#### Cubes Mode
Renders each voxel as an actual 3D cube:
- **Best for**: Small grids (4³, 8³, 16³, 32³)
- **Pros**: Shows exact voxel positions, blocky/minecraft-like
- **Cons**: Can be slow with >10,000 voxels
- **Perfect for**: Examining sub-volumes at higher levels

#### Isosurface Mode
Creates a smooth surface through the voxel data:
- **Best for**: Large grids (64³, 128³)
- **Pros**: Very fast, smooth appearance, handles any size
- **Cons**: Doesn't show individual voxels as clearly
- **Perfect for**: Top-level objects and quick overviews

### Key Visualizations

#### Octree 3D View (🌟 Recommended)
The default view shows all 8 octants from a subdivision in a 2×4 grid:
- **Each cell is interactive** - Rotate, zoom, pan independently
- **Position coordinates** - Shows (x, y, z) location in parent
- **Color-coded** - Different color palette per octant
- **Empty indication** - Empty octants clearly marked
- **Style aware** - Automatically uses cubes for smaller sub-volumes!

#### Single 3D Plot
- Large, detailed 3D visualization
- Choose between cubes, isosurface, or auto
- Good for detailed examination of individual volumes

### Deduplication Statistics
Understand how effectively sub-volumes are being reused:
- Bar charts comparing unique vs total sub-volumes
- Deduplication ratio per level
- Identify most common patterns across the dataset

## Customization

The notebook is fully customizable. Edit cells to:
- Change color schemes
- Adjust downsample factors for performance
- Export figures for papers/presentations
- Add your own analysis routines

## Performance Tips

For large datasets:
- Use the `downsample` parameter in 3D visualizations
- Load only a subset of objects for initial exploration
- Export static figures instead of interactive plots
