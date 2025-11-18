#!/usr/bin/env python3
"""Example demonstrating sparse voxel representations for PyTorch Geometric.

This script shows how to use the dataset with sparse voxel representations,
which is memory-efficient for sparse 3D data and compatible with PyTorch
Geometric sparse convolution operations.
"""

from pathlib import Path
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    compute_sparse_statistics,
    voxels_to_sparse_coo,
    voxels_to_sparse_index,
    sparse_coo_to_dense,
)
import torch


def basic_sparse_coo_example():
    """Basic example using sparse COO format."""
    print("="*70)
    print("Basic Sparse COO Format Example")
    print("="*70)

    # Create dataset with sparse COO format
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[0],
        sparse_voxels=True,
        sparse_mode='coo',  # Coordinate format
        cache_size=10,
    )

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Level distribution: {dataset.get_level_distribution()}")

    # Get a sample
    sample = dataset[0]

    print(f"\nSample 0 (Sparse COO format):")
    print(f"  Voxel positions shape: {sample['voxel_pos'].shape}")
    print(f"  Voxel features shape: {sample['voxel_features'].shape}")
    print(f"  Original grid shape: {sample['voxel_shape']}")
    print(f"  Number of occupied voxels: {sample['num_voxels']}")
    print(f"  Ray origins shape: {sample['origins'].shape}")
    print(f"  Ray directions shape: {sample['directions'].shape}")

    # Compute sparsity statistics
    sparse_data = {
        'pos': sample['voxel_pos'],
        'features': sample['voxel_features'],
        'shape': sample['voxel_shape'],
        'num_nodes': sample['num_voxels'],
    }
    stats = compute_sparse_statistics(sparse_data)

    print(f"\nSparsity statistics:")
    print(f"  Occupancy: {stats['occupancy']:.2%}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Memory saved: {(1 - 1/stats['compression_ratio']) * 100:.1f}%")


def sparse_graph_example():
    """Example using sparse graph format with edge indices."""
    print("\n" + "="*70)
    print("Sparse Graph Format Example (with Edge Indices)")
    print("="*70)

    # Create dataset with graph format
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[0],
        sparse_voxels=True,
        sparse_mode='graph',  # Graph format with edges
        sparse_connectivity=6,  # 6-connected neighborhood (face neighbors)
        cache_size=10,
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]

    print(f"\nSample 0 (Sparse Graph format):")
    print(f"  Voxel positions shape: {sample['voxel_pos'].shape}")
    print(f"  Voxel features shape: {sample['voxel_features'].shape}")
    print(f"  Edge index shape: {sample['voxel_edge_index'].shape}")
    print(f"  Number of voxel nodes: {sample['voxel_pos'].shape[0]}")
    print(f"  Number of edges: {sample['voxel_edge_index'].shape[1]}")

    if sample['voxel_pos'].shape[0] > 0:
        avg_degree = sample['voxel_edge_index'].shape[1] / sample['voxel_pos'].shape[0]
        print(f"  Average degree: {avg_degree:.1f}")

    print(f"\nThis format is ready for PyTorch Geometric GNN layers!")


def pytorch_geometric_usage_example():
    """Example showing how to use with PyTorch Geometric operations."""
    print("\n" + "="*70)
    print("PyTorch Geometric Usage Example")
    print("="*70)

    # Create sparse dataset
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        sparse_voxels=True,
        sparse_mode='graph',
        sparse_connectivity=6,
        levels=[4, 5],  # Use smaller grids for demo
        cache_size=5,
    )

    sample = dataset[0]

    print(f"\nSample voxel data:")
    print(f"  Positions (pos): {sample['voxel_pos'].shape}")
    print(f"  Features (x): {sample['voxel_features'].shape}")
    print(f"  Edge indices: {sample['voxel_edge_index'].shape}")

    print(f"\nTo use with PyTorch Geometric sparse convolutions:")
    print(f"  1. Install pytorch_geometric: pip install torch-geometric")
    print(f"  2. Use the data directly with GNN layers:")
    print(f"\n  Example code:")
    print(f"  ```python")
    print(f"  from torch_geometric.nn import GCNConv, MessagePassing")
    print(f"  ")
    print(f"  # Graph Convolutional Network")
    print(f"  conv1 = GCNConv(1, 16)")
    print(f"  conv2 = GCNConv(16, 32)")
    print(f"  ")
    print(f"  # Forward pass")
    print(f"  x = sample['voxel_features']")
    print(f"  edge_index = sample['voxel_edge_index']")
    print(f"  ")
    print(f"  x = conv1(x, edge_index)")
    print(f"  x = torch.relu(x)")
    print(f"  x = conv2(x, edge_index)")
    print(f"  ```")

    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        print(f"\n  PyTorch Geometric is installed! You can use the layers directly.")

        # Try a simple convolution
        from torch_geometric.nn import GCNConv

        if sample['voxel_pos'].shape[0] > 0:
            conv = GCNConv(1, 8)
            x = sample['voxel_features']
            edge_index = sample['voxel_edge_index']

            output = conv(x, edge_index)
            print(f"\n  Test convolution successful!")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
    except ImportError:
        print(f"\n  PyTorch Geometric not installed.")
        print(f"  Install it to use GNN layers: pip install torch-geometric")


def manual_conversion_example():
    """Example showing manual conversion from dense to sparse."""
    print("\n" + "="*70)
    print("Manual Conversion Example")
    print("="*70)

    # Load dataset in dense format
    dense_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        sparse_voxels=False,  # Dense format
        levels=[5],  # Small grids
        cache_size=5,
    )

    sample = dense_dataset[0]
    dense_voxels = sample['voxels']

    print(f"\nDense voxels shape: {dense_voxels.shape}")

    # Convert to sparse COO
    sparse_coo = voxels_to_sparse_coo(dense_voxels)
    print(f"\nSparse COO format:")
    print(f"  Positions: {sparse_coo['pos'].shape}")
    print(f"  Features: {sparse_coo['features'].shape}")

    # Convert to sparse graph
    sparse_graph = voxels_to_sparse_index(dense_voxels, connectivity=6)
    print(f"\nSparse Graph format:")
    print(f"  Positions: {sparse_graph['pos'].shape}")
    print(f"  Edge index: {sparse_graph['edge_index'].shape}")

    # Convert back to dense
    reconstructed = sparse_coo_to_dense(sparse_coo)
    print(f"\nReconstructed dense shape: {reconstructed.shape}")

    # Verify reconstruction
    diff = (dense_voxels - reconstructed).abs().sum()
    print(f"Reconstruction error: {diff.item()}")
    if diff.item() < 1e-6:
        print("Perfect reconstruction!")


def memory_comparison_example():
    """Compare memory usage between dense and sparse formats."""
    print("\n" + "="*70)
    print("Memory Comparison Example")
    print("="*70)

    # Dense format
    dense_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        sparse_voxels=False,
        levels=[3, 4, 5],
        cache_size=5,
    )

    # Sparse format
    sparse_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        sparse_voxels=True,
        sparse_mode='coo',
        levels=[3, 4, 5],
        cache_size=5,
    )

    # Compare first sample
    dense_sample = dense_dataset[0]
    sparse_sample = sparse_dataset[0]

    dense_bytes = dense_sample['voxels'].nbytes
    sparse_bytes = (sparse_sample['voxel_pos'].nbytes +
                   sparse_sample['voxel_features'].nbytes)

    print(f"\nSample 0 memory usage:")
    print(f"  Dense format: {dense_bytes / 1024:.2f} KB")
    print(f"  Sparse format: {sparse_bytes / 1024:.2f} KB")
    print(f"  Savings: {(1 - sparse_bytes/dense_bytes) * 100:.1f}%")
    print(f"  Compression: {dense_bytes/sparse_bytes:.1f}x")

    # Compute statistics
    sparse_data = {
        'pos': sparse_sample['voxel_pos'],
        'features': sparse_sample['voxel_features'],
        'shape': sparse_sample['voxel_shape'],
        'num_nodes': sparse_sample['num_voxels'],
    }
    stats = compute_sparse_statistics(sparse_data)
    print(f"\n  Voxel occupancy: {stats['occupancy']:.2%}")
    print(f"  Theoretical compression: {stats['compression_ratio']:.1f}x")


def main():
    """Run all examples."""
    try:
        basic_sparse_coo_example()
    except Exception as e:
        print(f"Error in basic sparse COO example: {e}")

    try:
        sparse_graph_example()
    except Exception as e:
        print(f"Error in sparse graph example: {e}")

    try:
        pytorch_geometric_usage_example()
    except Exception as e:
        print(f"Error in PyTorch Geometric usage example: {e}")

    try:
        manual_conversion_example()
    except Exception as e:
        print(f"Error in manual conversion example: {e}")

    try:
        memory_comparison_example()
    except Exception as e:
        print(f"Error in memory comparison example: {e}")

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Use sparse_voxels=True for memory-efficient representation")
    print("  2. Choose 'coo' mode for coordinates only")
    print("  3. Choose 'graph' mode for GNN operations with edge indices")
    print("  4. Sparse format can save 90%+ memory for typical voxel data")
    print("  5. Compatible with PyTorch Geometric sparse convolutions")


if __name__ == "__main__":
    main()
