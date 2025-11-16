# Examples

This directory contains example scripts demonstrating how to use the hierarchical voxel ray dataset for neural rendering.

## Quick Start

### 1. Quick Timing Test

Fast performance check to verify dataset is working correctly:

```bash
uv run python quick_timing_test.py
```

**Output includes:**
- Dataset initialization time
- Single sample load time
- Cache performance
- DataLoader throughput
- Custom batch sampler performance
- Transform overhead

**When to use:** Quick sanity check after setting up the dataset.

---

### 2. Basic Usage Examples

Learn the dataset API with simple examples:

```bash
uv run python dataset_usage_example.py
```

**Demonstrates:**
- Basic dataset loading
- Data augmentation transforms
- Level filtering
- Split comparison
- Memory-efficient configurations

**When to use:** Learning how to use the dataset API.

---

### 3. Comprehensive Benchmarking

Full performance analysis suite:

```bash
# Run all benchmarks
uv run python benchmark_dataset.py

# Run specific benchmarks
uv run python benchmark_dataset.py --benchmarks single loader cache

# Custom configuration
uv run python benchmark_dataset.py \
    --batch-size 16 \
    --num-workers 8 \
    --dataset-dir ../dataset \
    --ray-dataset-dir ../ray_dataset_hierarchical
```

**Available benchmarks:**
- `single`: Single sample load times with statistics
- `loader`: DataLoader iteration throughput over multiple epochs
- `batch`: Custom ray batch sampler performance
- `cache`: Impact of different cache sizes
- `transform`: Overhead of different augmentation pipelines
- `memory`: Memory usage profiling
- `workers`: Worker scaling analysis (1-8 workers)

**When to use:** Optimizing performance for your specific hardware and use case.

---

### 4. Training Example

Complete neural rendering training example:

```bash
uv run python train_neural_rendering.py \
    --dataset-dir ../dataset \
    --ray-dataset-dir ../ray_dataset_hierarchical \
    --batch-size 8 \
    --rays-per-batch 4096 \
    --epochs 10 \
    --lr 1e-4
```

**Features:**
- Simple voxel encoder (3D CNN)
- Ray decoder (MLP) for distance prediction
- Data augmentation pipeline
- Training and validation loops
- Model checkpointing

**When to use:** Starting point for building your own neural rendering model.

---

## Script Descriptions

### quick_timing_test.py

Lightweight script for quick performance checks. No arguments needed, just run it.

**Typical runtime:** 10-30 seconds

**Sample output:**
```
Quick Dataset Timing Test
============================================================

1. Initializing dataset...
   Time: 0.1234s
   Dataset size: 311 samples
   ...

Summary:
============================================================
  Dataset initialization:     0.1234s
  Single sample load:         12.34ms
  Cached sample load:         0.56ms
  DataLoader throughput:      123.4 samples/sec
  Ray batch throughput:       45,678 rays/sec
  Transform overhead:         15.2%
============================================================
```

---

### benchmark_dataset.py

Comprehensive benchmarking suite with multiple test scenarios.

**Arguments:**
- `--dataset-dir`: Path to voxel dataset (default: `dataset`)
- `--ray-dataset-dir`: Path to ray dataset (default: `ray_dataset_hierarchical`)
- `--split`: Dataset split to use (default: `train`)
- `--batch-size`: Batch size for tests (default: 8)
- `--num-workers`: Number of DataLoader workers (default: 4)
- `--benchmarks`: Which benchmarks to run (default: `all`)

**Typical runtime:** 2-10 minutes (depends on benchmarks selected)

**Use cases:**
- Optimizing cache size for your dataset
- Finding optimal number of workers
- Measuring transform overhead
- Memory profiling
- Comparing different configurations

---

### dataset_usage_example.py

Interactive examples showing different dataset features.

**No arguments needed.**

**Typical runtime:** 5-15 seconds

**Demonstrates:**
1. Basic usage
2. Transform usage
3. Level filtering
4. Split comparison
5. Memory-efficient configurations

---

### train_neural_rendering.py

Complete training pipeline example.

**Arguments:**
- `--dataset-dir`: Path to voxel dataset
- `--ray-dataset-dir`: Path to ray dataset
- `--batch-size`: Subvolumes per batch (default: 8)
- `--rays-per-batch`: Total rays per batch (default: 4096)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use (default: auto-detect)
- `--num-workers`: DataLoader workers (default: 4)

**Saves:** `best_model.pth` with best validation checkpoint

**Note:** This is a minimal example model. Real neural rendering models would be more sophisticated.

---

## Recommended Workflow

1. **First time setup:**
   ```bash
   # Verify dataset is accessible
   uv run python quick_timing_test.py
   ```

2. **Explore the API:**
   ```bash
   # Learn different usage patterns
   uv run python dataset_usage_example.py
   ```

3. **Optimize performance:**
   ```bash
   # Find best configuration for your hardware
   uv run python benchmark_dataset.py
   ```

4. **Start training:**
   ```bash
   # Use optimized settings from benchmarks
   uv run python train_neural_rendering.py \
       --num-workers 8 \
       --batch-size 16
   ```

---

## Troubleshooting

### "No samples found for split"
- Ensure ray dataset directory exists
- Check that `splits.json` exists in dataset directory
- Verify ray files exist with: `ls -la ray_dataset_hierarchical/level_*/`

### Out of Memory
- Reduce `--batch-size`
- Reduce `--num-workers`
- Use level filtering: modify code to only load `levels=[5]`
- Reduce cache size in dataset initialization

### Slow Performance
- Increase `--num-workers` (up to number of CPU cores)
- Increase cache size in dataset
- Use SSD for dataset storage
- Enable pin_memory in DataLoader

### Low GPU Utilization
- Increase `--batch-size` or `--rays-per-batch`
- Increase `--num-workers` to avoid CPU bottleneck
- Check if data loading is the bottleneck with `benchmark_dataset.py`

---

## Expected Performance

Typical performance on modern hardware (16 cores, SSD, 32GB RAM):

- **Dataset initialization:** 0.1-0.5s
- **Single sample load (uncached):** 10-50ms
- **Single sample load (cached):** 0.5-2ms
- **DataLoader throughput:** 100-500 samples/sec (depends on workers)
- **Ray throughput:** 50k-200k rays/sec

Your performance may vary based on:
- Storage speed (SSD vs HDD)
- Number of CPU cores
- Available RAM
- Dataset size and complexity
- Ray density per subvolume

Use `benchmark_dataset.py` to measure your specific setup.

---

## Next Steps

After running these examples, you can:

1. Modify `train_neural_rendering.py` to use your own model architecture
2. Experiment with different augmentation strategies in transforms
3. Implement custom collate functions for your specific use case
4. Add custom metrics and visualizations
5. Scale up to multi-GPU training with PyTorch DDP

See [NEURAL_RENDERING_DATASET.md](../NEURAL_RENDERING_DATASET.md) for full API documentation.
