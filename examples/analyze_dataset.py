"""Example of analyzing the generated dataset using Polars."""

from pathlib import Path
import polars as pl
from voxel_dataset_generator.utils.metadata import MetadataAnalyzer


def analyze_with_polars(dataset_dir: Path):
    """Analyze dataset using Polars for efficient data processing."""

    print("Loading subdivision data with Polars...")

    # Method 1: Load all JSON files into a dataframe
    objects_dir = dataset_dir / "objects"
    subdivision_files = list(objects_dir.glob("*/subdivision_map.json"))

    if not subdivision_files:
        print(f"No subdivision maps found in {objects_dir}")
        return

    # Load all files
    dfs = []
    for file in subdivision_files:
        df = pl.read_json(file)
        dfs.append(df)

    # Concatenate into single dataframe
    df = pl.concat(dfs)

    print(f"\nLoaded {len(df)} subdivision records")
    print(f"Dataset schema:\n{df.schema}")

    # Analysis 1: Sub-volume frequency by level
    print("\n" + "=" * 60)
    print("Analysis 1: Hash frequency by level")
    print("=" * 60)

    for level in sorted(df["level"].unique()):
        level_df = df.filter(pl.col("level") == level)
        hash_freq = level_df.group_by("hash").agg(
            pl.count().alias("frequency")
        ).sort("frequency", descending=True)

        print(f"\nLevel {level}:")
        print(f"  Total sub-volumes: {len(level_df)}")
        print(f"  Unique hashes: {len(hash_freq)}")
        print(f"  Deduplication ratio: {len(hash_freq) / len(level_df):.2%}")
        print(f"  Top 5 most common:")
        for row in hash_freq.head(5).iter_rows(named=True):
            print(f"    Hash {row['hash'][:16]}...: {row['frequency']} occurrences")

    # Analysis 2: Empty vs occupied sub-volumes
    print("\n" + "=" * 60)
    print("Analysis 2: Empty vs occupied sub-volumes")
    print("=" * 60)

    empty_stats = df.group_by("level").agg([
        pl.col("is_empty").sum().alias("empty_count"),
        pl.count().alias("total_count")
    ]).sort("level")

    for row in empty_stats.iter_rows(named=True):
        empty_pct = row["empty_count"] / row["total_count"] * 100
        print(f"  Level {row['level']}: "
              f"{row['empty_count']:,} empty / {row['total_count']:,} total "
              f"({empty_pct:.1f}% empty)")

    # Analysis 3: Object complexity
    print("\n" + "=" * 60)
    print("Analysis 3: Object complexity (unique sub-volumes per object)")
    print("=" * 60)

    object_complexity = df.group_by("object_id").agg(
        pl.col("hash").n_unique().alias("unique_subvolumes")
    ).sort("unique_subvolumes", descending=True)

    print(f"  Most complex objects:")
    for row in object_complexity.head(10).iter_rows(named=True):
        print(f"    Object {row['object_id']}: {row['unique_subvolumes']} unique sub-volumes")

    print(f"\n  Least complex objects:")
    for row in object_complexity.tail(10).iter_rows(named=True):
        print(f"    Object {row['object_id']}: {row['unique_subvolumes']} unique sub-volumes")

    # Analysis 4: Position distribution (where do sub-volumes occur)
    print("\n" + "=" * 60)
    print("Analysis 4: Octant distribution")
    print("=" * 60)

    octant_dist = df.group_by(["level", "octant_index"]).agg(
        pl.count().alias("count")
    ).sort(["level", "octant_index"])

    for level in sorted(df["level"].unique()):
        level_octants = octant_dist.filter(pl.col("level") == level)
        print(f"\n  Level {level} octant distribution:")
        for row in level_octants.iter_rows(named=True):
            print(f"    Octant {row['octant_index']}: {row['count']:,} sub-volumes")

    # Analysis 5: Save processed data for further analysis
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)

    output_file = dataset_dir / "analysis_data.parquet"
    df.write_parquet(output_file)
    print(f"  Saved to: {output_file}")

    # Also save as CSV for compatibility
    csv_file = dataset_dir / "analysis_data.csv"
    df.write_csv(csv_file)
    print(f"  Also saved CSV: {csv_file}")


def analyze_with_metadata_analyzer(dataset_dir: Path):
    """Analyze using the built-in MetadataAnalyzer."""

    print("\nUsing MetadataAnalyzer...")
    analyzer = MetadataAnalyzer(dataset_dir)

    # Get summary statistics
    summary = analyzer.summary_statistics()

    print(f"\nSummary Statistics:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Number of objects: {summary['num_objects']}")
    print(f"  Levels: {summary['levels']}")

    print("\n  Per-level statistics:")
    for level, stats in summary['level_statistics'].items():
        print(f"    Level {level}:")
        print(f"      Total: {stats['count']:,}")
        print(f"      Unique: {stats['unique']:,}")
        print(f"      Empty: {stats['empty_count']:,}")

    # Export to NDJSON for Polars
    ndjson_path = dataset_dir / "all_subdivisions.ndjson"
    analyzer.export_to_polars_compatible(ndjson_path)
    print(f"\n  Exported to NDJSON: {ndjson_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        dataset_dir = Path("dataset")

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        print("Usage: python analyze_dataset.py <dataset_dir>")
        sys.exit(1)

    print(f"Analyzing dataset in: {dataset_dir}")

    # Run analyses
    analyze_with_metadata_analyzer(dataset_dir)
    analyze_with_polars(dataset_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
