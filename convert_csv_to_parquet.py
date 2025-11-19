"""Quick script to convert large CSV grid results to Parquet format.

Usage:
    python convert_csv_to_parquet.py data/AAPL_grid_wide_results.csv
    python convert_csv_to_parquet.py data/AAPL_grid_wide_results.csv --output data/AAPL_grid_wide_results.parquet
"""

import argparse
from pathlib import Path
import pandas as pd


def convert_csv_to_parquet(csv_path: Path, output_path: Path = None, compression: str = 'snappy'):
    """Convert CSV file to Parquet format.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output Parquet file (default: same name with .parquet extension)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', or None)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.with_suffix('.parquet')
    else:
        output_path = Path(output_path)
    
    print(f"Reading CSV file: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / (1024**2):.1f} MB")
    
    # Read CSV in chunks if very large (optional optimization)
    # For now, read all at once since we need to sort anyway
    print("Loading CSV into memory...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Write to Parquet
    print(f"\nWriting to Parquet: {output_path}")
    print(f"Compression: {compression if compression else 'none'}")
    
    df.to_parquet(
        output_path,
        compression=compression,
        index=False,
        engine='pyarrow'  # Use pyarrow for better performance
    )
    
    # Compare file sizes
    csv_size = csv_path.stat().st_size / (1024**2)
    parquet_size = output_path.stat().st_size / (1024**2)
    reduction = (1 - parquet_size / csv_size) * 100
    
    print(f"\nConversion complete!")
    print(f"CSV size:    {csv_size:.1f} MB")
    print(f"Parquet size: {parquet_size:.1f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"\nSaved to: {output_path}")
    print(f"\nTo read back: df = pd.read_parquet('{output_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV grid results to Parquet format")
    parser.add_argument("csv_file", type=Path, help="Path to CSV file to convert")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output Parquet file path (default: same name with .parquet extension)")
    parser.add_argument("--compression", "-c", type=str, default="snappy", 
                       choices=["snappy", "gzip", "brotli", "none"],
                       help="Compression algorithm (default: snappy)")
    
    args = parser.parse_args()
    
    # Handle 'none' compression
    compression = None if args.compression == "none" else args.compression
    
    try:
        convert_csv_to_parquet(args.csv_file, args.output, compression)
    except Exception as e:
        print(f"Error: {e}")
        raise

