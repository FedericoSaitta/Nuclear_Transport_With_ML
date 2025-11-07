#!/usr/bin/env python3
"""
csv_to_hdf5_h5py.py

Convert a CSV file to an HDF5 (.h5) file using h5py (no PyTables dependency).
Automatically replaces NaN or missing values with -99.
"""

import argparse
import pandas as pd
import numpy as np
import h5py
import sys

def convert_csv_to_hdf5(csv_path, h5_path, replace_nan=-99):
    print(f"🔹 Reading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    print("\n📊 Dataset Summary:")
    print(f"   • Rows: {len(df)}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"   • Column names: {list(df.columns)}")

    # Replace NaNs
    df = df.fillna(replace_nan)

    print(f"\n📦 Saving to HDF5 (using h5py): {h5_path}")
    try:
        with h5py.File(h5_path, "w") as f:
            for col in df.columns:
                data = df[col].to_numpy()
                if df[col].dtype == "object":
                    data = data.astype("S")  # encode strings as bytes

                # Save with compression
                f.create_dataset(
                    col,
                    data=data,
                    compression="gzip",   # or 'lzf' for faster I/O
                    compression_opts=9,   # 1–9 (higher = smaller but slower)
                    chunks=True           # allows efficient partial reads
                )
    except Exception as e:
        print(f"❌ Error saving HDF5 file: {e}")
        sys.exit(1)

    print(f"✅ Done! Saved {len(df.columns)} datasets to {h5_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to HDF5 without PyTables")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("h5_path", help="Path to the output HDF5 file")
    parser.add_argument("--replace_nan", type=float, default=-99, help="Value to replace NaNs (default: -99)")

    args = parser.parse_args()
    convert_csv_to_hdf5(args.csv_path, args.h5_path, args.replace_nan)

    # The command: 
    # python csv_to_hdf5.py input_file output_file
