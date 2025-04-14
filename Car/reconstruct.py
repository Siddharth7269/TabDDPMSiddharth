#!/usr/bin/env python
import pandas as pd
import argparse
import os
import json

def reconstruct_data(input_file, metadata_file, output_file):
    """
    Reconstructs categorical data from numerical data (headerless CSV)
    and metadata.

    Args:
        input_file (str): Path to the preprocessed numerical CSV file (must have no header).
        metadata_file (str): Path to the JSON metadata file.
        output_file (str): Path to save the reconstructed categorical CSV file.
    """
    # --- 1. Load Metadata ---
    print(f"Loading metadata from '{metadata_file}'...")
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at '{metadata_file}'")
        return

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # --- Get necessary info from metadata ---
    original_columns = metadata.get("original_columns")
    categorical_columns = metadata.get("categorical_columns")
    encoding_method = metadata.get("encoding_method")
    mappings = metadata.get("mappings")

    if not original_columns:
         print("Error: 'original_columns' not found in metadata. Cannot read headerless CSV.")
         return

    # --- 2. Load Numerical Data (Headerless) ---
    print(f"Loading numerical data from headerless file '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return
    try:
        # Read the CSV specifying no header and providing column names from metadata
        df_numerical = pd.read_csv(input_file, header=None, names=original_columns)
        if df_numerical.empty:
            print(f"Error: Input file '{input_file}' is empty.")
            return
    except Exception as e:
        print(f"Error loading numerical data: {e}")
        return

    print(f"Loaded numerical data shape: {df_numerical.shape}")
    # print("Columns assigned:", df_numerical.columns.tolist()) # Optional: verify columns

    # --- 3. Reconstruct Categorical Data ---
    print("Reconstructing categorical data...")
    df_reconstructed = df_numerical.copy() # Start with a copy

    # Check if encoding was actually performed
    if encoding_method == "none" or not categorical_columns:
        print("Metadata indicates no encoding was performed. Saving data as is.")
        # Fall through to save df_reconstructed which is same as df_numerical here
    elif encoding_method == "ordinal":
        if not mappings:
            print("Error: Encoding method is 'ordinal' but mappings are missing in metadata.")
            return

        for col in categorical_columns:
            if col not in df_reconstructed.columns:
                print(f"Warning: Categorical column '{col}' (from metadata) not found in the loaded data. Skipping.")
                continue

            if col not in mappings:
                print(f"Warning: No mapping found for column '{col}' in metadata. Skipping reconstruction for this column.")
                continue

            category_list = mappings[col]

            def map_value(val):
                """Helper function to map a numerical value to its original category, handling non-integer input."""
                try:
                    # Round the value to the nearest integer and cast to int
                    index = int(round(val))
                    if 0 <= index < len(category_list):
                        return category_list[index]
                    else:
                        print(f"Warning: Numerical value {val} in column '{col}' maps to out-of-bounds index {index}. Returning 'Unknown'.")
                        return "Unknown"
                except (ValueError, TypeError): # Catch potential errors if value isn't number-like
                    print(f"Warning: Non-numerical or invalid value '{val}' found in column '{col}'. Returning 'Unknown'.")
                    return "Unknown"

            # Apply the mapping function to the column
            df_reconstructed[col] = df_reconstructed[col].apply(map_value)

        print("Reconstruction complete.")
    else:
        print(f"Error: Unsupported encoding method: {encoding_method}. This script only handles 'ordinal' or 'none'.")
        return

    # --- 4. Save Reconstructed Data ---
    try:
        # Save with header this time, as it's the reconstructed original format
        df_reconstructed.to_csv(output_file, index=False, header=True)
        print(f"Reconstructed data saved with header to '{output_file}'.")
        print("First 5 rows of reconstructed data:")
        print(df_reconstructed.head())
    except Exception as e:
        print(f"Error saving reconstructed data: {e}")

# -------------------------------
# Main execution block
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct categorical data from headerless numerical data and metadata (Ordinal Encoding)."
    )
    parser.add_argument(
        "input_file",
        help="Path to the preprocessed numerical CSV file (e.g., 'pre_processed.csv'). MUST NOT contain a header."
    )
    parser.add_argument(
        "metadata_file",
        help="Path to the JSON metadata file (e.g., 'metadata.json')."
    )
    parser.add_argument(
        "-o", "--output_file",
        default="reconstructed.csv",
        help="Path to save the reconstructed categorical CSV file (default: 'reconstructed.csv')."
    )

    args = parser.parse_args()

    reconstruct_data(args.input_file, args.metadata_file, args.output_file)
    print("\nReconstruction script finished.")

if __name__ == "__main__":
    main()
