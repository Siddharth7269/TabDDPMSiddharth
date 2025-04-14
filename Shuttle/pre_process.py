#!/usr/bin/env python
import pandas as pd
import argparse
import os
import json
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(input_file, output_file, metadata_file):
    """
    Loads data, performs ordinal encoding on categorical columns,
    saves the numerical data WITHOUT headers, and saves the encoding metadata.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the preprocessed numerical CSV file.
        metadata_file (str): Path to save the JSON metadata file.
    """
    # --- 1. Load Data ---
    print(f"Loading data from '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    try:
        # Read CSV, assuming header is present in the input file
        df = pd.read_csv(input_file)
        if df.empty:
            print(f"Error: Input file '{input_file}' is empty.")
            return
        # Store original column order for metadata consistency
        original_columns = df.columns.tolist()
    except Exception as e:
        print(f"Error loading CSV '{input_file}': {e}")
        return

    print(f"Loaded DataFrame shape: {df.shape}")

    # --- 2. Identify Categorical Columns ---
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols:
        print("No categorical columns found. Saving original data without header.")
        try:
            # Save the original data but without header and index
            df.to_csv(output_file, index=False, header=False)
            # Save metadata indicating no encoding happened but listing original columns
            metadata = {
                "original_columns": original_columns, # Store original column names
                "categorical_columns": [],
                "encoding_method": "none",
                "mappings": {}
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Original data saved without header to '{output_file}'.")
            print(f"Metadata saved to '{metadata_file}'.")
        except Exception as e:
            print(f"Error saving data/metadata: {e}")
        return

    print(f"Found categorical columns: {categorical_cols}")

    # --- 3. Apply Ordinal Encoding ---
    print("Applying Ordinal Encoding...")
    encoder = OrdinalEncoder()
    df_processed = df.copy() # Work on a copy

    # Fit the encoder on the categorical columns and transform them
    df_processed[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    print("Encoding complete.")

    # --- 4. Save Processed Numerical Data (No Header) ---
    try:
        # Save WITHOUT header row, also no index
        df_processed.to_csv(output_file, index=False, header=False)
        print(f"Preprocessed numerical data saved WITHOUT header to '{output_file}'.")
        print("First 5 rows of processed data (as saved):")
        # To display how it looks in the file, convert to csv string without header
        print(df_processed.head().to_csv(header=False, index=False))
    except Exception as e:
        print(f"Error saving processed data to '{output_file}': {e}")
        return

    # --- 5. Prepare and Save Metadata ---
    print(f"Saving encoding metadata to '{metadata_file}'...")
    mappings = {}
    for i, col_name in enumerate(categorical_cols):
        mappings[col_name] = encoder.categories_[i].tolist()

    metadata = {
        "original_columns": original_columns, # Store original column names/order
        "categorical_columns": categorical_cols,
        "encoding_method": "ordinal",
        "mappings": mappings
    }

    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print("Metadata saved successfully.")
    except Exception as e:
        print(f"Error saving metadata to '{metadata_file}': {e}")

# -------------------------------
# Main execution block
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CSV data: Ordinal encode categorical columns, save numerical data without header, and save metadata."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file (e.g., 'input.csv')."
    )
    parser.add_argument(
        "-o", "--output_file",
        default="pre_processed.csv",
        help="Path to save the processed numerical CSV file (default: 'pre_processed.csv')."
    )
    parser.add_argument(
        "-m", "--metadata_file",
        default="metadata.json",
        help="Path to save the JSON metadata file (default: 'metadata.json')."
    )

    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_file, args.metadata_file)
    print("\nPreprocessing script finished.")

if __name__ == "__main__":
    main()
