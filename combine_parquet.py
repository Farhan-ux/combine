import pandas as pd
import os
import glob

folders = [
    "mistral 7b",
    "phi",
    "qwen 2.5 3B",
    "qwen-3-235b-a22b-instruct-2507",
    "qwen32b"
]

def combine_parquet_files(folder):
    output_filename = folder.replace(" ", "_") + ".parquet"
    parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
    parquet_files.sort()

    if not parquet_files:
        print(f"No parquet files found in {folder}")
        return

    print(f"Combining {len(parquet_files)} files from '{folder}' into '{output_filename}'...")
    dfs = [pd.read_parquet(file) for file in parquet_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_parquet(output_filename)
    print(f"Successfully created {output_filename} with {len(combined_df)} rows.\n")

if __name__ == "__main__":
    for folder in folders:
        combine_parquet_files(folder)
