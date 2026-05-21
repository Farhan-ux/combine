import pandas as pd
import json
import os
import glob
import numpy as np

def count_empty_responses(val):
    """
    Counts how many empty or null responses are in the input.
    Expected input is either a JSON string of a list of strings,
    or a list/numpy array of strings.
    """
    if val is None:
        return 10  # Assuming a standard of 10 responses per prompt

    # Handle list or numpy array (common in some model outputs)
    if isinstance(val, (list, np.ndarray)):
        count = 0
        for r in val:
            if r is None or str(r).strip() == "":
                count += 1
        return count

    # Handle JSON string (common in other model outputs)
    if isinstance(val, str):
        val_stripped = val.strip()
        if not val_stripped:
            return 10
        try:
            resps = json.loads(val)
            if isinstance(resps, list):
                return sum(1 for r in resps if not r or str(r).strip() == "")
            return 1 if not resps or str(resps).strip() == "" else 0
        except (json.JSONDecodeError, TypeError):
            # If it's just a raw string and not JSON
            return 1 if not val_stripped else 0

    return 0

def analyze_models():
    # Find all combined parquet files in the root
    parquet_files = glob.glob("*.parquet")
    parquet_files.sort()

    if not parquet_files:
        print("No .parquet files found in the root directory.")
        return

    summary_data = []
    model_details = {}

    for file in parquet_files:
        model_name = file.replace(".parquet", "")
        print(f"Analyzing {file}...")
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        # Verify required columns
        if 'response_text' not in df.columns or 'prompt_id' not in df.columns:
            print(f"Skipping {file}: Missing required columns.")
            continue

        # Apply the counting function to each row
        df['empty_count'] = df['response_text'].apply(count_empty_responses)

        empty_prompts_df = df[df['empty_count'] > 0]

        summary_data.append({
            "Model": model_name,
            "Total Prompts": len(df),
            "Prompts with Issues": len(empty_prompts_df),
            "Total Empty Responses": int(df['empty_count'].sum())
        })

        # Store details for prompts with issues
        if not empty_prompts_df.empty:
            model_details[model_name] = empty_prompts_df[['prompt_id', 'empty_count']].to_dict('records')

    # Generate Markdown content for README.md
    content = "# Model Response Analysis\n\n"
    content += "This report identifies prompts with missing or empty responses across all 10 models.\n\n"

    # Overall Summary Table
    content += "## Overall Summary\n\n"
    content += "| Model | Total Prompts | Prompts with Issues | Total Empty Individual Responses |\n"
    content += "| --- | --- | --- | --- |\n"
    for s in summary_data:
        content += f"| {s['Model']} | {s['Total Prompts']} | {s['Prompts with Issues']} | {s['Total Empty Responses']} |\n"
    content += "\n"

    # Detail Tables for each model
    content += "## Detailed Tables per Model\n\n"
    for s in summary_data:
        model_name = s['Model']
        content += f"### Model: {model_name}\n\n"
        details = model_details.get(model_name, [])
        if not details:
            content += "✅ No empty responses found for this model.\n\n"
        else:
            content += "| Prompt ID | Empty Responses Count |\n"
            content += "| --- | --- |\n"
            for d in details:
                content += f"| {d['prompt_id']} | {d['empty_count']} |\n"
            content += "\n"

    with open("README.md", "w") as f:
        f.write(content)

    print(f"Analysis complete. Found issues in {len(model_details)} models. README.md updated.")

if __name__ == "__main__":
    analyze_models()
