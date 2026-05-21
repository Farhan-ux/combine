import pandas as pd
import json
import os
import glob

def count_empty_responses(json_str):
    if not json_str or (isinstance(json_str, str) and json_str.strip() == ""):
        return 10  # Assuming 10 responses per prompt if entirely empty
    try:
        resps = json.loads(json_str)
        if isinstance(resps, list):
            return sum(1 for r in resps if not r or str(r).strip() == "")
        return 1 if not resps or str(resps).strip() == "" else 0
    except (json.JSONDecodeError, TypeError):
        return 1 if not json_str or str(json_str).strip() == "" else 0

def analyze_models():
    parquet_files = glob.glob("*.parquet")
    parquet_files.sort()

    summary_data = []
    details_data = {}

    for file in parquet_files:
        model_name = file.replace(".parquet", "")
        df = pd.read_parquet(file)

        # Vectorized-ish application
        df['empty_count'] = df['response_text'].apply(count_empty_responses)

        empty_prompts_df = df[df['empty_count'] > 0]

        summary_data.append({
            "Model": model_name,
            "Total Prompts": len(df),
            "Prompts with Empty Responses": len(empty_prompts_df),
            "Total Empty Individual Responses": df['empty_count'].sum()
        })

        if len(empty_prompts_df) > 0:
            details_data[model_name] = empty_prompts_df[['prompt_id', 'empty_count']].to_dict('records')

    # Generate README.md
    content = "# Model Response Analysis\n\n"
    content += "## Overall Summary\n\n"
    content += "| Model | Total Prompts | Prompts with Empty Responses | Total Empty Individual Responses |\n"
    content += "| --- | --- | --- | --- |\n"
    for s in summary_data:
        content += f"| {s['Model']} | {s['Total Prompts']} | {s['Prompts with Empty Responses']} | {s['Total Empty Individual Responses']} |\n"
    content += "\n"

    for model, details in details_data.items():
        content += f"## Empty Response Details: {model}\n\n"
        content += "| Prompt ID | Empty Responses Count |\n"
        content += "| --- | --- |\n"
        for d in details:
            content += f"| {d['prompt_id']} | {d['empty_count']} |\n"
        content += "\n"

    with open("README.md", "w") as f:
        f.write(content)

    print("Analysis complete. README.md updated.")

if __name__ == "__main__":
    analyze_models()
