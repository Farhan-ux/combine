import pandas as pd
from sklearn.model_selection import train_test_split
import json

def main():
    df = pd.read_parquet('mistral_7b.parquet')
    prompt_data = df[['prompt_id', 'difficulty_type']].drop_duplicates()
    dev_ids, test_ids = train_test_split(
        prompt_data['prompt_id'],
        test_size=100,
        random_state=42,
        stratify=prompt_data['difficulty_type']
    )
    split = {'dev': dev_ids.tolist(), 'test': test_ids.tolist()}
    with open('data_split.json', 'w') as f:
        json.dump(split, f)
    print("Split completed.")

if __name__ == "__main__":
    main()
