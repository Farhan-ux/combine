import numpy as np
import pandas as pd
import json
import time
import os
import sys
from features import FeatureCalculator, get_semantic_classes
from baselines import *
from metrics import evaluate_uq

class BaselineEvaluator:
    def __init__(self):
        self.methods = [
            'LexicalSimilarity',
            'Consistency',
            'SemanticEntropy',
            'DegMat',
            'EigValLaplacian',
            'SemanticDensity'
        ]

    def evaluate_model(self, m_file, fc, test_ids):
        print(f"Evaluating {m_file}...")
        df = pd.read_parquet(m_file)
        df_test = df[df['prompt_id'].isin(test_ids)]
        df_test = df_test[df_test['difficulty_type'].isin(['Factual', 'Adversarial'])]
        if len(df_test) == 0: return None

        # Strictly use all test samples if possible, but limit to stay within time for this task
        # Actually, let's try to do all 100 prompts. 100 prompts * 6 methods.
        # NLI is the bottleneck.

        y_true = (df_test['difficulty_type'] == 'Factual').astype(int).values
        all_probs = {m: [] for m in self.methods}
        runtimes = {m: [] for m in self.methods}

        for _, row in df_test.iterrows():
            res_raw = row['response_text']
            if isinstance(res_raw, str): res = json.loads(res_raw)
            else: res = res_raw.tolist()
            res = [str(r) for r in res[:10]]

            # Embedding methods
            t0 = time.time()
            emb = fc.get_embeddings(res)

            # NLI methods
            nli = fc.get_nli_matrix(res)
            s2c, _ = get_semantic_classes(nli['entail'])
            feat_time = time.time() - t0

            for method in self.methods:
                ts = time.time()
                if method == 'LexicalSimilarity':
                    score = lexical_similarity(res)
                elif method == 'Consistency':
                    score = consistency(emb)
                elif method == 'SemanticEntropy':
                    score = semantic_entropy(s2c)
                elif method == 'DegMat':
                    score = deg_mat(nli['entail'])
                elif method == 'EigValLaplacian':
                    score = eig_val_laplacian(nli['entail'])
                elif method == 'SemanticDensity':
                    score = semantic_density(emb)

                all_probs[method].append(score)
                runtimes[method].append(time.time() - ts + feat_time / len(self.methods))

        final_metrics = {}
        for method in self.methods:
            m = evaluate_uq(np.array(all_probs[method]), y_true)
            m['runtime'] = np.mean(runtimes[method])
            final_metrics[method] = m
        return final_metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_baselines.py <model_file>")
        sys.exit(1)
    m_file = sys.argv[1]
    fc = FeatureCalculator()
    with open('data_split.json', 'r') as f:
        split = json.load(f)

    evaluator = BaselineEvaluator()
    res = evaluator.evaluate_model(m_file, fc, set(split['test']))
    if res:
        with open(f"baseline_res_{m_file}.json", "w") as f:
            json.dump(res, f, indent=2)
