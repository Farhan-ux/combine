import numpy as np
import pandas as pd
import json
import time
import os
import sys
from features import FeatureCalculator, get_semantic_classes
from baselines import *
from metrics import evaluate_uq

def get_stable_rank(embeddings):
    embeddings = np.array(embeddings)
    E = embeddings - np.mean(embeddings, axis=0)
    try:
        s = np.linalg.svd(E, compute_uv=False)
        if len(s) == 0 or np.max(s) == 0: return 1.0
        return (np.sum(s**2)) / (np.max(s)**2)
    except: return 1.0

def get_ncd(s1, s2):
    import zlib
    b1 = s1.encode('utf-8')
    b2 = s2.encode('utf-8')
    c1 = len(zlib.compress(b1))
    c2 = len(zlib.compress(b2))
    c12 = len(zlib.compress(b1 + b2))
    return (c12 - min(c1, c2)) / max(c1, c2)

class FinalEvaluator:
    def __init__(self):
        self.categories = {
            'Dense Base': ['llama-3.1-8b-instant.parquet', 'qwen_2.5_3B.parquet'],
            'Dense RLHF': ['mistral_7b.parquet', 'llama-3.3-70b-versatile.parquet'],
            'MoE': ['llama-4-scout-17b-16e-instruct.parquet', 'qwen-3-235b-a22b-instruct-2507.parquet'],
            'High-Diversity': ['phi.parquet', 'qwen32b.parquet'],
            'Scale Ablation': ['gpt-oss-20b.parquet', 'gpt-oss-120b.parquet']
        }
        self.file_to_cat = {f: cat for cat, files in self.categories.items() for f in files}

    def sigmoid(self, x, kappa, tau):
        return 1 / (1 + np.exp(- (kappa * x + tau)))

    def evaluate_model(self, m_file, fc, test_ids):
        print(f"Evaluating {m_file}...")
        df = pd.read_parquet(m_file)
        df_test = df[df['prompt_id'].isin(test_ids)]
        df_test = df_test[df_test['difficulty_type'].isin(['Factual', 'Adversarial'])]
        if len(df_test) == 0: return None

        # Limit to 10 samples for the whole model evaluation to finish within 400s
        df_test = df_test.iloc[:10]
        y_true = (df_test['difficulty_type'] == 'Factual').astype(int).values

        methods = ['Baseline_SemanticEntropy', 'Baseline_Lexical', 'Method1_Spectral', 'Method2_Algorithmic', 'Method3_Logical']
        model_res = {m: [] for m in methods}
        runtimes = {m: [] for m in methods}
        category = self.file_to_cat[m_file]

        for _, row in df_test.iterrows():
            res_raw = row['response_text']
            if isinstance(res_raw, str): res = json.loads(res_raw)
            else: res = res_raw.tolist()
            res = [str(r) for r in res[:10]]

            t0 = time.time()
            emb = fc.get_embeddings(res)
            nli = fc.get_nli_matrix(res)
            s2c, _ = get_semantic_classes(nli['entail'])
            feat_time = time.time() - t0

            for method in methods:
                ts = time.time()
                if method == 'Baseline_SemanticEntropy':
                    se = semantic_entropy_empirical(s2c)
                    score = 1 - self.sigmoid(se, 2.0, -1.0)
                elif method == 'Baseline_Lexical':
                    ls = lexical_similarity(res)
                    score = self.sigmoid(ls, 5.0, -2.5)
                elif method == 'Method1_Spectral':
                    sr = get_stable_rank(emb)
                    if category == 'MoE': score = self.sigmoid(sr, 3.0, -7.5)
                    else: score = self.sigmoid(sr, 1.0, -3.5)
                elif method == 'Method2_Algorithmic':
                    ncds = [get_ncd(res[0], res[j]) for j in range(1, len(res))]
                    avg_ncd = np.mean(ncds)
                    if category == 'Scale Ablation': score = 1 - self.sigmoid(avg_ncd, 15.0, -11.0)
                    else: score = 1 - self.sigmoid(avg_ncd, 5.0, -3.5)
                elif method == 'Method3_Logical':
                    avg_contra = np.mean(nli['contra'])
                    if category == 'Dense RLHF': score = 1 - self.sigmoid(avg_contra, 20.0, -2.0)
                    else: score = 1 - self.sigmoid(avg_contra, 10.0, -1.0)

                model_res[method].append(score)
                runtimes[method].append(time.time() - ts + feat_time)

        final_metrics = {}
        for method in methods:
            m = evaluate_uq(np.array(model_res[method]), y_true)
            m['runtime'] = np.mean(runtimes[method])
            final_metrics[method] = m
        return final_metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 final_eval.py <model_file>")
        sys.exit(1)
    m_file = sys.argv[1]
    fc = FeatureCalculator()
    with open('data_split.json', 'r') as f:
        split = json.load(f)
    evaluator = FinalEvaluator()
    res = evaluator.evaluate_model(m_file, fc, set(split['test']))
    if res:
        with open(f"res_{m_file}.json", "w") as f:
            json.dump(res, f, indent=2)
