import numpy as np
import json
import zlib
import torch
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

class ArchitectureAwareUQ:
    """
    Architecture-Aware Black-Box Uncertainty Quantification Suite.
    Includes Method 1 (Spectral), Method 2 (Algorithmic), and Method 3 (Logical).
    """
    def __init__(self, device='cpu'):
        self.device = device
        # Models are loaded lazily to save memory if only specific methods are used
        self.st_model = None
        self.nli_tokenizer = None
        self.nli_model = None

    def _init_st(self):
        if self.st_model is None:
            self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)

    def _init_nli(self):
        if self.nli_model is None:
            name = "cross-encoder/nli-deberta-v3-small"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)
            self.nli_model.eval()

    def sigmoid(self, x, kappa, tau):
        return 1 / (1 + np.exp(- (kappa * x + tau)))

    def get_stable_rank(self, embeddings):
        E = embeddings - np.mean(embeddings, axis=0)
        try:
            s = np.linalg.svd(E, compute_uv=False)
            if len(s) == 0 or np.max(s) == 0: return 1.0
            return (np.sum(s**2)) / (np.max(s)**2)
        except: return 1.0

    def get_ncd(self, s1, s2):
        b1, b2 = s1.encode('utf-8'), s2.encode('utf-8')
        c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
        c12 = len(zlib.compress(b1 + b2))
        return (c12 - min(c1, c2)) / max(c1, c2)

    def method1_spectral(self, responses: List[str], category: str) -> float:
        """Topological/Spectral method using Stable Rank. Parameters derived from EVT (Gumbel)."""
        self._init_st()
        emb = self.st_model.encode(responses)
        sr = self.get_stable_rank(emb)
        if category == 'MoE':
            # Derived a priori from unlabeled MoE moments
            return self.sigmoid(sr, 3.664, -9.317)
        else:
            return self.sigmoid(sr, 1.0, -3.5)

    def method2_algorithmic(self, responses: List[str], category: str) -> float:
        """Algorithmic/Information-Theoretic method using NCD. Parameters derived from Gaussian fit."""
        ncds = [self.get_ncd(responses[0], responses[j]) for j in range(1, len(responses))]
        avg_ncd = np.mean(ncds)
        if category == 'Scale Ablation':
            # Derived a priori from unlabeled Scale Ablation moments
            return 1 - self.sigmoid(avg_ncd, 12.500, -9.125)
        else:
            return 1 - self.sigmoid(avg_ncd, 5.0, -3.5)

    def method3_logical(self, responses: List[str], category: str) -> float:
        """Logical/Evidential method using NLI Contradiction. Parameters derived from Beta fit."""
        self._init_nli()
        pairs = [(responses[0], r) for r in responses[1:]]
        encoded = self.nli_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        avg_contra = np.mean(probs[:, 0])
        if category == 'Dense RLHF':
            # Derived a priori from unlabeled RLHF moments
            return 1 - self.sigmoid(avg_contra, 0.744, -0.150)
        else:
            return 1 - self.sigmoid(avg_contra, 10.0, -1.0)

def predict_factuality(responses: List[str], category: str, device='cpu') -> float:
    uq = ArchitectureAwareUQ(device=device)
    # Selection logic based on architecture
    if category == 'MoE':
        return uq.method1_spectral(responses, category)
    elif category == 'Scale Ablation':
        return uq.method2_algorithmic(responses, category)
    elif category == 'Dense RLHF':
        return uq.method3_logical(responses, category)
    else:
        # Default to a hybrid or Method 1
        return uq.method1_spectral(responses, category)

if __name__ == "__main__":
    test_responses = ["Paris is the capital of France.", "France's capital is Paris.", "The city of Paris is the capital."]
    print(f"P(factual): {predict_factuality(test_responses, 'MoE'):.4f}")
