import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class FeatureCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.nli_name = "cross-encoder/nli-deberta-v3-small"
        self.nli_tokenizer = None
        self.nli_model = None
        self.st_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model = SentenceTransformer(self.st_name, device=device)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.st_model.encode(texts)

    def _init_nli(self):
        if self.nli_model is None:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_name).to(self.device)
            self.nli_model.eval()

    def get_nli_matrix(self, texts: List[str]) -> Dict[str, np.ndarray]:
        self._init_nli()
        n = len(texts)
        entail_matrix = np.eye(n)
        contra_matrix = np.zeros((n, n))
        unique_texts, inv = np.unique(texts, return_inverse=True)
        nu = len(unique_texts)
        u_entail = np.eye(nu)
        u_contra = np.zeros((nu, nu))
        pairs = []
        indices = []
        for i in range(nu):
            for j in range(nu):
                if i != j:
                    pairs.append((unique_texts[i], unique_texts[j]))
                    indices.append((i, j))
        if pairs:
            batch_size = 32
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                encoded = self.nli_tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    logits = self.nli_model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for k, (idx_i, idx_j) in enumerate(indices[i:i+batch_size]):
                    u_contra[idx_i, idx_j] = probs[k, 0]
                    u_entail[idx_i, idx_j] = probs[k, 1]
        for i in range(n):
            for j in range(n):
                entail_matrix[i, j] = u_entail[inv[i], inv[j]]
                contra_matrix[i, j] = u_contra[inv[i], inv[j]]
        return {"entail": entail_matrix, "contra": contra_matrix}

def get_semantic_classes(entail_matrix, threshold=0.5):
    n = entail_matrix.shape[0]
    sample_to_class = {}
    class_to_sample = []
    is_entail = entail_matrix > threshold
    for i in range(n):
        found = False
        for class_id, members in enumerate(class_to_sample):
            representative = members[0]
            if is_entail[i, representative] and is_entail[representative, i]:
                class_to_sample[class_id].append(i)
                sample_to_class[i] = class_id
                found = True
                break
        if not found:
            sample_to_class[i] = len(class_to_sample)
            class_to_sample.append([i])
    return sample_to_class, class_to_sample
