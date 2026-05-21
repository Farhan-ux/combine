import numpy as np
from scipy.stats import entropy

def lexical_similarity(texts):
    n = len(texts)
    if n <= 1: return 0
    sims = []
    sets = [set(t.lower().split()) for t in texts]
    for i in range(n):
        for j in range(i+1, n):
            if not sets[i] or not sets[j]: sims.append(0)
            else: sims.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return np.mean(sims)

def semantic_entropy_empirical(sample_to_class):
    n = len(sample_to_class)
    counts = {}
    for c in sample_to_class.values():
        counts[c] = counts.get(c, 0) + 1
    probs = np.array(list(counts.values())) / n
    return entropy(probs)

def deg_mat(entail_matrix):
    n = entail_matrix.shape[0]
    W = (entail_matrix + entail_matrix.T) / 2
    D = np.diag(W.sum(axis=1))
    return np.trace(n * np.eye(n) - D) / (n ** 2)

def eig_val_laplacian(entail_matrix):
    n = entail_matrix.shape[0]
    W = (entail_matrix + entail_matrix.T) / 2
    D = np.diag(W.sum(axis=1))
    L = D - W
    try:
        eigs = np.linalg.eigvalsh(L)
        return np.max(eigs) / n
    except: return 1.0

def eccentricity(embeddings):
    center = np.mean(embeddings, axis=0)
    dists = np.linalg.norm(embeddings - center, axis=1)
    return np.mean(dists)

def semantic_density(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    return 1 - np.mean(sims)
