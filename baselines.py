import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

def lexical_similarity(texts):
    """Average Jaccard similarity across all pairs."""
    n = len(texts)
    if n <= 1: return 1.0
    sims = []
    sets = [set(t.lower().split()) for t in texts]
    for i in range(n):
        for j in range(i+1, n):
            if not sets[i] or not sets[j]:
                if not sets[i] and not sets[j]: sims.append(1.0)
                else: sims.append(0.0)
            else:
                sims.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return np.mean(sims)

def consistency(embeddings):
    """Mean pairwise semantic similarity (cosine similarity)."""
    # Prompt asked for "Mean pairwise semantic dissimilarity (1 - cosine similarity)"
    # but also said "higher = more factual". So we return similarity.
    n = len(embeddings)
    if n <= 1: return 1.0
    sims = cosine_similarity(embeddings)
    # Average off-diagonal
    mask = ~np.eye(n, dtype=bool)
    return np.mean(sims[mask])

def semantic_entropy(sample_to_class):
    """Entropy over cluster probability masses (negated)."""
    n = len(sample_to_class)
    if n == 0: return 1.0
    counts = {}
    for c in sample_to_class.values():
        counts[c] = counts.get(c, 0) + 1
    probs = np.array(list(counts.values())) / n
    ent = entropy(probs)
    # Higher entropy = more uncertain.
    # To get higher = more factual, we can use exp(-entropy)
    return np.exp(-ent)

def deg_mat(entail_matrix):
    """Normalized trace of degree matrix similarity."""
    n = entail_matrix.shape[0]
    if n <= 1: return 1.0
    W = (entail_matrix + entail_matrix.T) / 2
    D = W.sum(axis=1)
    # Degree matrix trace based confidence
    # Higher similarity sum -> higher confidence
    return np.mean(D) / n

def eig_val_laplacian(entail_matrix):
    """Sum of truncated eigenvalues max(0, 1-λ) of normalized Laplacian."""
    n = entail_matrix.shape[0]
    if n <= 1: return 1.0
    W = (entail_matrix + entail_matrix.T) / 2
    D = W.sum(axis=1)
    # Normalized Laplacian L = I - D^-1/2 W D^-1/2
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-9))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    try:
        eigs = np.linalg.eigvalsh(L)
        # Higher = more factual
        return np.sum(np.maximum(0, 1 - eigs)) / n
    except:
        return 0.5

def semantic_density(embeddings):
    """Inverse of average pairwise distance."""
    from scipy.spatial.distance import pdist
    n = len(embeddings)
    if n <= 1: return 1.0
    dists = pdist(embeddings, metric='euclidean')
    avg_dist = np.mean(dists)
    # Confidence: higher = closer
    return 1.0 / (1.0 + avg_dist)
