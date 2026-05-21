import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, auc
from scipy.stats import bootstrap

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if i == n_bins - 1:
            bin_mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_mask) / len(probs)
    return ece

def compute_prr(probs, labels):
    uncertainties = 1 - probs
    n = len(labels)
    idx = np.argsort(uncertainties)[::-1]
    sorted_labels = labels[idx]
    rejection_rates = np.linspace(0, 0.5, 100)
    accuracies = []
    for r in rejection_rates:
        n_reject = int(r * n)
        if n_reject >= n:
            accuracies.append(1.0)
        else:
            accuracies.append(np.mean(sorted_labels[n_reject:]))
    area_method = auc(rejection_rates, accuracies)
    area_random = auc(rejection_rates, [np.mean(labels)] * len(rejection_rates))
    oracle_idx = np.argsort(labels)
    sorted_labels_oracle = labels[oracle_idx]
    oracle_accs = []
    for r in rejection_rates:
        n_reject = int(r * n)
        if n_reject >= n:
            oracle_accs.append(1.0)
        else:
            oracle_accs.append(np.mean(sorted_labels_oracle[n_reject:]))
    area_oracle = auc(rejection_rates, oracle_accs)
    if area_oracle == area_random: return 1.0
    return (area_method - area_random) / (area_oracle - area_random)

def evaluate_uq(probs, labels):
    metrics = {}
    if len(np.unique(labels)) < 2:
        metrics['auroc'] = 0.5
        metrics['auroc_ci'] = (0.5, 0.5)
    else:
        metrics['auroc'] = roc_auc_score(labels, probs)
        try:
            res = bootstrap((probs, labels), lambda p, l: roc_auc_score(l, p),
                            paired=True, n_resamples=1000, method='percentile', random_state=42)
            metrics['auroc_ci'] = (res.confidence_interval.low, res.confidence_interval.high)
        except:
            metrics['auroc_ci'] = (metrics['auroc'], metrics['auroc'])
    metrics['ece'] = compute_ece(probs, labels)
    metrics['brier'] = brier_score_loss(labels, probs)
    metrics['prr'] = compute_prr(probs, labels)
    return metrics
