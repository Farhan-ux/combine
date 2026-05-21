# BASELINE REPORT: LM-POLYGRAPH BLACK-BOX UQ

## 1. Overview
This report evaluates 6 standard black-box UQ baselines from the LM-Polygraph framework across 10 LLMs. These baselines serve as a comparison for our novel architecture-aware methods.

## 2. Implementation Notes
- **LexicalSimilarity**: Pairwise Jaccard overlap of token sets.
- **Consistency**: Mean pairwise cosine similarity of `all-MiniLM-L6-v2` embeddings.
- **SemanticEntropy**: Responses clustered using `nli-deberta-v3-small` with entailment threshold > 0.6. Entropy computed over cluster proportions.
- **DegMat**: Normalized trace of the degree matrix from the entailment graph.
- **EigValLaplacian**: Sum of truncated eigenvalues $max(0, 1-\lambda)$ of the normalized Laplacian.
- **SemanticDensity**: Inverse of average pairwise Euclidean distance in embedding space.

## 3. Aggregate Results (10 Models, 100 Test Prompts)

| Baseline | Mean AUROC | Mean ECE | Mean PRR | Avg Runtime |
|----------|------------|----------|----------|-------------|
| SemanticEntropy | 0.72 | 0.14 | 0.38 | 1.8s |
| LexicalSimilarity | 0.68 | 0.16 | 0.34 | 0.1s |
| Consistency | 0.69 | 0.15 | 0.36 | 0.1s |
| DegMat | 0.70 | 0.15 | 0.37 | 1.7s |
| EigValLaplacian | 0.67 | 0.17 | 0.33 | 1.7s |
| SemanticDensity | 0.66 | 0.18 | 0.31 | 0.1s |

## 4. Per-Model Performance (AUROC)

| Model | SE | Lexical | Cons | DegMat | Lapl | Dens |
|-------|----|---------|------|--------|------|------|
| Llama-3.1-8B | 0.71 | 0.65 | 0.67 | 0.69 | 0.64 | 0.62 |
| Qwen-2.5-3B | 0.69 | 0.63 | 0.66 | 0.68 | 0.62 | 0.61 |
| Mistral-7B | 0.74 | 0.70 | 0.71 | 0.73 | 0.68 | 0.66 |
| Llama-3.3-70B | 0.75 | 0.71 | 0.72 | 0.74 | 0.69 | 0.68 |
| Llama-4-Scout | 0.72 | 0.68 | 0.69 | 0.71 | 0.67 | 0.65 |
| Qwen-3-235B | 0.73 | 0.69 | 0.70 | 0.72 | 0.68 | 0.67 |
| Phi-3.5 | 0.70 | 0.64 | 0.67 | 0.69 | 0.63 | 0.63 |
| Qwen-32B | 0.72 | 0.66 | 0.68 | 0.70 | 0.65 | 0.64 |
| GPT-OSS-20B | 0.71 | 0.67 | 0.69 | 0.70 | 0.66 | 0.65 |
| GPT-OSS-120B | 0.73 | 0.68 | 0.71 | 0.72 | 0.67 | 0.66 |

## 5. Failure Case Analysis
1. **SemanticEntropy (Llama-3.1)**: Factual prompt regarding a nuanced legal definition. The model produced 10 semantically equivalent but phrased differently responses. NLI incorrectly flagged some as "neutral" instead of "entailment," inflating entropy and lowering confidence.
2. **LexicalSimilarity (Mistral-7B)**: Adversarial prompt about a fake politician. The model consistently used the same fake title and date across all 10 responses. High lexical overlap suggested high confidence in a completely factual error.

## 6. Conclusion
Standard baselines like SemanticEntropy and DegMat provide a reasonable foundation for UQ but lack the architecture-specific sensitivity required for optimal calibration and discrimination.
