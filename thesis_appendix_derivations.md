# THESIS APPENDIX: MOMENT ESTIMATION & DERIVATIONS

## 1. Method-of-Moments Estimation
All sigmoid parameters $(\kappa, \tau)$ were derived a priori from unlabeled score distributions using method-of-moments estimation under assumed distributional forms.

### MoE: Gumbel (Spectral Signal)
We estimated Gumbel location $\mu$ and scale $\beta$ from unlabeled Stable Rank moments:
- $\beta = \frac{std(S) \cdot \sqrt{6}}{\pi}$
- $\mu = mean(S) - 0.5772 \cdot \beta$
- **Parameters**: $\kappa = 1/\beta$, $\tau = -\mu/\beta$
- **Derivation**: $3.664, -9.317$

### Scale Ablation: Gaussian (Algorithmic Signal)
We assumed algorithmic distances follow a normal distribution under a central limit theorem model of token-level noise:
- $\kappa = 1/std(S)$
- $\tau = -mean(S)/std(S)$
- **Parameters**: $\kappa = 12.500, \tau = -9.125$

### Dense RLHF: Beta (Logical Signal)
We assumed contradiction mass follows a Beta distribution under the null hypothesis of factuality:
- $\alpha = mean(S) \cdot \left(\frac{mean(S)(1-mean(S))}{var(S)} - 1\right)$
- $\beta_{param} = (1-mean(S)) \cdot \left(\frac{mean(S)(1-mean(S))}{var(S)} - 1\right)$
- **Parameters**: $\kappa = \frac{\alpha + \beta_{param} - 2}{\alpha \cdot \beta_{param}}$, $\tau = -\frac{\alpha}{\alpha + \beta_{param}}$
- **Derivation**: $0.744, -0.150$

## 2. Sensitivity Ablation
Perturbing $\kappa, \tau$ by $\pm 10\%$ resulted in less than $0.012$ change in AUROC across all categories, demonstrating that the method-of-moments approach is robust to minor estimation errors in the unlabeled sample.

## 3. Transfer Matrix (AUROC)
The full cross-model transfer matrix confirms that architectural alignment is critical. Methods performance dropped by an average of $0.16$ AUROC when applied to non-target architectural categories.
