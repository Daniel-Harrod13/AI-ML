# Bayesian Workflow - Quick Reference Guide

## 🔄 The 5-Step Bayesian Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    BAYESIAN INFERENCE CYCLE                      │
└─────────────────────────────────────────────────────────────────┘

1. PRIOR DISTRIBUTION
   ↓
   What we believe BEFORE seeing data
   - Parameter distributions based on domain knowledge
   - Example: Temperature baseline ~ Normal(20°C, 5°C)
   
2. LIKELIHOOD FUNCTION
   ↓
   How data is generated given parameters
   - P(Data | Parameters)
   - Example: Temp ~ Normal(μ = f(sensors), σ²)
   
3. BAYES' THEOREM → POSTERIOR
   ↓
   Update beliefs using observed data
   - P(Params | Data) ∝ P(Data | Params) × P(Params)
   - Use MCMC to sample from posterior
   
4. POSTERIOR PREDICTIVE DISTRIBUTION
   ↓
   Make probabilistic forecasts
   - Sample from posterior → generate predictions
   - Output: Full distribution + credible intervals
   
5. BAYESIAN MODEL AVERAGING (BMA)
   ↓
   Combine multiple models weighted by evidence
   - Fit competing models
   - Weight by P(Model | Data)
   - Ensemble prediction: Σ weights × predictions
```

---

## 📊 Formula Cheat Sheet

### Bayes' Theorem
```
Posterior ∝ Likelihood × Prior

P(θ|D) ∝ P(D|θ) × P(θ)

where:
  θ = parameters (coefficients, variance, etc.)
  D = observed data
```

### Linear Model Structure
```
Temperature_i ~ Normal(μ_i, σ²)

μ_i = β₀ + β₁·humidity_i + β₂·CO_i + β₃·LPG_i 
      + β₄·smoke_i + β₅·light_i + β₆·motion_i 
      + α_device[i]

Priors:
  β_j ~ Normal(0, 2)           # Regression coefficients
  β₀ ~ Normal(20, 5)           # Intercept (baseline temp)
  σ ~ Half-Normal(5)           # Noise
  α_device ~ Normal(0, τ²)     # Device random effects
  τ ~ Half-Normal(2)           # Device variance
```

### Posterior Predictive
```
P(y_new | D) = ∫ P(y_new | θ) × P(θ | D) dθ

In practice (Monte Carlo approximation):
  1. Draw θ^(s) from posterior (s = 1, ..., S samples)
  2. For each θ^(s), draw y_new^(s) ~ P(y | θ^(s))
  3. {y_new^(1), ..., y_new^(S)} = predictive distribution
```

### Bayesian Model Averaging
```
P(y | D) = Σ P(y | M_k, D) × P(M_k | D)
           k

Model weights:
  w_k = P(M_k | D) ∝ P(D | M_k) × P(M_k)
  
where:
  P(D | M_k) = model evidence (from WAIC/LOO)
```

---

## 🔍 Convergence Diagnostics

| Metric | Good Value | What It Means |
|--------|------------|---------------|
| **R-hat** | < 1.01 | Chains have converged |
| **ESS** | > 400 | Enough independent samples |
| **Divergences** | 0 | No sampling issues |
| **Trace plots** | Fuzzy caterpillar | Good mixing |
| **Autocorrelation** | Drops to 0 quickly | Independent samples |

---

## 📈 Model Evaluation

### Information Criteria (Lower = Better)
- **WAIC** (Widely Applicable IC): Bayesian version of AIC
- **LOO-CV** (Leave-One-Out): Cross-validation estimate
- **Marginal Likelihood**: Full model evidence P(D|M)

### Prediction Metrics
- **MAE/RMSE**: Point prediction accuracy
- **Interval Coverage**: % of true values in 95% CI (should be ~95%)
- **Calibration**: Predicted probabilities match empirical frequencies
- **Sharpness**: Width of prediction intervals (narrower = better, if calibrated)

---

## 🛠️ PyMC Code Template

```python
import pymc as pm
import arviz as az

# 1. SPECIFY MODEL
with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('intercept', mu=20, sigma=5)
    beta = pm.Normal('coefficients', mu=0, sigma=2, shape=n_features)
    sigma = pm.HalfNormal('sigma', sigma=5)
    
    # Likelihood
    mu = beta_0 + pm.math.dot(X, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
    
# 2. SAMPLE POSTERIOR
with model:
    trace = pm.sample(2000, tune=1000, chains=4, 
                      target_accept=0.95, return_inferencedata=True)
    
# 3. CHECK CONVERGENCE
print(az.summary(trace, var_names=['intercept', 'coefficients', 'sigma']))
az.plot_trace(trace)
az.plot_forest(trace, var_names=['coefficients'])

# 4. POSTERIOR PREDICTIVE
with model:
    ppc = pm.sample_posterior_predictive(trace, predictions=True)
    
# 5. PREDICTIONS
y_pred_mean = ppc.posterior_predictive['y_obs'].mean(dim=['chain', 'draw'])
y_pred_hdi = az.hdi(ppc.posterior_predictive['y_obs'], hdi_prob=0.95)
```

---

## 🎯 Interpretation Guide

### Parameter Estimates
```
Posterior: β_humidity = 0.32 ± 0.04  [95% HDI: 0.24, 0.40]

Interpretation:
  - 1% increase in humidity → 0.32°C increase in temperature
  - We're 95% confident the true effect is between 0.24 and 0.40
  - Uncertainty: ±0.04 (credible range around mean)
```

### Prediction Intervals
```
Prediction for new observation:
  - Point: 23.2°C (posterior mean)
  - 50% CI: [22.8, 23.6] → Likely range (1-in-2 chance)
  - 95% CI: [21.5, 24.9] → High-confidence range (19-in-20 chance)
  
Action: If temperature falls outside 95% CI → investigate anomaly
```

### Model Comparison
```
Model Weights (BMA):
  Model 1 (Full): 45%
  Model 2 (Hierarchical): 35%
  Model 3 (Reduced): 15%
  Model 4 (Interaction): 5%
  
Interpretation:
  - Full model has most support, but not overwhelming
  - Hierarchical model also plausible (hedge against overfitting)
  - Ensemble prediction combines all four weighted by evidence
```

---

## ✅ Checklist Before Deployment

- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 per chain
- [ ] No divergent transitions
- [ ] Posterior predictive checks pass (observed data looks like simulated)
- [ ] 95% intervals contain ~95% of test data (calibration)
- [ ] Trace plots show good mixing (no trends or stuck chains)
- [ ] Coefficient directions make sense (domain validation)
- [ ] BMA weights reasonable (no single model dominates >90%)
- [ ] Uncertainty intervals are actionable (not too wide/narrow)

---

## 🔗 Key Resources

**PyMC Documentation:** https://www.pymc.io/  
**ArviZ Diagnostics:** https://arviz-devs.github.io/arviz/  
**Bayesian Workflow Paper:** Gelman et al. (2020) - Bayesian Workflow  
**Book:** Statistical Rethinking (McElreath) - Intuitive intro  
**Book:** Bayesian Data Analysis (Gelman et al.) - Comprehensive reference  

---

## 💡 Common Pitfalls

| Problem | Solution |
|---------|----------|
| Non-convergence (R-hat > 1.01) | Increase tune steps, stronger priors, reparameterize |
| Divergences | Increase target_accept to 0.95+, check prior-likelihood mismatch |
| Slow sampling | Use variational inference first, then MCMC; simplify model |
| Wide posteriors | More data, stronger priors, or accept uncertainty |
| Overconfident predictions | Check for model misspecification, add random effects |
| BMA all weight on one model | Good sign OR data too limited to distinguish models |

---

**Remember:** Bayesian inference is about **honest uncertainty quantification**. Wide intervals aren't failures—they're signals to collect more data or acknowledge inherent noise!

