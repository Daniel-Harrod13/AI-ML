# Bayesian Temperature Prediction - Execution Guide

## Quick Start (5 Minutes)

### Step 1: Install Required Libraries
```bash
cd /Users/danielharrod/AI:ML/Telemetry_project
pip install -r requirements.txt
```

This will install:
- PyMC 5.10+ (Bayesian modeling)
- ArviZ 0.17+ (diagnostics & visualization)
- Bambi 0.13+ (high-level interface)
- Standard data science stack (pandas, numpy, scipy, matplotlib, seaborn)

### Step 2: Verify Data is Available
```bash
ls -lh data/iot_telemetry_data.csv
```

If the file exists (~59MB), you're ready. If not, download it:
```bash
python download_dataset.py
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook notebooks/bayesian_temperature_prediction.ipynb
```

### Step 4: Run All Cells
- In Jupyter: `Kernel` → `Restart & Run All`
- **Estimated runtime**: 15-30 minutes
- **Expected output**: 68 cells with EDA, modeling, and results

---

## What the Notebook Does

### Section-by-Section Breakdown

| Section | Cells | Runtime | Output |
|---------|-------|---------|--------|
| **1. Data Loading & EDA** | 1-18 | ~2 min | Dataset overview, visualizations |
| **2. Preprocessing** | 19-27 | ~1 min | Feature engineering, train/test split |
| **3. Prior Specification** | 28-29 | <1 min | Prior distributions visualization |
| **4. Bayesian Models (5x)** | 30-48 | ~12-20 min | MCMC sampling, convergence checks |
| **5. Posterior Predictive** | 49-54 | ~2-3 min | Forecasts with uncertainty |
| **6. Model Averaging** | 55-59 | ~1 min | WAIC/LOO comparison, weights |
| **7. Feature Importance** | 60-64 | <1 min | Coefficient analysis |
| **8. Key Findings** | 65-66 | <1 min | Summary statistics |
| **9. Conclusions** | 67 | <1 min | Documentation |

### Computational Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Disk: 1GB free space
- Time: ~30 minutes

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- Disk: 2GB free space
- Time: ~15 minutes

**Optional GPU Acceleration**:
```bash
# Install JAX for GPU support (NVIDIA only)
pip install jax[cuda12]
```
PyMC can leverage JAX for faster MCMC on GPUs.

---

## Understanding the Output

### 1. Convergence Diagnostics (Critical!)

After each model, check these metrics:

```python
# Good convergence indicators:
R-hat < 1.01        # ✓ Chains have converged
ESS > 400           # ✓ Sufficient effective samples
Divergences = 0     # ✓ No sampling issues
```

**What to do if convergence fails:**
- Increase `tune=2000` (more burn-in)
- Increase `draws=2000` (more samples)
- Adjust priors (less informative)
- Check for multicollinearity (remove correlated features)

### 2. Model Comparison Results

The notebook will output:
```
Model                             WAIC      Δ WAIC    Weight
================================================================
Model 2: Hierarchical            12345.67    0.00     0.4523
Model 5: Time Series (AR)        12347.89    2.22     0.3012
Model 1: Simple Linear           12352.14    6.47     0.1845
...
```

**Interpretation:**
- **Δ WAIC < 2**: Models are equivalent
- **Δ WAIC 2-6**: Weak evidence for best model
- **Δ WAIC > 10**: Strong evidence for best model
- **Weight**: Probability mass; use BMA if no model has > 90%

### 3. Prediction Performance

Expected test set metrics:
```
RMSE:  0.5-0.8°C   (target: < 1.0°C)
MAE:   0.4-0.6°C
R²:    0.90-0.95

Credible Interval Coverage:
  50% CI: ~50% of observations
  80% CI: ~80% of observations
  95% CI: ~95% of observations (good calibration!)
```

### 4. Feature Importance

The notebook ranks sensors by predictive power:
```
Feature         Coef Mean    95% HDI              P(β>0)    Importance
==========================================================================
HUMIDITY        +0.8543     [0.8234, 0.8852]     100.0%    ***
SMOKE           +0.3421     [0.3187, 0.3655]      99.8%    **
LPG             +0.2134     [0.1923, 0.2345]      98.2%    *
...
```

**Interpretation:**
- `***` = Strong effect (use in all models)
- `**` = Moderate effect (include if computationally feasible)
- `*` = Weak effect (consider dropping for simpler models)

---

## Troubleshooting

### Issue 1: "ImportError: No module named 'pymc'"
**Solution:**
```bash
pip install --upgrade pymc arviz bambi
```

### Issue 2: MCMC sampling is very slow
**Solutions:**
- Reduce training sample size in Cell 31:
  ```python
  train_sample_idx = np.random.choice(len(train_data), size=20000, replace=False)  # Reduce from 50k to 20k
  ```
- Reduce MCMC iterations:
  ```python
  trace = pm.sample(draws=500, tune=500, chains=2)  # Instead of 1000/1000/4
  ```
- Use fewer models (comment out Models 3-5)

### Issue 3: "Memory Error" during sampling
**Solutions:**
- Reduce sample size (see Issue 2)
- Close other applications
- Use `pm.sample(cores=2)` instead of default (uses fewer parallel chains)

### Issue 4: Divergent transitions detected
**Solution:**
```python
# Increase target_accept in sampling
trace = pm.sample(draws=1000, tune=1000, target_accept=0.95)  # Default is 0.8
```

### Issue 5: Notebook kernel crashes
**Solutions:**
- Restart kernel: `Kernel` → `Restart`
- Clear output: `Cell` → `All Output` → `Clear`
- Run cells individually instead of "Run All"
- Check memory: Close browser tabs, restart Jupyter

---

## Customization Options

### 1. Change Training Sample Size
In Cell 31:
```python
# For faster execution (lower accuracy):
train_sample_idx = np.random.choice(len(train_data), size=20000, replace=False)

# For better accuracy (slower):
train_sample_idx = np.random.choice(len(train_data), size=100000, replace=False)
```

### 2. Modify Priors
In model definition cells (32, 37, 41, 44, 47):
```python
# More informative prior (stronger regularization):
beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=1.0)  # Instead of 2.5

# Less informative prior (let data dominate):
beta_humidity = pm.Normal('beta_humidity', mu=0, sigma=5.0)
```

### 3. Add Your Own Model (Model 6)
After Cell 48, add:
```python
# MODEL 6: Your custom model
with pm.Model() as model_6:
    # Define your priors
    intercept = pm.Normal('intercept', mu=20, sigma=5)
    beta_custom = pm.Normal('beta_custom', mu=0, sigma=2.5)
    
    # Your likelihood
    mu = intercept + beta_custom * X_custom
    
    # Sampling
    trace_6 = pm.sample(draws=1000, tune=1000, chains=4)
```

### 4. Change Test Set Predictions
In Cell 50:
```python
# Predict on full test set (slower):
test_sample = test_data.copy()

# Or predict on specific device:
test_sample = test_data[test_data['device'] == 'b8:27:eb:bf:9d:51'].copy()
```

---

## Expected Results Summary

After running the full notebook, you should have:

### Quantitative Results
- ✓ 5 converged Bayesian models (R-hat < 1.01)
- ✓ Test RMSE < 1.0°C (success criterion)
- ✓ Well-calibrated 95% credible intervals
- ✓ Identified top 3 predictive sensors
- ✓ Quantified device-specific variation

### Visualizations Generated
- ~30 publication-quality figures
- All major plots automatically saved in notebook

### Insights Delivered
1. **Humidity** is the most predictive sensor (80%+ of signal)
2. **Devices differ** by ~0.5-1.0°C systematically
3. **Temperature is highly predictable** (R² > 0.90)
4. **Model uncertainty exists** (BMA recommended over single model)
5. **2-5% of readings** are anomalies (potential sensor faults)

---

## Next Steps After Completion

### 1. Save Trained Models
Add at the end of notebook:
```python
# Save best model trace
import pickle
with open('best_model_trace.pkl', 'wb') as f:
    pickle.dump(trace_2, f)

# Save predictions
np.save('test_predictions_mean.npy', pp_mean)
np.save('test_predictions_lower95.npy', pp_lower_95)
np.save('test_predictions_upper95.npy', pp_upper_95)
```

### 2. Export Report
In Jupyter:
- `File` → `Download as` → `HTML` (for sharing)
- `File` → `Download as` → `PDF via LaTeX` (for reports)

### 3. Deploy for Production
Consider:
- PyMC can't be easily serialized for deployment
- **Option A**: Use scikit-learn with posterior means as point estimates
- **Option B**: Rebuild model in Stan (easier serialization)
- **Option C**: API wrapper around saved trace (heavier but full Bayesian)

---

## Support & References

### Documentation
- **PyMC**: https://www.pymc.io/
- **ArviZ**: https://arviz-devs.github.io/arviz/
- **Bambi**: https://bambinos.github.io/bambi/

### Learning Resources
- *Statistical Rethinking* by Richard McElreath
- *Bayesian Modeling and Computation in Python* by Martin et al.
- PyMC Discourse: https://discourse.pymc.io/

### Common Questions

**Q: Can I use this with my own IoT data?**  
A: Yes! Replace `iot_telemetry_data.csv` and update column names in cells 3-4.

**Q: How do I interpret credible intervals?**  
A: "95% CI [18.2, 22.4]" means we're 95% certain the true temp is between 18.2-22.4°C.

**Q: Is Bayesian better than XGBoost/Neural Nets?**  
A: Different use cases:
- **Bayesian**: Uncertainty quantification, interpretability, small data
- **XGBoost/NNs**: Pure predictive accuracy, large data, less interpretable

**Q: Can I speed up inference?**  
A: Yes - use variational inference:
```python
with model:
    approx = pm.fit(n=50000, method='advi')  # Much faster than MCMC
    trace = approx.sample(1000)
```

---

## Troubleshooting Contact

If you encounter issues not covered here:
1. Check PyMC Discourse: https://discourse.pymc.io/
2. Review error messages carefully (often suggest fixes)
3. Verify library versions: `pip list | grep pymc`

---

**Happy Bayesian Modeling! 🎲📊**

