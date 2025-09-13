# Machine Learning Methodology Guidelines

## Critical Learnings from BHR MemTrax Analysis

### ⚠️ The 0.798 AUC Incident
We discovered that `bhr_memtrax_stable_0798.py` reported an AUC of 0.798, but this was **completely invalid** due to evaluating the model on its training data. The actual performance would likely be 0.70-0.75.

## 🚨 Critical ML Evaluation Rules

### 1. **Always Use Train/Test Split**
```python
# CORRECT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]  # Evaluate on TEST
auc = roc_auc_score(y_test, y_pred)

# WRONG - Training set evaluation
model.fit(X, y)
y_pred = model.predict_proba(X)[:, 1]  # Same data!
auc = roc_auc_score(y, y_pred)  # Invalid!
```

### 2. **Common Data Leakage Patterns**

#### Pattern 1: Feature Selection Before Split
```python
# WRONG
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Sees all data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y)

# CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y)
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)  # Only transform
```

#### Pattern 2: Calibration on Full Dataset
```python
# WRONG
model.fit(X, y)
calibrated = CalibratedClassifierCV(model)
calibrated.fit(X, y)  # Calibrating on same data

# CORRECT
model.fit(X_train, y_train)
calibrated = CalibratedClassifierCV(model)
calibrated.fit(X_train, y_train)
y_pred = calibrated.predict_proba(X_test)[:, 1]
```

#### Pattern 3: Stacking Without Holdout
```python
# WRONG
stack = StackingClassifier(estimators, cv=5)
stack.fit(X, y)
y_pred = stack.predict_proba(X)[:, 1]  # Internal CV doesn't help!

# CORRECT
stack = StackingClassifier(estimators, cv=5)
stack.fit(X_train, y_train)
y_pred = stack.predict_proba(X_test)[:, 1]
```

## 📊 Expected Performance Ranges

### Medical/Cognitive Data (Tabular)
- **Demographics only**: 0.60-0.65 AUC
- **Clinical assessments**: 0.70-0.80 AUC  
- **With biomarkers**: 0.75-0.85 AUC
- **With neuroimaging**: 0.80-0.90 AUC

### Red Flags
- AUC >0.90 on small datasets (<10k samples)
- Test AUC exactly matching CV AUC
- No train/test sizes reported
- Single X, y variables throughout code

## 🔍 Quick Code Audit Checklist

1. **Check imports**: Look for `train_test_split` or `StratifiedKFold`
2. **Search for `.fit(`**: What data is being used? Should be `X_train`
3. **Search for `.predict`**: Should use `X_test`, not same as fit
4. **Search for `roc_auc_score`**: First arg should be `y_test`
5. **Trace data flow**: X, y → split → X_train/test → model → metrics

## 🔧 How to Fix Invalid Code

### Step 1: Add Proper Split
```python
# Right after loading and merging data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
```

### Step 2: Fix All `.fit()` Calls
```python
# Find: model.fit(X
# Replace: model.fit(X_train
```

### Step 3: Fix All Predictions
```python
# Find: predict_proba(X)
# Replace: predict_proba(X_test)
```

### Step 4: Use Pipelines for Preprocessing
```python
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=20)),
    ('classifier', LogisticRegression())
])
pipe.fit(X_train, y_train)  # Entire pipeline on train
y_pred = pipe.predict_proba(X_test)[:, 1]
```

## 📈 Impact of Methodology Issues

| Issue | Typical AUC Inflation | Example |
|-------|----------------------|---------|
| Training set evaluation | +0.05-0.15 | 0.798 → 0.70 |
| Feature selection leakage | +0.03-0.05 | 0.75 → 0.70 |
| Calibration on full data | +0.02-0.03 | 0.73 → 0.70 |
| No regularization | +0.05-0.10 | 0.80 → 0.72 |
| Cherry-picked threshold | +0.02-0.05 | 0.75 → 0.71 |

## ✅ Best Practices

1. **Always report**:
   - Train/test sample sizes
   - Class distribution in each set
   - Cross-validation protocol
   - Random seeds for reproducibility

2. **Use nested CV for model selection**:
   - Outer loop: Final evaluation
   - Inner loop: Hyperparameter tuning

3. **Save predictions**:
   - Store test set predictions for later analysis
   - Enables error analysis and model debugging

4. **Compare to baselines**:
   - Always include demographics-only model
   - Report improvement over baseline

## 6. Decision Threshold Optimization

### The Hidden Performance Killer

Default 0.5 probability threshold can make excellent models appear useless:

```python
# WRONG - Using default threshold blindly
predictions = (model.predict_proba(X)[:, 1] >= 0.5)
# Result: 0.2% sensitivity (misses 99.8% of cases!)

# RIGHT - Optimize threshold for clinical goals
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
# Find threshold for 80% sensitivity
idx = np.argmin(np.abs(tpr - 0.80))
optimal_threshold = thresholds[idx]

predictions = (model.predict_proba(X)[:, 1] >= optimal_threshold)
# Result: 80% sensitivity (catches most cases!)
```

### When This Matters Most

1. **Imbalanced datasets** (prevalence < 10%)
2. **Calibrated models** (output conservative probabilities)
3. **Medical screening** (false negatives worse than false positives)
4. **Ensemble methods** (averaged probabilities tend toward 0.5)

### Optimization Strategies

| Strategy | Use Case | Method |
|----------|----------|--------|
| **Youden's J** | Balanced performance | `max(sensitivity + specificity - 1)` |
| **Target Sensitivity** | Screening | Find threshold for 80-90% sensitivity |
| **F1 Maximum** | Precision-recall balance | Grid search F1 scores |
| **Cost-Weighted** | Economic optimization | Minimize total misclassification cost |

### Real Impact Example

BHR MemTrax MCI Detection (5.9% prevalence):
- **Default (0.5)**: Detects 1/424 cases (0.2% sensitivity)
- **Optimized (0.043)**: Detects 340/424 cases (80% sensitivity)
- **Same model, 339 more patients helped!**

### Implementation Checklist

✅ Always evaluate multiple thresholds
✅ Report sensitivity/specificity at each threshold
✅ Choose based on clinical goals, not defaults
✅ Document threshold in deployment code
✅ Re-optimize for new populations

### Key Insight

**AUC doesn't change with threshold, but clinical utility does!**

A 0.73 AUC model can be either:
- Useless (0.2% sensitivity with default threshold)
- Highly valuable (80% sensitivity with optimized threshold)

## 🎯 Key Principles

1. **Lower honest metrics > Higher invalid metrics**
   - 0.75 AUC with proper methodology > 0.90 AUC with data leakage

2. **Threshold optimization is "free" performance**
   - No retraining needed, just change one number
   - Can improve sensitivity by 50-80% absolute

3. **Clinical utility > Statistical significance**
   - Focus on actionable metrics (sensitivity at specificity)
   - Consider deployment context and costs

## References
- Incident: `bhr_memtrax_stable_0798.py` (invalid 0.798 AUC)
- Threshold discovery: `bhr_memtrax_optimized_threshold.py` (0.2% → 80% sensitivity)
- Valid examples: `bhr_memtrax_best_result_0744.py` (proper methodology)
