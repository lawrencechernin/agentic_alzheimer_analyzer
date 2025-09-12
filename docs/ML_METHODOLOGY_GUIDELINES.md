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

## 🎯 Key Principle

**Lower honest metrics > Higher invalid metrics**

A properly evaluated 0.75 AUC is infinitely more valuable than an improperly evaluated 0.90 AUC. The former can be trusted for clinical deployment; the latter is dangerous misinformation.

## References
- Incident: `bhr_memtrax_stable_0798.py` (invalid 0.798 AUC)
- Corrected: `bhr_memtrax_corrected_methodology.py` (proper methodology)
- Valid examples: `bhr_memtrax_minimal_0798.py` (uses train/test split)
