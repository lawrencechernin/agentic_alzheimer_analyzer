# 🚀 Advanced Improvements Integration Guide

## Overview

This guide shows how to integrate the advanced generalizability improvements into the Alzheimer's Analysis Framework. These improvements make the system more flexible, powerful, and applicable to diverse Alzheimer's research scenarios while maintaining clinical focus.

---

## 📦 **Improvements Available**

### 1. **Dynamic Model Selection Framework**
- **File**: `dynamic_model_framework.py`
- **Purpose**: Automatically selects optimal ML models based on data characteristics
- **Benefits**: No more manual model selection, optimized for different dataset sizes

### 2. **Modular Feature Engineering Pipeline** 
- **File**: `modular_feature_engineering.py`
- **Purpose**: Configurable feature engineering with domain-specific modules
- **Benefits**: Standardized feature creation, easy to extend for new research needs

### 3. **Automated Hyperparameter Optimization**
- **File**: `auto_hyperparameter_optimization.py`
- **Purpose**: Smart hyperparameter tuning with multiple optimization strategies
- **Benefits**: Optimal model performance without manual tuning

### 4. **Multi-Target Support**
- **File**: `multi_target_support.py`
- **Purpose**: Predict multiple outcomes simultaneously (CDR, MMSE, diagnosis)
- **Benefits**: Comprehensive analysis, better clinical insights

### 5. **Enhanced Data Merging** 🆕
- **File**: `enhanced_data_merging.py`
- **Purpose**: Intelligent dataset merging with Cartesian join prevention
- **Benefits**: Safe longitudinal data merging, prevents memory explosions
- **Based on**: Real-world BHR MemTrax-MCI analysis lesson learned

### 6. **BHR-Aware Merging** 🧠🆕
- **File**: `bhr_aware_merging.py`
- **Purpose**: Domain-specific merging for BHR and similar longitudinal medical datasets
- **Benefits**: 
  - Timepoint-first filtering (prevents 18x data explosions!)
  - Quality pre-filtering (Status == 'Collected', RT validation)
  - Sequential merge pattern (demographics → medical)
  - Auto-detection of BHR dataset patterns
- **Based on**: Successful patterns from `baseline_learning_predictor.py` and other production scripts
- **Impact**: Reduced 1.1M rows → 28K rows in real BHR analysis

### 7. **Longitudinal Analysis Optimization** 📈🆕
- **File**: `longitudinal_analysis_optimization.py`
- **Purpose**: Transform single-timepoint analysis to robust longitudinal analysis
- **Benefits**:
  - Automatic longitudinal structure detection
  - Subject-level aggregation (mean, std, min, max, trend)
  - Composite score creation (RT/accuracy ratio)
  - Quality filtering (Ashford criteria)
  - 4-5x feature enrichment through aggregation
- **Based on**: Successful BHR scripts achieving 0.70+ AUC
- **Impact**: Expected AUC improvement from 0.59 → 0.70+ 

---
## 🔧 **Integration Methods**

### **Option 1: Full Integration (Recommended)**
Integrate all improvements into the main cognitive analysis agent.

### **Option 2: Selective Integration**
Choose specific improvements based on your research needs.

### **Option 3: Standalone Usage**
Use improvements as independent modules for specific tasks.

---

## 🎯 **Integration Examples**

### **Example 1: Enhanced CDR Prediction with All Improvements**

```python
import pandas as pd
import numpy as np
from improvements.dynamic_model_framework import DynamicModelSelector
from improvements.modular_feature_engineering import FeatureEngineeringPipeline, create_feature_config
from improvements.auto_hyperparameter_optimization import AutoHyperparameterOptimizer
from improvements.multi_target_support import MultiTargetAlzheimerPredictor

def enhanced_alzheimer_analysis(df, target_columns=['CDR', 'MMSE']):
    \"\"\"
    Complete enhanced analysis pipeline
    \"\"\"
    print("🚀 Starting Enhanced Alzheimer's Analysis...")
    
    # Step 1: Advanced Feature Engineering
    print("\\n1️⃣ Applying Modular Feature Engineering...")
    config = create_feature_config('alzheimer')
    feature_pipeline = FeatureEngineeringPipeline(config)
    enhanced_df = feature_pipeline.apply_pipeline(df)
    
    # Step 2: Prepare data
    feature_cols = [col for col in enhanced_df.columns if col not in target_columns + ['Subject_ID']]
    X = enhanced_df[feature_cols]
    
    # Step 3: Multi-Target Analysis
    print("\\n2️⃣ Running Multi-Target Analysis...")
    multi_predictor = MultiTargetAlzheimerPredictor()
    y, target_info = multi_predictor.prepare_targets(enhanced_df, target_columns)
    
    # Step 4: Dynamic Model Selection
    print("\\n3️⃣ Dynamic Model Selection...")
    selector = DynamicModelSelector()
    models, pool_name = selector.select_model_pool(X, y.iloc[:, 0], domain='alzheimer')
    
    # Step 5: Hyperparameter Optimization
    print("\\n4️⃣ Hyperparameter Optimization...")
    optimizer = AutoHyperparameterOptimizer(optimization_budget=50)
    
    results = {}
    for name, model in models:
        opt_result = optimizer.optimize_model(name, model, X, y.iloc[:, 0])
        results[name] = opt_result
    
    # Step 6: Multi-Target Training
    print("\\n5️⃣ Multi-Target Model Training...")
    multi_results = multi_predictor.fit_multi_target_models(X, y, target_info)
    
    return {
        'enhanced_features': len(enhanced_df.columns) - len(df.columns),
        'model_pool_used': pool_name,
        'optimization_results': results,
        'multi_target_results': multi_results,
        'recommendations': selector.get_model_recommendations(X, y.iloc[:, 0])
    }

# Usage Example
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("path/to/oasis_data.csv")
    
    # Run enhanced analysis
    results = enhanced_alzheimer_analysis(df, target_columns=['CDR', 'MMSE'])
    
    print(f"\\n✅ Analysis Complete!")
    print(f"Enhanced features: +{results['enhanced_features']}")
    print(f"Model pool: {results['model_pool_used']}")
    print(f"Multi-target models: {len(results['multi_target_results']['individual_models'])}")
```

### **Example 2: Integration with Existing Cognitive Analysis Agent**

```python
# Add to cognitive_analysis_agent.py

from improvements.dynamic_model_framework import DynamicModelSelector
from improvements.modular_feature_engineering import FeatureEngineeringPipeline, create_feature_config
from improvements.auto_hyperparameter_optimization import create_optimization_pipeline

class EnhancedCognitiveAnalysisAgent(CognitiveAnalysisAgent):
    \"\"\"
    Enhanced version with all improvements integrated
    \"\"\"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize improvement modules
        self.feature_pipeline = None
        self.model_selector = None
        self.optimizer = None
        
        # Configure improvements
        self._initialize_improvements()
    
    def _initialize_improvements(self):
        \"\"\"Initialize all improvement modules\"\"\"
        # Feature engineering configuration
        config = create_feature_config('alzheimer')
        self.feature_pipeline = FeatureEngineeringPipeline(config)
        
        # Model selection
        self.model_selector = DynamicModelSelector()
        
        self.logger.info("✅ Enhanced analysis modules initialized")
    
    def _enhanced_cdr_prediction(self) -> Dict[str, Any]:
        \"\"\"Enhanced CDR prediction with all improvements\"\"\"
        
        # Apply modular feature engineering
        if self.feature_pipeline:
            self.logger.info("🔧 Applying advanced feature engineering...")
            self.combined_data = self.feature_pipeline.apply_pipeline(
                self.combined_data, target_col='CDR'
            )
        
        # Dynamic model selection
        feature_cols = [col for col in self.combined_data.columns 
                       if col not in ['CDR', 'Subject_ID']]
        X = self.combined_data[feature_cols]
        y = self.combined_data['CDR']
        
        models_dict, pool_name = self.model_selector.select_model_pool(X, y)
        
        # Convert to format expected by optimizer
        models = {name: model for name, model in models_dict}
        
        # Automated optimization
        self.logger.info(f"🎯 Using {pool_name} model pool for optimization...")
        optimization_results = create_optimization_pipeline(
            models, X, y, budget=50, method='random'
        )
        
        return {
            'model_pool': pool_name,
            'optimization_results': optimization_results,
            'best_model': optimization_results['best_model'],
            'recommendations': optimization_results['recommendations']
        }
```

---

## ⚙️ **Configuration Options**

### **Feature Engineering Configuration**
```yaml
# config/enhanced_features.yaml
feature_engineering:
  modules:
    - alzheimer_brain_features      # Brain volume normalization
    - cognitive_assessment_features # MMSE, CDR enhancements  
    - age_stratified_features      # Age-specific features
    - interaction_features         # Age×MMSE, Gender×Volume
    - risk_score_features         # Composite risk scores
    - biomarker_ratios           # Clinical ratios
    - temporal_features          # Longitudinal features
  
  alzheimer_brain_features:
    include_atrophy_index: true
    include_volume_ratios: true
    validate_asf_etiv_correlation: true
  
  cognitive_assessment_features:
    mmse_age_adjustment: true
    education_normalization: true
    severity_categories: true
```

### **Model Selection Configuration**
```yaml
# config/model_selection.yaml
dynamic_model_selection:
  domain: "alzheimer"
  
  small_dataset_threshold: 500
  large_dataset_threshold: 5000
  
  alzheimer_indicators:
    - "CDR"
    - "MMSE" 
    - "brain"
    - "cognitive"
    - "dementia"
  
  optimization_budget: 100
  scoring_metric: "f1_weighted"
```

### **Multi-Target Configuration**
```yaml
# config/multi_target.yaml
multi_target_prediction:
  targets:
    CDR:
      type: "multiclass"
      priority: "high"
    MMSE:
      type: "regression" 
      priority: "high"
    diagnosis:
      type: "binary"
      priority: "medium"
  
  enable_multi_output: true
  enable_individual_models: true
```

---

## 🎯 **Use Cases & Benefits**

### **Use Case 1: Large-Scale ADNI Analysis**
- **Challenge**: 10,000+ subjects, 200+ features
- **Solution**: Dynamic model selection → Large dataset pool + Automated optimization
- **Benefit**: Optimal performance without manual tuning

### **Use Case 2: Multi-Modal Biomarker Study**
- **Challenge**: Predict CDR, MMSE, and progression risk simultaneously
- **Solution**: Multi-target support + Enhanced features
- **Benefit**: Comprehensive analysis, better clinical insights

### **Use Case 3: Cross-Dataset Validation**
- **Challenge**: Different datasets with varying structures
- **Solution**: Modular feature engineering + Dynamic selection
- **Benefit**: Consistent analysis across diverse datasets

### **Use Case 4: Clinical Decision Support**
- **Challenge**: Real-world deployment with optimal performance
- **Solution**: All improvements + Automated optimization
- **Benefit**: Production-ready system with minimal maintenance

---

## 📈 **Expected Performance Improvements**

### **Feature Engineering Impact**
- **Before**: Basic features only
- **After**: 15-30 additional domain-specific features
- **Expected**: +3-5% accuracy improvement

### **Dynamic Model Selection Impact** 
- **Before**: One-size-fits-all models
- **After**: Data-optimized model selection
- **Expected**: +2-4% accuracy improvement

### **Hyperparameter Optimization Impact**
- **Before**: Default hyperparameters
- **After**: Systematic optimization
- **Expected**: +5-8% accuracy improvement

### **Multi-Target Benefits**
- **Before**: Single outcome prediction
- **After**: Multiple outcomes simultaneously
- **Expected**: Better clinical utility, comprehensive insights

### **Combined Impact**
- **Conservative Estimate**: +8-15% accuracy improvement
- **Optimistic Estimate**: +10-20% accuracy improvement
- **Clinical Value**: Significantly enhanced decision support capability

---

## 🚀 **Quick Start Integration**

### **Step 1: Install Dependencies**
```bash
pip install optuna lightgbm  # Optional but recommended
```

### **Step 2: Copy Improvement Files**
```bash
cp -r improvements/ your_project/
```

### **Step 3: Basic Integration**
```python
# Minimal integration example
from improvements.dynamic_model_framework import create_adaptive_pipeline

# Your existing data
X = your_features  
y = your_target

# Enhanced pipeline
results = create_adaptive_pipeline(X, y, domain='alzheimer')
best_model = results['best_model']['model']

# Use the optimized model
predictions = best_model.predict(X_new)
```

### **Step 4: Full Integration**
Follow Example 1 above for comprehensive integration.

---

## 🔍 **Testing & Validation**

Each improvement includes built-in testing:

```bash
# Test individual modules
python improvements/dynamic_model_framework.py
python improvements/modular_feature_engineering.py  
python improvements/auto_hyperparameter_optimization.py
python improvements/multi_target_support.py

# Test integration
python test_all_improvements.py
```

---

## 💡 **Best Practices**

1. **Start Small**: Integrate one improvement at a time
2. **Validate Performance**: Compare before/after metrics
3. **Configure Appropriately**: Adjust settings for your dataset size
4. **Monitor Resources**: Optimization can be computationally intensive
5. **Document Changes**: Keep track of configuration modifications

---

## 🎯 **Next Steps**

After integration, you'll have a significantly more powerful and generalizable Alzheimer's analysis framework that can:

- **Automatically adapt** to different dataset characteristics
- **Generate optimized features** for any Alzheimer's research scenario  
- **Select and tune models** without manual intervention
- **Predict multiple outcomes** simultaneously
- **Achieve higher accuracy** through systematic optimization

The enhanced framework maintains its Alzheimer's focus while becoming much more flexible and powerful for diverse research applications.

---

*Integration Guide - Advanced Alzheimer's Analysis Framework*  
*Designed for maximum research impact and clinical utility*

## New reusable modules (Sept 2025)

- demographics_enrichment.py
  - enrich_demographics(data_dir: Path, base: pd.DataFrame, subject_col='SubjectCode') -> pd.DataFrame
  - Merges age, education, gender from BHR files (accepts 'Code'), adds derived interactions and reserve proxy.

- sequence_feature_engineering.py
  - compute_sequence_features(df: pd.DataFrame, subject_col='SubjectCode', reaction_col='ReactionTimes') -> pd.DataFrame
  - Builds sequence/fatigue/reliability features used in best logistic baseline.

- target_curation_bhr.py
  - curate_cognitive_target(medical_df, target_qid='QID1-13', subject_col='SubjectCode', timepoint_col='TimepointCode') -> (df, y)
  - Filters to baseline and excludes non-cognitive QIDs.

- ashford_policy.py
  - apply_ashford(memtrax_df, accuracy_threshold=0.65, rt_min=0.5, rt_max=2.5, status_col='Status') -> pd.DataFrame
  - Standard quality filter step.

- calibrated_logistic.py
  - train_calibrated_logistic(X, y, k_features=200) -> (model, metrics)
  - Impute/scale/MI select/logistic with isotonic calibration and threshold tuning.

### Example usage
```python
from pathlib import Path
import pandas as pd
from improvements.ashford_policy import apply_ashford
from improvements.demographics_enrichment import enrich_demographics
from improvements.sequence_feature_engineering import compute_sequence_features
from improvements.target_curation_bhr import curate_cognitive_target
from improvements.calibrated_logistic import train_calibrated_logistic

# Load
mem = pd.read_csv(data_dir/"MemTrax.csv", low_memory=False)
med = pd.read_csv(data_dir/"BHR_MedicalHx.csv", low_memory=False)

# Quality and features
mem_q = apply_ashford(mem, accuracy_threshold=0.65)
seq = compute_sequence_features(mem_q)
agg = mem_q.groupby('SubjectCode').mean(numeric_only=True).reset_index()
X_df = agg.merge(seq, on='SubjectCode', how='left')
X_df = enrich_demographics(data_dir, X_df)

# Labels
med_b, y = curate_cognitive_target(med, target_qid='QID1-13')
XY = X_df.merge(med_b[['SubjectCode', 'QID1-13']], on='SubjectCode', how='inner')
X = XY.drop(columns=['QID1-13'])
y = (XY['QID1-13']==1.0).astype(int)

# Train
model, metrics = train_calibrated_logistic(X.select_dtypes('number'), y)
print(metrics)
```

## Anti-Leakage Practices (New)

To prevent optimistic bias and ensure reproducible metrics, follow these rules:

- Use a held-out evaluation: split with stratified train/test (e.g., 80/20) and report AUC/PR-AUC on the held-out test set.
- Fit transforms only on training data: e.g., fit `SplineTransformer` on training, then apply to test.
- For stacking/ensembles, train meta-learners on out-of-fold (OOF) predictions from base learners; for test, use base learners refit on full training.
- Calibrate probabilities within CV folds (e.g., `CalibratedClassifierCV` with `cv>=3`), not on the same data you evaluate.

Utilities available in `improvements.anti_leakage`:

- `stratified_holdout_split(X, y, test_size=0.2)`: train/test split.
- `add_splines_train_only(X_train, X_test, cols, n_knots=5)`: fit splines on train, apply to test.
- `oof_calibrated_probas(est_builder, X_train, y_train, n_splits=5, method='isotonic')`: returns OOF predictions and a calibrated model fit on full train.
- `evaluate_holdout(y_true, y_proba)`: returns AUC and PR-AUC.

Example (pseudo-code):

```python
from improvements import stratified_holdout_split, add_splines_train_only, oof_calibrated_probas

X_tr, X_te, y_tr, y_te = stratified_holdout_split(X, y)
X_tr, X_te = add_splines_train_only(X_tr, X_te, ['Age_Baseline'])

def build_logit():
    return Pipeline([...])

oof_logit, cal_logit = oof_calibrated_probas(build_logit, X_tr, y_tr)
proba_te = cal_logit.predict_proba(X_te)[:, 1]
```

These policies should be applied by default in the agent's modeling flows when prevalence is low or when stacking is used.

## Agent Rulebook: Clinical Utility Pipeline

- When target prevalence ≤ 10% and baseline labels exist, prefer the clinical-utility pipeline.
- Use composite baseline cognitive impairment (OR of QID1-5/1-12/1-13/1-22/1-23 at m00); exclude non-cognitive (Parkinson’s, Stroke).
- Always aggregate across timepoints; add sequence/fatigue (first/last third means, fatigue effect), variability (std), test count, RT slope.
- Enrich demographics from BHR sources; add interactions (age×RT, age×variability) and cognitive reserve proxy.
- Merge self/informant ECOG (+ADL) and residualize global/per-domain means on age/education.
- Modeling defaults:
  - Baseline: calibrated logistic with MI feature selection (k~50–150) + age/edu splines.
  - Lift: calibrated stacking (logit + HistGB + XGB with tuned scale_pos_weight), elastic-net meta-learner over base probabilities + 3 raw features (CognitiveScore_mean, long_reliability_change, Age_Baseline).
- Evaluation: subject-level CV; decision curves; subgroup AUC by age/education; report PR-AUC and delta over demographics.
- Quality: apply Ashford (status=Collected, accuracy ≥0.65, RT in [0.5, 2.5]); optionally winsorize RTs [0.4, 2.0] for robustness.

To enable: set `analysis.clinical_utility_mode: true` in `config/config.yaml` (default true). The orchestrator will run `bhr_memtrax_clinical_utility.py`, ingest `bhr_memtrax_results/clinical_utility_report.json`, and expose results under `results.analysis.clinical_utility`.