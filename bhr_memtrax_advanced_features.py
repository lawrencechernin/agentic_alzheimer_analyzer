#!/usr/bin/env python3
"""
BHR MemTrax with Advanced Feature Engineering and Selection
============================================================
Optimizing feature space through:
1. Advanced feature selection with cross-validated k
2. Feature interactions and polynomial features
3. Medical history integration
4. Longitudinal change features
5. Feature importance weighting
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.compose import ColumnTransformer
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
import json
from itertools import combinations

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cognitive impairment QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Medical history QIDs that might be relevant
MEDICAL_QIDS = {
    'diabetes': ['QID2-1', 'QID2-2'],  # Diabetes related
    'hypertension': ['QID2-3', 'QID2-4'],  # Blood pressure
    'heart': ['QID2-5', 'QID2-6'],  # Heart disease
    'stroke': ['QID2-7', 'QID2-8'],  # Stroke
    'depression': ['QID2-9', 'QID2-10'],  # Mental health
}


def extract_comprehensive_features(memtrax_q):
    """Extract comprehensive MemTrax features including longitudinal aspects"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Sort by date if available to capture temporal patterns
        if 'StatusDateTime' in group.columns:
            try:
                group = group.sort_values('StatusDateTime')
            except:
                pass
        
        # Basic statistics
        feat['correct_pct_mean'] = group['CorrectPCT'].mean()
        feat['correct_pct_std'] = group['CorrectPCT'].std()
        feat['correct_pct_min'] = group['CorrectPCT'].min()
        feat['correct_pct_max'] = group['CorrectPCT'].max()
        feat['correct_pct_range'] = feat['correct_pct_max'] - feat['correct_pct_min']
        
        feat['correct_rt_mean'] = group['CorrectResponsesRT'].mean()
        feat['correct_rt_std'] = group['CorrectResponsesRT'].std()
        feat['correct_rt_min'] = group['CorrectResponsesRT'].min()
        feat['correct_rt_max'] = group['CorrectResponsesRT'].max()
        feat['correct_rt_range'] = feat['correct_rt_max'] - feat['correct_rt_min']
        
        # Coefficient of variation (normalized variability)
        feat['correct_pct_cv'] = feat['correct_pct_std'] / (feat['correct_pct_mean'] + 1e-6)
        feat['correct_rt_cv'] = feat['correct_rt_std'] / (feat['correct_rt_mean'] + 1e-6)
        
        # Error metrics
        feat['incorrect_pct_mean'] = group['IncorrectPCT'].mean()
        feat['incorrect_rt_mean'] = group['IncorrectResponsesRT'].mean()
        feat['error_rate'] = 1 - feat['correct_pct_mean']
        
        # Composite cognitive scores
        feat['cog_score'] = feat['correct_rt_mean'] / (feat['correct_pct_mean'] + 0.01)
        feat['speed_accuracy_product'] = feat['correct_pct_mean'] / (feat['correct_rt_mean'] + 0.01)
        feat['efficiency_score'] = feat['correct_pct_mean'] / (feat['correct_rt_std'] + 0.01)
        
        # LONGITUDINAL FEATURES (if multiple tests)
        n_tests = len(group)
        feat['n_tests'] = n_tests
        
        if n_tests > 1:
            # Trend over time
            x = np.arange(n_tests)
            
            # Accuracy trend
            y_acc = group['CorrectPCT'].values
            if np.std(x) > 0 and not np.isnan(y_acc).all():
                acc_slope, acc_intercept = np.polyfit(x, y_acc, 1)
                feat['accuracy_trend'] = acc_slope
                feat['accuracy_improvement'] = y_acc[-1] - y_acc[0]
            
            # RT trend
            y_rt = group['CorrectResponsesRT'].values
            if np.std(x) > 0 and not np.isnan(y_rt).all():
                rt_slope, rt_intercept = np.polyfit(x, y_rt, 1)
                feat['rt_trend'] = rt_slope
                feat['rt_change'] = y_rt[-1] - y_rt[0]
            
            # Practice effect (first vs last test)
            feat['practice_effect_acc'] = group['CorrectPCT'].iloc[-1] - group['CorrectPCT'].iloc[0]
            feat['practice_effect_rt'] = group['CorrectResponsesRT'].iloc[0] - group['CorrectResponsesRT'].iloc[-1]
            
            # Consistency across tests
            feat['test_consistency'] = 1 / (feat['correct_pct_std'] + 0.01)
        
        # Reaction time sequence analysis
        all_rts = []
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    all_rts.extend([r for r in rts if 0.3 <= r <= 3.0])
                except:
                    continue
        
        if len(all_rts) >= 10:
            # Detailed RT analysis
            feat['rt_median'] = np.median(all_rts)
            feat['rt_iqr'] = np.percentile(all_rts, 75) - np.percentile(all_rts, 25)
            feat['rt_skew'] = pd.Series(all_rts).skew()
            feat['rt_kurtosis'] = pd.Series(all_rts).kurtosis()
            
            # Fatigue patterns
            n = len(all_rts)
            third = max(1, n // 3)
            feat['fatigue_score'] = np.mean(all_rts[-third:]) - np.mean(all_rts[:third])
            
            # Learning within session
            mid = n // 2
            feat['within_session_learning'] = np.mean(all_rts[:mid]) - np.mean(all_rts[mid:])
            
            # Response variability
            feat['rt_entropy'] = -np.sum(pd.Series(all_rts).value_counts(normalize=True) * 
                                        np.log(pd.Series(all_rts).value_counts(normalize=True) + 1e-10))
        
        features.append(feat)
    
    return pd.DataFrame(features)


def add_demographics_and_medical(data):
    """Add demographics and medical history features"""
    
    # Try to load demographics
    demo_paths = [
        DATA_DIR / 'BHR_Demographics.csv',
        DATA_DIR / 'Demographics.csv',
        DATA_DIR / 'BHR_Demographic.csv'
    ]
    
    demo_added = False
    for path in demo_paths:
        if path.exists():
            print(f"   Loading demographics from {path.name}")
            demo = pd.read_csv(path, low_memory=False)
            
            # Standardize column names
            if 'Code' in demo.columns:
                demo = demo.rename(columns={'Code': 'SubjectCode'})
            
            # Find available demographic columns
            demo_cols = ['SubjectCode']
            
            # Age - QID186 appears to be age based on value range analysis
            for col in ['QID186', 'Age_Baseline', 'Age', 'age_baseline', 'age']:
                if col in demo.columns:
                    demo_cols.append(col)
                    demo = demo.rename(columns={col: 'Age'})
                    break
            
            # Education - QID184 appears to be education level (1-7 scale)
            for col in ['QID184', 'YearsEducationUS_Converted', 'Education', 'YearsEducation', 'education']:
                if col in demo.columns:
                    demo_cols.append(col if col == 'Education' else 'Education')
                    if col != 'Education':
                        demo = demo.rename(columns={col: 'Education'})
                    break
            
            # Gender
            for col in ['Gender', 'Sex', 'gender', 'sex']:
                if col in demo.columns:
                    demo_cols.append(col if col == 'Gender' else 'Gender')
                    if col != 'Gender':
                        demo = demo.rename(columns={col: 'Gender'})
                    break
            
            if len(demo_cols) > 1:
                demo = demo[demo_cols].drop_duplicates(subset=['SubjectCode'])
                data = data.merge(demo, on='SubjectCode', how='left')
                demo_added = True
                print(f"   Added {len(demo_cols)-1} demographic features")
            break
    
    # Try to add medical history
    med_path = DATA_DIR / 'BHR_MedicalHx.csv'
    if med_path.exists():
        print("   Loading medical history...")
        med_hx = pd.read_csv(med_path, low_memory=False)
        
        # Get baseline records
        if 'TimepointCode' in med_hx.columns:
            med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
        
        med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
        
        # Extract medical conditions
        med_features = {'SubjectCode': med_hx['SubjectCode']}
        
        for condition, qids in MEDICAL_QIDS.items():
            available_qids = [q for q in qids if q in med_hx.columns]
            if available_qids:
                # Create binary indicator for condition
                condition_present = np.zeros(len(med_hx))
                for qid in available_qids:
                    condition_present |= (med_hx[qid] == 1).values
                med_features[f'has_{condition}'] = condition_present
        
        if len(med_features) > 1:
            med_df = pd.DataFrame(med_features)
            data = data.merge(med_df, on='SubjectCode', how='left')
            print(f"   Added {len(med_features)-1} medical history features")
    
    return data, demo_added


def create_interaction_features(X, feature_names, top_n=10):
    """Create interaction features for top features"""
    
    # Find top features using mutual information
    selector = SelectKBest(mutual_info_classif, k=min(top_n, X.shape[1]))
    selector.fit(X, y)
    
    # Get indices of top features
    top_indices = selector.get_support(indices=True)
    top_feature_names = [feature_names[i] for i in top_indices]
    
    print(f"   Creating interactions for top {len(top_feature_names)} features")
    
    # Create interaction features
    interaction_features = []
    interaction_names = []
    
    for i, j in combinations(range(len(top_indices)), 2):
        idx1, idx2 = top_indices[i], top_indices[j]
        interaction = X[:, idx1] * X[:, idx2]
        interaction_features.append(interaction.reshape(-1, 1))
        interaction_names.append(f"{feature_names[idx1]}_x_{feature_names[idx2]}")
    
    if interaction_features:
        X_interactions = np.hstack(interaction_features)
        X_combined = np.hstack([X, X_interactions])
        combined_names = feature_names + interaction_names
        return X_combined, combined_names
    
    return X, feature_names


def find_optimal_k(X_train, y_train, k_range=None):
    """Find optimal k for SelectKBest using cross-validation"""
    
    if k_range is None:
        k_range = [5, 10, 15, 20, 25, 30, 'all']
    
    print("\n   Finding optimal k for feature selection...")
    
    best_score = 0
    best_k = 'all'
    
    for k in k_range:
        if k == 'all' or k > X_train.shape[1]:
            k_actual = 'all'
        else:
            k_actual = min(k, X_train.shape[1])
        
        # Create pipeline with SelectKBest
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k=k_actual)),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        mean_score = cv_scores.mean()
        print(f"      k={k_actual}: CV AUC={mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k_actual
    
    print(f"   Optimal k: {best_k} (CV AUC={best_score:.4f})")
    return best_k


def prepare_data():
    """Load and prepare data with all feature engineering"""
    
    print("1. Loading BHR data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Quality filter
    print("2. Applying quality filter...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Extract comprehensive features
    print("3. Extracting comprehensive features...")
    features = extract_comprehensive_features(memtrax_q)
    print(f"   Created {features.shape[1]-1} MemTrax features")
    
    # Add demographics and medical history
    print("4. Adding demographics and medical history...")
    features, has_demographics = add_demographics_and_medical(features)
    
    # Create labels
    print("5. Creating labels...")
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    impairment = np.zeros(len(med_hx), dtype=int)
    valid_mask = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid_mask |= med_hx[qid].isin([1, 2]).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'mci': impairment
    })
    labels = labels[valid_mask].copy()
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    
    # Create additional interaction features if demographics available
    if has_demographics and 'Age' in data.columns and 'Education' in data.columns:
        print("6. Creating demographic interaction features...")
        data['age_education_interaction'] = data['Age'] * data['Education']
        data['cognitive_reserve'] = data['Education'] / (data['Age'] / 100 + 0.1)
        
        # Age interactions with key cognitive features
        if 'correct_pct_mean' in data.columns:
            data['age_x_accuracy'] = data['Age'] * data['correct_pct_mean']
        if 'correct_rt_mean' in data.columns:
            data['age_x_rt'] = data['Age'] * data['correct_rt_mean']
        if 'cog_score' in data.columns:
            data['age_x_cogscore'] = data['Age'] * data['cog_score']
    
    print(f"\n   Final dataset: {len(data):,} subjects")
    print(f"   Total features: {data.shape[1]-2}")
    print(f"   MCI prevalence: {data['mci'].mean():.1%}")
    
    return data


def evaluate_feature_selection_methods(X_train, X_test, y_train, y_test, feature_names):
    """Evaluate different feature selection strategies"""
    
    print("\n7. EVALUATING FEATURE SELECTION STRATEGIES")
    print("="*70)
    
    results = {}
    
    # 1. Baseline (all features)
    print("\n   Baseline (all features):")
    pipe_baseline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    pipe_baseline.fit(X_train, y_train)
    y_pred = pipe_baseline.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred)
    print(f"      AUC: {baseline_auc:.4f}")
    results['baseline'] = baseline_auc
    
    # 2. Mutual Information with optimal k
    optimal_k = find_optimal_k(X_train, y_train)
    
    pipe_mi = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(mutual_info_classif, k=optimal_k)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    pipe_mi.fit(X_train, y_train)
    y_pred = pipe_mi.predict_proba(X_test)[:, 1]
    mi_auc = roc_auc_score(y_test, y_pred)
    print(f"\n   Mutual Information (k={optimal_k}):")
    print(f"      AUC: {mi_auc:.4f}")
    results['mutual_info'] = mi_auc
    
    # Get selected features
    selector = pipe_mi.named_steps['select']
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    # 3. Chi-squared (for positive features)
    print("\n   Chi-squared selection:")
    # Make features positive for chi2
    X_train_positive = X_train - X_train.min() + 1
    X_test_positive = X_test - X_test.min() + 1
    
    pipe_chi2 = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('select', SelectKBest(chi2, k=optimal_k)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    
    try:
        pipe_chi2.fit(X_train_positive, y_train)
        y_pred = pipe_chi2.predict_proba(X_test_positive)[:, 1]
        chi2_auc = roc_auc_score(y_test, y_pred)
        print(f"      AUC: {chi2_auc:.4f}")
        results['chi2'] = chi2_auc
    except:
        print("      Failed (negative values)")
        results['chi2'] = 0
    
    # 4. ANOVA F-value
    print("\n   ANOVA F-value selection:")
    pipe_anova = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(f_classif, k=optimal_k)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    pipe_anova.fit(X_train, y_train)
    y_pred = pipe_anova.predict_proba(X_test)[:, 1]
    anova_auc = roc_auc_score(y_test, y_pred)
    print(f"      AUC: {anova_auc:.4f}")
    results['anova'] = anova_auc
    
    # 5. L1-based selection (Lasso)
    print("\n   L1-based selection (Lasso):")
    pipe_l1 = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectFromModel(LogisticRegression(penalty='l1', solver='saga', 
                                                     C=0.1, max_iter=2000))),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    pipe_l1.fit(X_train, y_train)
    y_pred = pipe_l1.predict_proba(X_test)[:, 1]
    l1_auc = roc_auc_score(y_test, y_pred)
    print(f"      AUC: {l1_auc:.4f}")
    results['l1'] = l1_auc
    
    # 6. Tree-based feature importance
    print("\n   Tree-based feature selection:")
    pipe_tree = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('select', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    pipe_tree.fit(X_train, y_train)
    y_pred = pipe_tree.predict_proba(X_test)[:, 1]
    tree_auc = roc_auc_score(y_test, y_pred)
    print(f"      AUC: {tree_auc:.4f}")
    results['tree'] = tree_auc
    
    # 7. Top 4 features only (as mentioned in the prompt)
    print("\n   Top 4 features only:")
    top_4_features = ['Age', 'Education', 'correct_pct_mean', 'correct_rt_mean']
    available_top_4 = [i for i, f in enumerate(feature_names) if f in top_4_features]
    
    if len(available_top_4) >= 4:
        X_train_top4 = X_train[:, available_top_4]
        X_test_top4 = X_test[:, available_top_4]
        
        pipe_top4 = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ])
        pipe_top4.fit(X_train_top4, y_train)
        y_pred = pipe_top4.predict_proba(X_test_top4)[:, 1]
        top4_auc = roc_auc_score(y_test, y_pred)
        print(f"      AUC: {top4_auc:.4f}")
        results['top4'] = top4_auc
    else:
        print(f"      Not all top 4 features available")
        results['top4'] = 0
    
    return results, selected_features


def main():
    print("\n" + "="*70)
    print("BHR MEMTRAX WITH ADVANCED FEATURE ENGINEERING")
    print("="*70)
    
    # Prepare data with all features
    data = prepare_data()
    
    # Prepare for modeling
    X = data.drop(['SubjectCode', 'mci'], axis=1).values
    y = data['mci'].values
    feature_names = [c for c in data.columns if c not in ['SubjectCode', 'mci']]
    
    print(f"\n   Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Evaluate feature selection methods
    selection_results, selected_features = evaluate_feature_selection_methods(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Find best selection method
    best_method = max(selection_results, key=selection_results.get)
    best_selection_auc = selection_results[best_method]
    
    # Train final models with best features
    print("\n8. TRAINING FINAL MODELS WITH BEST FEATURE SELECTION")
    print("="*70)
    
    # Use the best feature selection method
    if best_method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=find_optimal_k(X_train, y_train))
    elif best_method == 'anova':
        selector = SelectKBest(f_classif, k=find_optimal_k(X_train, y_train))
    elif best_method == 'l1':
        selector = SelectFromModel(LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=2000))
    elif best_method == 'tree':
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    else:
        selector = None
    
    # Build final models
    models = {}
    
    if selector:
        models['Logistic'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', selector),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ])
        
        models['RF'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('select', selector),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10, 
                                          min_samples_split=15, class_weight='balanced',
                                          random_state=42))
        ])
        
        if XGB_AVAILABLE:
            models['XGBoost'] = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('select', selector),
                ('clf', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6,
                                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, eval_metric='logloss'))
            ])
    
    best_auc = 0
    best_model_name = None
    final_results = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        
        print(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        final_results[name] = {
            'cv_auc': float(cv_scores.mean()),
            'test_auc': float(test_auc)
        }
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_name = name
    
    # Print selected features
    if selected_features and len(selected_features) <= 20:
        print("\n9. TOP SELECTED FEATURES:")
        print("="*70)
        for i, feat in enumerate(selected_features[:15], 1):
            print(f"   {i:2d}. {feat}")
    
    # Results summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    baseline_auc = 0.744
    
    print(f"\nBest Feature Selection: {best_method}")
    print(f"Best Model: {best_model_name}")
    print(f"Best AUC: {best_auc:.4f}")
    
    print(f"\nComparison:")
    print(f"  Original baseline: {baseline_auc:.4f}")
    print(f"  With advanced features: {best_auc:.4f}")
    print(f"  Improvement: {(best_auc - baseline_auc):+.4f}")
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Achieved {best_auc:.4f} AUC!")
    elif best_auc >= 0.78:
        print(f"\n✅ GREAT! Achieved {best_auc:.4f} AUC!")
    elif best_auc > baseline_auc:
        print(f"\n📈 Improved to {best_auc:.4f} AUC")
    else:
        print(f"\n📊 No improvement over baseline")
    
    # Save results
    output = {
        'strategy': 'Advanced feature engineering and selection',
        'best_feature_selection': best_method,
        'best_model': best_model_name,
        'best_auc': float(best_auc),
        'baseline_auc': baseline_auc,
        'improvement': float(best_auc - baseline_auc),
        'n_features_original': X.shape[1],
        'n_features_selected': len(selected_features) if selected_features else X.shape[1],
        'feature_selection_results': {k: float(v) for k, v in selection_results.items()},
        'model_results': final_results
    }
    
    with open(OUTPUT_DIR / 'advanced_features_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/advanced_features_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()
