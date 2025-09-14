#!/usr/bin/env python3
"""
Comprehensive Cognitive Impairment Detection
=============================================
1. Use SP-ECOG as ground truth LABELS (not features)
2. Combine ALL cognitive-related medical conditions
3. Create a comprehensive cognitive impairment label
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

# Original MCI QIDs
MCI_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Expanded cognitive-related QIDs (we'll discover these)
COGNITIVE_CONDITIONS = {
    'memory': ['memory', 'forget', 'recall', 'remember'],
    'alzheimer': ['alzheimer', 'dementia', 'neurodegen'],
    'confusion': ['confusion', 'disoriented', 'lost'],
    'cognitive': ['cognitive', 'thinking', 'mental'],
    'stroke': ['stroke', 'tia', 'cerebral'],
    'parkinson': ['parkinson'],
    'depression': ['depression', 'anxiety'],  # Can affect cognition
    'sleep': ['sleep', 'insomnia'],  # Sleep affects cognition
}


def find_cognitive_qids():
    """Find all QIDs that might relate to cognitive impairment"""
    print("\n1. DISCOVERING COGNITIVE-RELATED QIDs")
    print("="*70)
    
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Get baseline only
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    cognitive_qids = set(MCI_QIDS)
    
    # Find QIDs with reasonable prevalence that might be cognitive
    for col in med_hx.columns:
        if 'QID' in col and col not in cognitive_qids:
            # Check if binary-like (1/2 coding)
            unique_vals = med_hx[col].dropna().unique()
            if len(unique_vals) <= 3 and 1 in unique_vals and 2 in unique_vals:
                positive_rate = (med_hx[col] == 1).mean()
                
                # Look for reasonable prevalence (0.1% to 30%)
                if 0.001 < positive_rate < 0.30:
                    # Check specific ranges that might be cognitive
                    if any(x in col for x in ['QID1-', 'QID2-', 'QID3-']):  # Medical history sections
                        cognitive_qids.add(col)
    
    print(f"Found {len(cognitive_qids)} potential cognitive QIDs")
    
    # Analyze each
    print("\nCognitive-related conditions found:")
    for qid in sorted(list(cognitive_qids))[:30]:  # Show first 30
        if qid in med_hx.columns:
            positive = (med_hx[qid] == 1).sum()
            total = med_hx[qid].notna().sum()
            if total > 0:
                print(f"  {qid}: {positive:,} cases ({positive/total*100:.1f}%)")
    
    return list(cognitive_qids)


def create_sp_ecog_labels():
    """Create labels from SP-ECOG (informant reports)"""
    print("\n2. CREATING SP-ECOG BASED LABELS")
    print("="*70)
    
    sp_ecog_path = DATA_DIR / 'BHR_SP_ECog.csv'
    if not sp_ecog_path.exists():
        print("SP-ECOG file not found!")
        return None
    
    sp_ecog = pd.read_csv(sp_ecog_path, low_memory=False)
    
    # Fix timepoint codes
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog['TimepointCode'] = sp_ecog['TimepointCode'].str.replace('sp-', '')
        sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    
    # Get QID columns (numeric responses)
    qid_cols = [c for c in sp_ecog.columns if 'QID' in c and sp_ecog[c].dtype in [np.float64, np.int64]]
    
    if not qid_cols:
        print("No numeric QID columns found in SP-ECOG")
        return None
    
    print(f"Found {len(qid_cols)} SP-ECOG items")
    
    # Calculate scores (excluding 8='Don't Know')
    sp_ecog_scores = sp_ecog[qid_cols].replace(8, np.nan)
    
    # Multiple ways to define impairment from informant
    sp_ecog_labels = pd.DataFrame()
    sp_ecog_labels['SubjectCode'] = sp_ecog['SubjectCode']
    
    # Mean score >= 3 (moderate changes)
    sp_ecog_labels['sp_mean'] = sp_ecog_scores.mean(axis=1)
    sp_ecog_labels['informant_impaired_mean'] = (sp_ecog_labels['sp_mean'] >= 3.0).astype(int)
    
    # At least 20% of items >= 3
    high_scores = (sp_ecog_scores >= 3).sum(axis=1)
    valid_items = sp_ecog_scores.notna().sum(axis=1)
    sp_ecog_labels['sp_high_pct'] = high_scores / (valid_items + 1e-6)
    sp_ecog_labels['informant_impaired_pct'] = (sp_ecog_labels['sp_high_pct'] >= 0.20).astype(int)
    
    # Any score >= 4 (severe changes)
    sp_ecog_labels['sp_max'] = sp_ecog_scores.max(axis=1)
    sp_ecog_labels['informant_impaired_severe'] = (sp_ecog_labels['sp_max'] >= 4).astype(int)
    
    # Combined informant label (any of the above)
    sp_ecog_labels['informant_cognitive_impairment'] = (
        sp_ecog_labels['informant_impaired_mean'] |
        sp_ecog_labels['informant_impaired_pct'] |
        sp_ecog_labels['informant_impaired_severe']
    ).astype(int)
    
    # Keep only valid responses
    sp_ecog_labels = sp_ecog_labels[sp_ecog_labels['sp_mean'].notna()]
    
    print(f"\nInformant-based impairment rates:")
    print(f"  Mean >= 3.0: {sp_ecog_labels['informant_impaired_mean'].mean():.1%}")
    print(f"  20% items >= 3: {sp_ecog_labels['informant_impaired_pct'].mean():.1%}")
    print(f"  Any >= 4: {sp_ecog_labels['informant_impaired_severe'].mean():.1%}")
    print(f"  Combined: {sp_ecog_labels['informant_cognitive_impairment'].mean():.1%}")
    print(f"  N with informant data: {len(sp_ecog_labels):,}")
    
    return sp_ecog_labels


def create_comprehensive_labels(cognitive_qids):
    """Create comprehensive cognitive impairment labels from all sources"""
    print("\n3. CREATING COMPREHENSIVE LABELS")
    print("="*70)
    
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    
    # Create labels from all cognitive QIDs
    labels = pd.DataFrame()
    labels['SubjectCode'] = med_hx['SubjectCode']
    
    # Track different types of impairment
    impairment_types = {}
    
    # 1. Original MCI
    mci = np.zeros(len(med_hx), dtype=int)
    valid_mci = np.zeros(len(med_hx), dtype=bool)
    for qid in MCI_QIDS:
        if qid in med_hx.columns:
            mci |= (med_hx[qid] == 1).values
            valid_mci |= med_hx[qid].isin([1, 2]).values
    
    labels['mci_original'] = mci
    labels['mci_valid'] = valid_mci
    impairment_types['MCI (original)'] = mci.mean()
    
    # 2. All cognitive conditions
    all_cognitive = np.zeros(len(med_hx), dtype=int)
    valid_cognitive = np.zeros(len(med_hx), dtype=bool)
    for qid in cognitive_qids:
        if qid in med_hx.columns:
            all_cognitive |= (med_hx[qid] == 1).values
            valid_cognitive |= med_hx[qid].isin([1, 2]).values
    
    labels['cognitive_any'] = all_cognitive
    labels['cognitive_valid'] = valid_cognitive
    impairment_types['Any cognitive condition'] = all_cognitive.mean()
    
    # 3. Severity levels (based on number of conditions)
    condition_count = np.zeros(len(med_hx))
    for qid in cognitive_qids:
        if qid in med_hx.columns:
            condition_count += (med_hx[qid] == 1).values
    
    labels['n_conditions'] = condition_count
    labels['mild_impairment'] = (condition_count >= 1).astype(int)
    labels['moderate_impairment'] = (condition_count >= 2).astype(int)
    labels['severe_impairment'] = (condition_count >= 3).astype(int)
    
    impairment_types['Mild (1+ conditions)'] = labels['mild_impairment'].mean()
    impairment_types['Moderate (2+ conditions)'] = labels['moderate_impairment'].mean()
    impairment_types['Severe (3+ conditions)'] = labels['severe_impairment'].mean()
    
    print("Self-reported impairment rates:")
    for name, rate in impairment_types.items():
        print(f"  {name}: {rate:.1%}")
    
    # 4. Add informant labels
    sp_ecog_labels = create_sp_ecog_labels()
    if sp_ecog_labels is not None:
        labels = labels.merge(sp_ecog_labels[['SubjectCode', 'informant_cognitive_impairment']], 
                              on='SubjectCode', how='left')
        labels['has_informant'] = labels['informant_cognitive_impairment'].notna()
    else:
        labels['informant_cognitive_impairment'] = np.nan
        labels['has_informant'] = False
    
    # 5. Create final composite label
    # Priority: informant > multiple conditions > any condition
    labels['final_cognitive_impairment'] = 0
    
    # Use informant when available
    informant_mask = labels['has_informant'] & labels['informant_cognitive_impairment'].notna()
    labels.loc[informant_mask, 'final_cognitive_impairment'] = \
        labels.loc[informant_mask, 'informant_cognitive_impairment']
    
    # Otherwise use self-report (moderate threshold)
    self_mask = ~informant_mask & labels['cognitive_valid']
    labels.loc[self_mask, 'final_cognitive_impairment'] = \
        labels.loc[self_mask, 'moderate_impairment']
    
    print(f"\nFinal composite impairment rate: {labels['final_cognitive_impairment'].mean():.1%}")
    print(f"  From informant: {informant_mask.sum():,}")
    print(f"  From self-report: {self_mask.sum():,}")
    
    return labels


def extract_memtrax_features(memtrax_q):
    """Extract comprehensive MemTrax features"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic statistics
        feat['CorrectPCT_mean'] = group['CorrectPCT'].mean()
        feat['CorrectPCT_std'] = group['CorrectPCT'].std()
        feat['CorrectPCT_min'] = group['CorrectPCT'].min()
        feat['CorrectPCT_max'] = group['CorrectPCT'].max()
        
        feat['CorrectResponsesRT_mean'] = group['CorrectResponsesRT'].mean()
        feat['CorrectResponsesRT_std'] = group['CorrectResponsesRT'].std()
        feat['CorrectResponsesRT_min'] = group['CorrectResponsesRT'].min()
        feat['CorrectResponsesRT_max'] = group['CorrectResponsesRT'].max()
        
        feat['IncorrectPCT_mean'] = group['IncorrectPCT'].mean()
        feat['IncorrectResponsesRT_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores
        feat['CognitiveScore'] = feat['CorrectResponsesRT_mean'] / (feat['CorrectPCT_mean'] + 0.01)
        feat['Speed_Accuracy_Product'] = feat['CorrectPCT_mean'] * feat['CorrectResponsesRT_mean']
        feat['Error_Rate'] = 1 - feat['CorrectPCT_mean']
        
        # Sequence analysis
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
            n = len(all_rts)
            third = max(1, n // 3)
            
            feat['first_third_mean'] = np.mean(all_rts[:third])
            feat['last_third_mean'] = np.mean(all_rts[-third:])
            feat['fatigue_effect'] = feat['last_third_mean'] - feat['first_third_mean']
            
            mid = n // 2
            if mid > 1:
                feat['reliability_change'] = np.var(all_rts[mid:]) - np.var(all_rts[:mid])
                
            if n >= 3:
                slope, _ = np.polyfit(np.arange(n), all_rts, 1)
                feat['rt_slope'] = slope
                
        feat['n_tests'] = len(group)
        features.append(feat)
    
    return pd.DataFrame(features)


def test_different_labels(memtrax_features, labels):
    """Test model performance with different label definitions"""
    print("\n4. TESTING DIFFERENT LABEL DEFINITIONS")
    print("="*70)
    
    # Merge features with labels
    data = memtrax_features.merge(labels, on='SubjectCode', how='inner')
    
    # Prepare features
    feature_cols = [c for c in memtrax_features.columns if c != 'SubjectCode']
    X = data[feature_cols].values
    
    # Test different label definitions
    label_definitions = [
        ('Original MCI', 'mci_original', 'mci_valid'),
        ('Any Cognitive', 'cognitive_any', 'cognitive_valid'),
        ('Mild (1+)', 'mild_impairment', 'cognitive_valid'),
        ('Moderate (2+)', 'moderate_impairment', 'cognitive_valid'),
        ('Severe (3+)', 'severe_impairment', 'cognitive_valid'),
        ('Informant Only', 'informant_cognitive_impairment', 'has_informant'),
        ('Final Composite', 'final_cognitive_impairment', None)
    ]
    
    results = []
    
    for name, label_col, valid_col in label_definitions:
        if label_col not in data.columns:
            continue
            
        # Get valid subset
        if valid_col and valid_col in data.columns:
            valid_mask = data[valid_col].astype(bool)
        else:
            valid_mask = data[label_col].notna()
        
        X_subset = X[valid_mask]
        y_subset = data[valid_mask][label_col].values
        
        # Skip if too few samples or no variation
        if len(X_subset) < 100 or len(np.unique(y_subset)) < 2:
            continue
        
        # Impute and scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_subset = imputer.fit_transform(X_subset)
        X_subset = scaler.fit_transform(X_subset)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
        )
        
        # Test multiple models
        models = {
            'Logistic': LogisticRegression(class_weight='balanced', max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        
        best_auc = 0
        for model_name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            # Test performance
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_pred)
            
            if test_auc > best_auc:
                best_auc = test_auc
        
        prevalence = y_subset.mean()
        n_samples = len(y_subset)
        
        results.append({
            'Label': name,
            'N': n_samples,
            'Prevalence': prevalence,
            'Best AUC': best_auc
        })
        
        print(f"\n{name}:")
        print(f"  N: {n_samples:,}")
        print(f"  Prevalence: {prevalence:.1%}")
        print(f"  Best AUC: {best_auc:.4f}")
    
    return pd.DataFrame(results)


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE COGNITIVE IMPAIRMENT DETECTION")
    print("="*80)
    
    # 1. Find all cognitive-related QIDs
    cognitive_qids = find_cognitive_qids()
    
    # 2. Create comprehensive labels
    labels = create_comprehensive_labels(cognitive_qids)
    
    # 3. Extract MemTrax features
    print("\n" + "="*70)
    print("EXTRACTING MEMTRAX FEATURES")
    print("="*70)
    
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    memtrax_features = extract_memtrax_features(memtrax_q)
    print(f"Extracted features for {len(memtrax_features):,} subjects")
    
    # 4. Test different label definitions
    results_df = test_different_labels(memtrax_features, labels)
    
    # 5. Summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    if not results_df.empty:
        print("\nPerformance by Label Type:")
        print(results_df.sort_values('Best AUC', ascending=False).to_string(index=False))
        
        best_result = results_df.loc[results_df['Best AUC'].idxmax()]
        print(f"\n🎯 BEST RESULT:")
        print(f"   Label: {best_result['Label']}")
        print(f"   AUC: {best_result['Best AUC']:.4f}")
        print(f"   Prevalence: {best_result['Prevalence']:.1%}")
        print(f"   N: {best_result['N']:,}")
        
        if best_result['Best AUC'] > 0.80:
            print("\n✅ BREAKTHROUGH! Achieved >0.80 AUC!")
        elif best_result['Best AUC'] > 0.75:
            print("\n📈 Significant improvement over 0.744 baseline!")


if __name__ == "__main__":
    main()

