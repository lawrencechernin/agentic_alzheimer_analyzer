#!/usr/bin/env python3
"""
Analyze SP-ECOG by cognitive domains
=====================================
SP-ECOG typically has domains like:
- Memory (remembering appointments, conversations)
- Language (finding words, following conversations)
- Visuospatial (getting lost, finding things)
- Executive (planning, organizing, multitasking)

Let's see which domains correlate with MemTrax performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")

def analyze_ecog_structure():
    """Analyze the structure of SP-ECOG questions"""
    print("\n" + "="*70)
    print("1. ANALYZING SP-ECOG STRUCTURE")
    print("="*70)
    
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    
    # Fix timepoint
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog['TimepointCode'] = sp_ecog['TimepointCode'].str.replace('sp-', '')
        sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    
    # Get QID columns
    qid_cols = [c for c in sp_ecog.columns if 'QID' in c and sp_ecog[c].dtype in [np.float64, np.int64]]
    
    print(f"Found {len(qid_cols)} SP-ECOG items")
    
    # Analyze each QID
    qid_info = []
    for qid in sorted(qid_cols):
        valid_values = sp_ecog[qid].dropna()
        # Exclude 8 (Don't Know)
        valid_values = valid_values[valid_values != 8]
        
        if len(valid_values) > 0:
            qid_info.append({
                'QID': qid,
                'N': len(valid_values),
                'Mean': valid_values.mean(),
                'Std': valid_values.std(),
                'Pct_Normal': (valid_values == 1).mean() * 100,
                'Pct_Mild': (valid_values == 2).mean() * 100,
                'Pct_Moderate': (valid_values == 3).mean() * 100,
                'Pct_Severe': (valid_values >= 4).mean() * 100
            })
    
    qid_df = pd.DataFrame(qid_info)
    
    # Group QIDs by patterns (likely domains)
    # Usually ECOG has numbered patterns like QID4-1 to QID4-10 for memory, etc.
    print("\nQID Groupings (potential domains):")
    
    # Extract QID prefixes
    qid_df['prefix'] = qid_df['QID'].str.extract(r'(QID\d+)')
    
    for prefix, group in qid_df.groupby('prefix'):
        print(f"\n{prefix} ({len(group)} items):")
        print(f"  Mean severity: {group['Mean'].mean():.2f}")
        print(f"  % with moderate+ changes: {group['Pct_Moderate'].mean() + group['Pct_Severe'].mean():.1f}%")
        print(f"  Items: {', '.join(group['QID'].head(5).tolist())}")
        if len(group) > 5:
            print(f"         ... and {len(group)-5} more")
    
    return sp_ecog, qid_cols, qid_df


def create_domain_scores(sp_ecog, qid_cols):
    """Create domain-specific scores from SP-ECOG"""
    print("\n" + "="*70)
    print("2. CREATING DOMAIN SCORES")
    print("="*70)
    
    # Define domains based on typical ECOG structure
    # We'll infer these from QID patterns
    domains = {}
    
    # Group QIDs by their prefix (e.g., QID4, QID5, etc.)
    for qid in qid_cols:
        prefix = qid.split('-')[0] if '-' in qid else qid[:5]
        if prefix not in domains:
            domains[prefix] = []
        domains[prefix].append(qid)
    
    # Calculate domain scores
    domain_scores = pd.DataFrame()
    domain_scores['SubjectCode'] = sp_ecog['SubjectCode']
    
    print("Domain scores created:")
    for domain_name, domain_qids in domains.items():
        if len(domain_qids) >= 2:  # Need at least 2 items for a domain
            # Calculate mean score for this domain (excluding 8='Don't Know')
            domain_data = sp_ecog[domain_qids].replace(8, np.nan)
            
            # Different aggregations
            domain_scores[f'{domain_name}_mean'] = domain_data.mean(axis=1)
            domain_scores[f'{domain_name}_max'] = domain_data.max(axis=1)
            domain_scores[f'{domain_name}_pct_abnormal'] = (domain_data >= 3).mean(axis=1)
            
            # Count valid responses
            domain_scores[f'{domain_name}_n_valid'] = domain_data.notna().sum(axis=1)
            
            print(f"  {domain_name}: {len(domain_qids)} items, Mean score={domain_scores[f'{domain_name}_mean'].mean():.2f}")
    
    # Overall scores
    all_items = sp_ecog[qid_cols].replace(8, np.nan)
    domain_scores['overall_mean'] = all_items.mean(axis=1)
    domain_scores['overall_max'] = all_items.max(axis=1)
    domain_scores['overall_pct_abnormal'] = (all_items >= 3).mean(axis=1)
    
    # Create binary impairment indicators for each domain
    for col in domain_scores.columns:
        if '_mean' in col:
            domain_name = col.replace('_mean', '')
            # Impaired if mean >= 2.5 or max >= 4
            domain_scores[f'{domain_name}_impaired'] = (
                (domain_scores[f'{domain_name}_mean'] >= 2.5) |
                (domain_scores.get(f'{domain_name}_max', 0) >= 4)
            ).astype(int)
    
    return domain_scores, domains


def correlate_with_memtrax(domain_scores):
    """Correlate domain scores with MemTrax performance"""
    print("\n" + "="*70)
    print("3. CORRELATING DOMAINS WITH MEMTRAX")
    print("="*70)
    
    # Load MemTrax data
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    
    # Quality filter
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Aggregate MemTrax features
    memtrax_agg = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'min'],
        'CorrectResponsesRT': ['mean', 'std', 'max'],
        'IncorrectPCT': 'mean'
    }).reset_index()
    
    memtrax_agg.columns = ['SubjectCode'] + [
        f"{col[0]}_{col[1]}" for col in memtrax_agg.columns[1:]
    ]
    
    # Add composite score
    memtrax_agg['cognitive_score'] = (
        memtrax_agg['CorrectResponsesRT_mean'] / 
        (memtrax_agg['CorrectPCT_mean'] + 0.01)
    )
    
    # Merge with domain scores
    combined = memtrax_agg.merge(domain_scores, on='SubjectCode', how='inner')
    
    print(f"Merged data: {len(combined):,} subjects with both MemTrax and SP-ECOG")
    
    if len(combined) == 0:
        print("No overlap found!")
        return None
    
    # Calculate correlations
    print("\nCorrelations with MemTrax Cognitive Score:")
    print("(Higher cognitive score = worse performance)")
    print("-" * 50)
    
    correlations = []
    for col in domain_scores.columns:
        if col != 'SubjectCode' and '_mean' in col and col in combined.columns:
            corr = combined['cognitive_score'].corr(combined[col])
            correlations.append({
                'Domain': col.replace('_mean', ''),
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    for _, row in corr_df.iterrows():
        print(f"  {row['Domain']:20s}: {row['Correlation']:+.3f}")
    
    return combined, corr_df


def test_domain_predictions(combined, domains):
    """Test how well each domain predicts MemTrax-based impairment"""
    print("\n" + "="*70)
    print("4. TESTING DOMAIN-SPECIFIC PREDICTIONS")
    print("="*70)
    
    # Create MemTrax-based impairment label (bottom 20% performers)
    threshold = combined['cognitive_score'].quantile(0.80)
    combined['memtrax_impaired'] = (combined['cognitive_score'] >= threshold).astype(int)
    
    print(f"MemTrax impairment rate (top 20% cognitive score): {combined['memtrax_impaired'].mean():.1%}")
    
    # Test each domain as predictor
    results = []
    
    # Test individual domains
    for domain_prefix in domains.keys():
        if f'{domain_prefix}_mean' in combined.columns:
            # Use domain mean score as predictor
            X = combined[[f'{domain_prefix}_mean']].fillna(0).values
            y = combined['memtrax_impaired'].values
            
            if len(np.unique(y)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Simple logistic regression
                model = LogisticRegression(class_weight='balanced')
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                
                results.append({
                    'Domain': domain_prefix,
                    'AUC': auc,
                    'N_items': len(domains[domain_prefix])
                })
    
    # Test combinations
    print("\nDomain combination results:")
    
    # All domains
    domain_features = [f'{d}_mean' for d in domains.keys() if f'{d}_mean' in combined.columns]
    if len(domain_features) > 1:
        X = combined[domain_features].fillna(0).values
        y = combined['memtrax_impaired'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        results.append({
            'Domain': 'ALL_COMBINED',
            'AUC': auc,
            'N_items': sum(len(v) for v in domains.values())
        })
    
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    
    print("\nPredicting MemTrax impairment from SP-ECOG domains:")
    print(results_df.to_string(index=False))
    
    return results_df, combined


def visualize_relationships(combined, domains):
    """Visualize the relationship between domains and MemTrax"""
    print("\n" + "="*70)
    print("5. VISUALIZING DOMAIN RELATIONSHIPS")
    print("="*70)
    
    # Select top domains by correlation
    domain_cols = [f'{d}_mean' for d in list(domains.keys())[:6] 
                   if f'{d}_mean' in combined.columns]
    
    if len(domain_cols) >= 2:
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, domain_col in enumerate(domain_cols[:6]):
            ax = axes[idx]
            
            # Remove NaN values for plotting
            plot_data = combined[[domain_col, 'cognitive_score']].dropna()
            
            # Scatter plot with trend line
            ax.scatter(plot_data[domain_col], plot_data['cognitive_score'], 
                      alpha=0.3, s=10)
            
            # Add trend line
            z = np.polyfit(plot_data[domain_col], plot_data['cognitive_score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data[domain_col].min(), plot_data[domain_col].max(), 100)
            ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)
            
            # Labels
            ax.set_xlabel(f'{domain_col.replace("_mean", "")} Score')
            ax.set_ylabel('MemTrax Cognitive Score')
            
            # Correlation
            corr = plot_data[domain_col].corr(plot_data['cognitive_score'])
            ax.set_title(f'r = {corr:.3f}')
        
        plt.suptitle('SP-ECOG Domains vs MemTrax Performance\n(Higher values = worse performance)')
        plt.tight_layout()
        plt.savefig('ecog_domains_memtrax.png', dpi=150)
        plt.show()
        
        print(f"Visualization saved to ecog_domains_memtrax.png")


def test_as_labels():
    """Test using domain-specific impairment as labels for MemTrax prediction"""
    print("\n" + "="*70)
    print("6. TESTING DOMAIN-SPECIFIC LABELS")
    print("="*70)
    
    # Load all data
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    if 'TimepointCode' in sp_ecog.columns:
        sp_ecog['TimepointCode'] = sp_ecog['TimepointCode'].str.replace('sp-', '')
        sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'm00']
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    
    # Get MemTrax features
    memtrax_features = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std'],
        'CorrectResponsesRT': ['mean', 'std'],
        'IncorrectPCT': 'mean'
    }).reset_index()
    
    memtrax_features.columns = ['SubjectCode'] + [
        f"{col[0]}_{col[1]}" for col in memtrax_features.columns[1:]
    ]
    
    # Get QID columns
    qid_cols = [c for c in sp_ecog.columns if 'QID' in c and sp_ecog[c].dtype in [np.float64, np.int64]]
    
    # Group by prefix
    domains = {}
    for qid in qid_cols:
        prefix = qid.split('-')[0] if '-' in qid else qid[:5]
        if prefix not in domains:
            domains[prefix] = []
        domains[prefix].append(qid)
    
    print("Testing each domain as label for MemTrax prediction:")
    
    results = []
    for domain_name, domain_qids in domains.items():
        if len(domain_qids) >= 3:  # Need sufficient items
            # Create domain-specific impairment label
            domain_data = sp_ecog[domain_qids].replace(8, np.nan)
            domain_mean = domain_data.mean(axis=1)
            domain_max = domain_data.max(axis=1)
            
            # Different thresholds for impairment
            labels_df = pd.DataFrame()
            labels_df['SubjectCode'] = sp_ecog['SubjectCode']
            labels_df['impaired_mean25'] = (domain_mean >= 2.5).astype(int)
            labels_df['impaired_mean30'] = (domain_mean >= 3.0).astype(int)
            labels_df['impaired_any4'] = (domain_max >= 4).astype(int)
            
            # Merge with MemTrax
            data = memtrax_features.merge(labels_df, on='SubjectCode', how='inner')
            
            if len(data) > 100:
                X = data[['CorrectPCT_mean', 'CorrectResponsesRT_mean', 
                         'CorrectPCT_std', 'CorrectResponsesRT_std']].fillna(0).values
                
                for label_type in ['impaired_mean25', 'impaired_mean30', 'impaired_any4']:
                    y = data[label_type].values
                    
                    if y.mean() > 0.01 and y.mean() < 0.99:  # Need some positive cases
                        # Simple train/test split
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42, stratify=y
                            )
                        except:
                            continue
                        
                        # Scale and train
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        
                        model = LogisticRegression(class_weight='balanced')
                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_pred)
                        
                        results.append({
                            'Domain': domain_name,
                            'Label_Type': label_type,
                            'N': len(data),
                            'Prevalence': y.mean(),
                            'AUC': auc
                        })
    
    if results:
        results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
        print("\nTop results:")
        print(results_df.head(10).to_string(index=False))
        
        best = results_df.iloc[0]
        print(f"\n🎯 BEST DOMAIN-SPECIFIC RESULT:")
        print(f"   Domain: {best['Domain']}")
        print(f"   Label: {best['Label_Type']}")
        print(f"   AUC: {best['AUC']:.4f}")
        print(f"   Prevalence: {best['Prevalence']:.1%}")


def main():
    print("\n" + "="*80)
    print("SP-ECOG DOMAIN ANALYSIS FOR MEMTRAX PREDICTION")
    print("="*80)
    
    # 1. Analyze SP-ECOG structure
    sp_ecog, qid_cols, qid_df = analyze_ecog_structure()
    
    # 2. Create domain scores
    domain_scores, domains = create_domain_scores(sp_ecog, qid_cols)
    
    # 3. Correlate with MemTrax
    combined, corr_df = correlate_with_memtrax(domain_scores)
    
    if combined is not None:
        # 4. Test predictions
        results_df, combined = test_domain_predictions(combined, domains)
        
        # 5. Visualize
        visualize_relationships(combined, domains)
    
    # 6. Test as labels
    test_as_labels()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
    Different SP-ECOG domains show varying relationships with MemTrax:
    - Some domains may correlate better (e.g., memory vs planning)
    - Using domain-specific thresholds might improve prediction
    - The overall weak correlation suggests fundamental measurement differences
    """)


if __name__ == "__main__":
    main()
