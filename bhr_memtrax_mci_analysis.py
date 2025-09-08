#!/usr/bin/env python3
"""
BHR MemTrax-MCI Prediction Analysis
==================================
Analyze how well MemTrax reaction times predict Mild Cognitive Impairment (MCI) 
in the Brain Health Registry (BHR) dataset.

Usage:
    python bhr_memtrax_mci_analysis.py

Requirements:
    pip install pandas scikit-learn matplotlib seaborn numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.impute import SimpleImputer

# Enhanced Data Merging (prevents Cartesian joins)
try:
    from improvements.enhanced_data_merging import smart_merge_datasets, EnhancedDataMerger
    ENHANCED_MERGING_AVAILABLE = True
    print("✅ Enhanced data merging available - Cartesian join protection enabled!")
except ImportError:
    ENHANCED_MERGING_AVAILABLE = False
    print("⚠️  Enhanced data merging not available - using basic merge (risk of Cartesian joins)")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BHRMemTraxMCIAnalyzer:
    """Analyzer for MemTrax reaction time prediction of MCI in BHR data"""
    
    def __init__(self, data_dir="../bhr/BHR-ALL-EXT_Mem_2022"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("bhr_memtrax_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.memtrax_data = None
        self.medical_data = None
        self.combined_data = None
        self.X = None
        self.y = None
        
        # Results storage
        self.results = {}
        self.models = {}
        
        print("🧠 BHR MEMTRAX-MCI PREDICTION ANALYSIS")
        print("=" * 50)
        print(f"📁 Data directory: {data_dir}")
        print(f"📊 Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load MemTrax and Medical History datasets"""
        print("\n📊 LOADING BHR DATASETS")
        print("-" * 30)
        
        # Load MemTrax data
        memtrax_file = self.data_dir / "MemTrax.csv"
        if not memtrax_file.exists():
            raise FileNotFoundError(f"MemTrax file not found: {memtrax_file}")
        
        print("Loading MemTrax data...")
        self.memtrax_data = pd.read_csv(memtrax_file, low_memory=False)
        print(f"✅ MemTrax: {self.memtrax_data.shape[0]:,} rows, {self.memtrax_data.shape[1]} columns")
        
        # Load Medical History data  
        medical_file = self.data_dir / "BHR_MedicalHx.csv"
        if not medical_file.exists():
            raise FileNotFoundError(f"Medical History file not found: {medical_file}")
        
        print("Loading Medical History data...")
        self.medical_data = pd.read_csv(medical_file, low_memory=False)
        print(f"✅ Medical: {self.medical_data.shape[0]:,} rows, {self.medical_data.shape[1]} columns")
    
    def aggregate_longitudinal_features(self, df):
        """Aggregate features across all timepoints per subject"""
        print("   📈 Aggregating features per subject (mean, std, min, max)...")
        
        # Select numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['SubjectCode', 'TimepointCode', 'DaysAfterBaseline']
        cols_to_agg = [col for col in numeric_cols if col not in exclude_cols]
        
        # Define aggregation functions
        agg_funcs = {}
        for col in cols_to_agg:
            agg_funcs[col] = ['mean', 'std', 'min', 'max']
        
        # Add test count
        agg_funcs['TimepointCode'] = 'count'
        
        # Perform aggregation
        df_agg = df.groupby('SubjectCode').agg(agg_funcs)
        
        # Flatten column names
        df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                          for col in df_agg.columns]
        df_agg.rename(columns={'TimepointCode_count': 'TestCount'}, inplace=True)
        
        # Add coefficient of variation for key features
        for col in cols_to_agg:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            if mean_col in df_agg.columns and std_col in df_agg.columns:
                df_agg[f'{col}_cv'] = df_agg[std_col] / (df_agg[mean_col] + 1e-6)
        
        # Add range features
        for col in cols_to_agg:
            max_col = f'{col}_max'
            min_col = f'{col}_min'
            if max_col in df_agg.columns and min_col in df_agg.columns:
                df_agg[f'{col}_range'] = df_agg[max_col] - df_agg[min_col]
        
        df_agg = df_agg.reset_index()
        print(f"   ✅ Aggregated {len(df):,} tests → {len(df_agg):,} subjects")
        print(f"   📊 Features: {len(cols_to_agg)} → {len(df_agg.columns)-1}")
        
        return df_agg
    
    def merge_datasets(self):
        """Enhanced merge using LONGITUDINAL AGGREGATION approach"""
        print("\n🔗 MERGING DATASETS (LONGITUDINAL OPTIMIZATION)")
        print("-" * 20)
        
        if self.memtrax_data is None or self.medical_data is None:
            raise ValueError("Must load data first")
        
        # Step 1: Apply Ashford quality criteria to ALL MemTrax tests
        print("✅ Step 1: Applying Ashford quality criteria...")
        memtrax_valid = self.memtrax_data[
            (self.memtrax_data['Status'] == 'Collected') &
            (self.memtrax_data['CorrectPCT'] >= 0.65) &  # 65% per best_model_runner
            (self.memtrax_data['CorrectResponsesRT'] >= 0.5) &
            (self.memtrax_data['CorrectResponsesRT'] <= 2.5) &
            (self.memtrax_data['IncorrectRejectionsN'].notna())
        ].copy()
        
        print(f"   Filtered: {len(self.memtrax_data):,} → {len(memtrax_valid):,} tests")
        print(f"   Subjects with valid tests: {memtrax_valid['SubjectCode'].nunique():,}")
        
        # Step 2: Add the CRITICAL cognitive score feature
        print("🧠 Step 2: Creating composite cognitive score...")
        memtrax_valid['CognitiveScore'] = (
            memtrax_valid['CorrectResponsesRT'] / 
            (memtrax_valid['CorrectPCT'] + 0.01)
        )
        print("   ✅ Added CognitiveScore (RT/accuracy ratio)")
        
        # Step 2b: Derive per-test advanced RT features before aggregation
        try:
            print("🧪 Step 2b: Deriving per-test RT features...")
            advanced_features_list = []
            for _, row in memtrax_valid.iterrows():
                advanced_features_list.append(self.extract_advanced_rt_features(row))
            adv_df = pd.DataFrame(advanced_features_list)
            # Coerce to numeric to avoid non-numeric aggregate noise
            adv_df = adv_df.apply(pd.to_numeric, errors='coerce')
            memtrax_valid = pd.concat([memtrax_valid.reset_index(drop=True), adv_df.reset_index(drop=True)], axis=1)
            print(f"   ✅ Added per-test RT feature columns: {len(adv_df.columns)}")
        except Exception as e:
            print(f"   ⚠️ Skipping per-test RT features due to error: {e}")

        # Step 3: Aggregate across ALL timepoints per subject
        print("📊 Step 3: Aggregating across all timepoints...")
        # Compute subject-level sequence features inspired by best_model_runner
        def compute_subject_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
            features = []
            for subject, g in df.groupby('SubjectCode'):
                # Concatenate individual RTs across tests
                all_rts: list[float] = []
                for rt_str in g.get('ReactionTimes', []):
                    rt_str = str(rt_str)
                    if rt_str and rt_str != 'nan':
                        try:
                            vals = [float(x.strip()) for x in rt_str.split(',') if x.strip()]
                            vals = [v for v in vals if 0.2 < v < 2.5]
                            all_rts.extend(vals)
                        except Exception:
                            continue
                seq_first_third_mean = 0.0
                seq_last_third_mean = 0.0
                seq_fatigue_effect = 0.0
                seq_mean_rt = np.nan
                seq_median_rt = np.nan
                long_reliability_change = 0.0
                if len(all_rts) >= 6:
                    n = len(all_rts)
                    k = max(1, n // 3)
                    seq_first_third_mean = float(np.mean(all_rts[:k]))
                    seq_last_third_mean = float(np.mean(all_rts[-k:]))
                    seq_fatigue_effect = float(seq_last_third_mean - seq_first_third_mean)
                    seq_mean_rt = float(np.mean(all_rts))
                    seq_median_rt = float(np.median(all_rts))
                    if len(all_rts) >= 2:
                        long_reliability_change = float(np.std(all_rts))
                # Longitudinal slopes (fallback to index if no time)
                long_n_timepoints = int(len(g))
                rt_series = pd.to_numeric(g['CorrectResponsesRT'], errors='coerce')
                if 'DaysAfterBaseline' in g.columns and g['DaysAfterBaseline'].notna().any():
                    x = pd.to_numeric(g['DaysAfterBaseline'], errors='coerce').fillna(method='ffill').fillna(0).values
                else:
                    x = np.arange(len(rt_series))
                try:
                    if len(rt_series.dropna()) >= 2:
                        slope = float(np.polyfit(x[:len(rt_series)], rt_series.fillna(method='ffill').fillna(rt_series.median()).values, 1)[0])
                    else:
                        slope = 0.0
                except Exception:
                    slope = 0.0
                features.append({
                    'SubjectCode': subject,
                    'seq_first_third_mean': seq_first_third_mean,
                    'seq_last_third_mean': seq_last_third_mean,
                    'seq_fatigue_effect': seq_fatigue_effect,
                    'seq_mean_rt': seq_mean_rt if not np.isnan(seq_mean_rt) else 0.0,
                    'seq_median_rt': seq_median_rt if not np.isnan(seq_median_rt) else 0.0,
                    'long_reliability_change': long_reliability_change,
                    'long_n_timepoints': long_n_timepoints,
                    'long_rt_slope': slope
                })
            return pd.DataFrame(features)

        seq_features = compute_subject_sequence_features(memtrax_valid)
        memtrax_aggregated = self.aggregate_longitudinal_features(memtrax_valid)
        # Merge aggregated with sequence features
        memtrax_aggregated = memtrax_aggregated.merge(seq_features, on='SubjectCode', how='left')
        
        # Step 4: Get baseline medical labels only (avoid leakage)
        print("🏥 Step 4: Extracting baseline medical labels...")
        medical_baseline = self.medical_data[
            self.medical_data['TimepointCode'] == 'm00'
        ].drop_duplicates(subset=['SubjectCode'], keep='first')
        print(f"   Medical baseline subjects: {len(medical_baseline):,}")
        
        # Step 5: Merge aggregated features with baseline labels
        print("🔗 Step 5: Final merge...")
        memtrax_for_merge = memtrax_aggregated
        medical_for_merge = medical_baseline        # Use Enhanced Smart Merging to prevent Cartesian joins
        print("Using enhanced smart merge to prevent Cartesian joins...")
        if ENHANCED_MERGING_AVAILABLE:
            try:
                self.combined_data = smart_merge_datasets(
                    memtrax_for_merge, medical_for_merge, 
                    subject_col="SubjectCode",
                    df1_name="MemTrax", df2_name="Medical History"
                )
                print("🛡️ Enhanced merge completed successfully!")
            except Exception as e:
                print(f"⚠️ Enhanced merge failed: {e}")
                print("Falling back to basic merge...")
                self.combined_data = memtrax_for_merge.merge(
                    medical_for_merge,
                    on='SubjectCode',
                    how='inner',
                    suffixes=('_memtrax', '_medical')
                )
        else:
            print("⚠️ Using basic merge - risk of Cartesian joins!")
            self.combined_data = self.memtrax_data.merge(
                self.medical_data,
                on='SubjectCode',
                how='inner',
                suffixes=('_memtrax', '_medical')
            )
        
        print(f"✅ Combined: {self.combined_data.shape[0]:,} rows, {self.combined_data.shape[1]} columns")
        
        # Check merge quality
        memtrax_subjects = set(self.memtrax_data['SubjectCode'].unique())
        medical_subjects = set(self.medical_data['SubjectCode'].unique())
        combined_subjects = set(self.combined_data['SubjectCode'].unique())
        
        print(f"📊 Merge Statistics:")
        print(f"   MemTrax subjects: {len(memtrax_subjects):,}")
        print(f"   Medical subjects: {len(medical_subjects):,}")  
        print(f"   Overlapping subjects: {len(combined_subjects):,}")
        print(f"   Merge rate: {len(combined_subjects)/min(len(memtrax_subjects), len(medical_subjects))*100:.1f}%")

    def add_demographic_features(self):
        """Ensure age and education features (and interactions) are present and clean."""
        print("\n🧬 ADDING DEMOGRAPHIC FEATURES (age, education)")
        if self.combined_data is None:
            raise ValueError("Must merge datasets before adding demographics")
        
        df = self.combined_data
        
        age_candidates = [
            'Age_Baseline', 'Age', 'AgeAtBaseline', 'AgeYears', 'age'
        ]
        edu_candidates = [
            'YearsEducationUS_Converted', 'Education_Years', 'education_years',
            'YearsEducationUS', 'EducationYears', 'Education', 'years_education'
        ]
        
        def find_col(candidates, columns):
            return next((c for c in candidates if c in columns), None)
        
        age_col = find_col(age_candidates, df.columns)
        edu_col = find_col(edu_candidates, df.columns)
        
        # If missing, try to load from demographics/profile files and merge
        if age_col is None or edu_col is None:
            sources = [
                ('BHR_Demographics.csv', ['SubjectCode', 'Age_Baseline', 'YearsEducationUS_Converted', 'Gender']),
                ('Profile.csv', ['SubjectCode', 'YearsEducationUS_Converted', 'Age', 'Gender']),
                ('Participants.csv', ['SubjectCode', 'Age_Baseline', 'YearsEducationUS_Converted', 'Gender']),
                ('Subjects.csv', ['SubjectCode', 'Age_Baseline'])
            ]
            for filename, desired_cols in sources:
                csv_path = self.data_dir / filename
                if not csv_path.exists():
                    continue
                try:
                    src = pd.read_csv(csv_path, low_memory=False)
                    # Accept 'Code' as subject key
                    if 'SubjectCode' not in src.columns and 'Code' in src.columns:
                        src = src.rename(columns={'Code': 'SubjectCode'})
                    if 'SubjectCode' not in src.columns:
                        continue
                    keep_cols = [c for c in desired_cols if c in src.columns]
                    if len(keep_cols) <= 1:  # only SubjectCode present
                        continue
                    src_small = src[keep_cols].copy()
                    src_small = src_small.drop_duplicates(subset=['SubjectCode'], keep='first')
                    before_cols = set(df.columns)
                    df = df.merge(src_small, on='SubjectCode', how='left')
                    added_cols = [c for c in df.columns if c not in before_cols]
                    if added_cols:
                        print(f"   ➕ Merged {filename} (added: {', '.join(added_cols)[:120]}{'...' if len(', '.join(added_cols))>120 else ''})")
                except Exception as e:
                    print(f"   ⚠️ Failed reading/merging {filename}: {e}")
            # Refresh detections after merges
            age_col = find_col(age_candidates, df.columns)
            edu_col = find_col(edu_candidates, df.columns)
        
        if age_col is None and edu_col is None:
            print("   ⚠️ No age or education columns found in merged data")
            self.combined_data = df
            return
        
        if age_col is not None and age_col != 'Age_Baseline':
            df['Age_Baseline'] = pd.to_numeric(df[age_col], errors='coerce')
        elif age_col == 'Age_Baseline':
            df['Age_Baseline'] = pd.to_numeric(df['Age_Baseline'], errors='coerce')
        
        if edu_col is not None and edu_col != 'YearsEducationUS_Converted':
            df['YearsEducationUS_Converted'] = pd.to_numeric(df[edu_col], errors='coerce')
        elif edu_col == 'YearsEducationUS_Converted':
            df['YearsEducationUS_Converted'] = pd.to_numeric(df['YearsEducationUS_Converted'], errors='coerce')
        
        # Derived features (will be imputed later with other features)
        if 'Age_Baseline' in df.columns:
            df['Age_Baseline_Squared'] = df['Age_Baseline'] ** 2
            df['Age_Per_Decade'] = df['Age_Baseline'] / 10.0
        
        if 'YearsEducationUS_Converted' in df.columns:
            df['Education_Years'] = df['YearsEducationUS_Converted']
            df['Education_Squared'] = df['YearsEducationUS_Converted'] ** 2
        
        if 'Age_Baseline' in df.columns and 'YearsEducationUS_Converted' in df.columns:
            df['Age_Education_Interaction'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
            # Cognitive reserve proxy inspired feature
            df['CognitiveReserveProxy'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] / 50.0 + 1e-6)
        
        self.combined_data = df
        available = [c for c in ['Age_Baseline','YearsEducationUS_Converted','Age_Education_Interaction','CognitiveReserveProxy'] if c in df.columns]
        # Encode gender if present
        if 'Gender' in df.columns and 'Gender_Numeric' not in df.columns:
            gender_map = {'Male': 1, 'M': 1, 'Female': 0, 'F': 0}
            df['Gender_Numeric'] = df['Gender'].map(gender_map)
        # Add age interactions if base features available
        if 'Age_Baseline' in df.columns:
            if 'CorrectResponsesRT_mean' in df.columns:
                df['age_rt_interaction'] = df['CorrectResponsesRT_mean'] * (df['Age_Baseline'] / 65.0)
            if 'long_reliability_change' in df.columns:
                df['age_variability_interaction'] = df['long_reliability_change'] * (df['Age_Baseline'] / 65.0)
            if 'CorrectPCT_mean' in df.columns and 'long_reliability_change' in df.columns:
                df['accuracy_stability'] = df['CorrectPCT_mean'] / (df['long_reliability_change'] + 1e-6)
        self.combined_data = df
        available = [c for c in ['Age_Baseline','YearsEducationUS_Converted','Age_Education_Interaction','CognitiveReserveProxy','Gender_Numeric','age_rt_interaction','age_variability_interaction','accuracy_stability'] if c in df.columns]
        print(f"   ✅ Added/standardized: {', '.join(available) if available else 'none'}")
    
    def prepare_target_variable(self):
        """Prepare MCI target variable from QID1-13"""
        print("\n🎯 PREPARING MCI TARGET VARIABLE")
        print("-" * 35)
        
        # Check for MCI column (QID1-13)
        mci_col = 'QID1-13'
        if mci_col not in self.combined_data.columns:
            raise ValueError(f"MCI column '{mci_col}' not found in data")
        
        print(f"Using {mci_col}: 'Mild Cognitive Impairment'")
        
        # Show raw distribution
        raw_dist = self.combined_data[mci_col].value_counts().sort_index()
        print(f"Raw distribution: {dict(raw_dist)}")
        print("(1=Yes MCI, 2=No MCI, NaN=Missing)")
        
        # Create clean binary target
        valid_mask = self.combined_data[mci_col].isin([1.0, 2.0])
        self.combined_data = self.combined_data[valid_mask].copy()
        self.y = (self.combined_data[mci_col] == 1.0).astype(int)
        
        print(f"\n📈 Final Target Distribution:")
        print(f"   Valid responses: {len(self.y):,}")
        print(f"   MCI cases (1): {self.y.sum():,}")
        print(f"   Healthy (0): {(~self.y.astype(bool)).sum():,}")
        print(f"   MCI prevalence: {self.y.mean()*100:.2f}%")
        
        if self.y.sum() < 50:
            print("⚠️  WARNING: Very few MCI cases - results may be unreliable")
        elif self.y.sum() < 500:
            print("⚠️  CAUTION: Limited MCI cases - results should be interpreted carefully")
        else:
            print("✅ Sufficient MCI cases for reliable machine learning")
    
    def extract_advanced_rt_features(self, row):
        """Extract advanced features from individual reaction times"""
        import numpy as np
        from scipy import stats
        
        features = {}
        
        # Parse individual reaction times from ReactionTimes column
        rt_str = str(row.get('ReactionTimes', ''))
        reaction_times = []
        
        if rt_str and rt_str != 'nan':
            try:
                # Parse comma-separated RTs
                all_rts = [float(x.strip()) for x in rt_str.split(',') if x.strip()]
                # Filter valid RTs (exclude 3.0 timeouts and invalid values)
                reaction_times = [rt for rt in all_rts if 0.2 < rt < 2.5]
            except:
                pass
        
        # Compute advanced features if we have enough valid RTs
        if len(reaction_times) >= 10:
            rt_array = np.array(reaction_times)
            
            # Basic statistics
            features['RT_Mean_Individual'] = np.mean(rt_array)
            features['RT_Std_Individual'] = np.std(rt_array)
            features['RT_CV'] = np.std(rt_array) / (np.mean(rt_array) + 0.001)
            features['RT_Median_Individual'] = np.median(rt_array)
            features['RT_Range'] = np.max(rt_array) - np.min(rt_array)
            features['RT_IQR'] = np.percentile(rt_array, 75) - np.percentile(rt_array, 25)
            
            # Distribution shape
            features['RT_Skewness'] = stats.skew(rt_array)
            features['RT_Kurtosis'] = stats.kurtosis(rt_array)
            
            # Percentiles
            for p in [10, 25, 75, 90]:
                features[f'RT_P{p}'] = np.percentile(rt_array, p)
            
            # Temporal patterns
            if len(rt_array) >= 20:
                n_half = len(rt_array) // 2
                first_half = rt_array[:n_half]
                second_half = rt_array[n_half:]
                
                features['RT_FatigueEffect'] = np.mean(second_half) - np.mean(first_half)
                features['RT_VariabilityChange'] = np.std(second_half) - np.std(first_half)
                features['RT_Trend'] = np.polyfit(range(len(rt_array)), rt_array, 1)[0]
            
            # Response quality indicators
            features['RT_LapseRate'] = np.mean(rt_array > 2.0)
            features['RT_FastRate'] = np.mean(rt_array < 0.4)
            features['RT_OptimalRate'] = np.mean((rt_array >= 0.5) & (rt_array <= 1.5))
            
            # Count of valid responses
            features['RT_ValidCount'] = len(reaction_times)
        else:
            # Set defaults if insufficient data
            feature_names = ['RT_Mean_Individual', 'RT_Std_Individual', 'RT_CV', 'RT_Median_Individual',
                           'RT_Range', 'RT_IQR', 'RT_Skewness', 'RT_Kurtosis', 'RT_P10', 'RT_P25',
                           'RT_P75', 'RT_P90', 'RT_FatigueEffect', 'RT_VariabilityChange', 'RT_Trend',
                           'RT_LapseRate', 'RT_FastRate', 'RT_OptimalRate', 'RT_ValidCount']
            for name in feature_names:
                features[name] = 0
        
        return features
    
    def prepare_features(self):
        """Prepare features from aggregated longitudinal data"""
        print("\n📊 PREPARING MEMTRAX FEATURES")
        print("-" * 32)
        
        if self.combined_data is None:
            raise ValueError("Must merge datasets first")
        
        # Since we've already aggregated, most columns are now features
        # Exclude non-feature columns
        exclude_cols = ['SubjectCode', 'TimepointCode', 'Date', 'Status'] + \
                      [col for col in self.combined_data.columns if 'QID' in col]
        
        # Extract advanced features from individual RTs if available (but less critical now)
        advanced_df = pd.DataFrame()
        if 'ReactionTimes' in self.combined_data.columns:
            print("🧠 Extracting individual RT features...")
            advanced_features_list = []
            for idx, row in self.combined_data.iterrows():
                advanced_features_list.append(self.extract_advanced_rt_features(row))
            advanced_df = pd.DataFrame(advanced_features_list)
            print(f"   ✅ Extracted {len(advanced_df.columns)} individual RT features")
        
        # Get all numeric columns as features
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Combine aggregated features with any advanced features
        if not advanced_df.empty:
            X_raw = pd.concat([
                self.combined_data[feature_cols].reset_index(drop=True),
                advanced_df.reset_index(drop=True)
            ], axis=1)
        else:
            X_raw = self.combined_data[feature_cols].copy()
        
        # Coerce all to numeric and drop columns that are entirely NaN
        X_raw = X_raw.apply(pd.to_numeric, errors='coerce')
        all_nan_cols = [c for c in X_raw.columns if X_raw[c].notna().sum() == 0]
        if all_nan_cols:
            X_raw = X_raw.drop(columns=all_nan_cols)
            print(f"   ⚠️ Dropped {len(all_nan_cols)} all-NaN feature columns")
        
        # Re-align feature column list after cleaning
        all_feature_cols = list(X_raw.columns)
        
        # Count feature types
        rt_features = [col for col in all_feature_cols if 'rt' in col.lower() or 'reaction' in col.lower()]
        score_features = [col for col in all_feature_cols if 'score' in col.lower() or 'cognitive' in col.lower()]
        stat_features = [col for col in all_feature_cols if '_std' in col or '_cv' in col or '_range' in col]
        
        print(f"Found features:")
        print(f"   RT-related: {len(rt_features)}")
        print(f"   Score-related: {len(score_features)}")
        print(f"   Variability/stats: {len(stat_features)}")
        print(f"   Total: {len(all_feature_cols)}")
        
        if len(all_feature_cols) == 0:
            raise ValueError("No MemTrax features found")
        
        # Show feature details
        print(f"\nSample features:")
        for i, col in enumerate(all_feature_cols[:10], 1):  # Show first 10
            print(f"   {i:2d}. {col}")
        if len(all_feature_cols) > 10:
            print(f"   ... and {len(all_feature_cols) - 10} more")
        
        # Handle missing values
        missing_pct = X_raw.isnull().mean() * 100
        print(f"\nMissing data: {missing_pct.mean():.1f}% average across features")
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_raw)
        # Use the cleaned column list to avoid shape mismatch
        self.X = pd.DataFrame(X_imputed, columns=list(X_raw.columns), index=X_raw.index)
        
        print(f"✅ Feature matrix: {self.X.shape[0]:,} samples × {self.X.shape[1]} features")
        
        # Show key feature statistics
        if 'CognitiveScore_mean' in self.X.columns:
            print(f"\n🧠 Key feature - CognitiveScore_mean:")
            print(f"   Mean: {self.X['CognitiveScore_mean'].mean():.3f}")
            print(f"   Std: {self.X['CognitiveScore_mean'].std():.3f}")
        
        if 'TestCount' in self.X.columns:
            print(f"\n📊 Test counts per subject:")
            print(f"   Mean: {self.X['TestCount'].mean():.1f}")
            print(f"   Max: {self.X['TestCount'].max():.0f}")
    
    def train_models(self):
        """Train multiple ML models for MCI prediction"""
        print("\n🤖 TRAINING MACHINE LEARNING MODELS")
        print("-" * 40)
        
        if self.X is None or self.y is None:
            raise ValueError("Must prepare features and target first")
        
        # Ensure we have matching indices
        common_idx = self.X.index.intersection(self.y.index)
        X = self.X.loc[common_idx]
        y = self.y.loc[common_idx]
        
        print(f"Training dataset: {len(X):,} samples, {X.shape[1]} features")
        print(f"Class balance: {y.sum()} MCI, {(~y.astype(bool)).sum()} Healthy")
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        
        # Convert back to DataFrame for feature names

        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)

        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

        
        # Define models (including Logistic Regression which achieved 0.71 AUC)
        from sklearn.linear_model import LogisticRegression
        
        models = {
            'Logistic Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        self.results = {}
        self.models = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'train_f1': f1_score(y_train, y_pred_train),
                'test_f1': f1_score(y_test, y_pred_test),
                'test_precision': precision_score(y_test, y_pred_test),
                'test_recall': recall_score(y_test, y_pred_test), 
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
            }
            
            self.results[name] = metrics
            self.models[name] = {
                'model': model,
                'scaler': scaler,
                'y_test': y_test,
                'y_pred': y_pred_test,
                'y_pred_proba': y_pred_proba_test
            }
            
            print(f"   CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
            print(f"   Test F1: {metrics['test_f1']:.3f}")
            print(f"   Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
            print(f"   Test Precision: {metrics['test_precision']:.3f}")
            print(f"   Test Recall: {metrics['test_recall']:.3f}")
    
    def analyze_feature_importance(self):
        """Analyze which MemTrax features are most predictive of MCI"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 35)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
        best_model = self.models[best_model_name]['model']
        
        print(f"Analyzing {best_model_name} (best F1 score)")
        
        if hasattr(best_model, 'feature_importances_'):
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🏆 TOP 10 MOST PREDICTIVE FEATURES:")
            print("-" * 45)
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                feature = row['feature']
                importance = row['importance']
                print(f"{i:2d}. {feature[:35]:35} {importance:.4f}")
            
            # Save full importance ranking
            importance_file = self.output_dir / 'feature_importance.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"\n💾 Full feature importance saved: {importance_file}")
            
            return importance_df
        else:
            print("❌ Selected model doesn't provide feature importance")
            return None
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 CREATING VISUALIZATIONS")
        print("-" * 28)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        self._plot_model_comparison()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Feature Importance (if available)
        importance_df = self.analyze_feature_importance()
        if importance_df is not None:
            self._plot_feature_importance(importance_df)
        
        # 4. Confusion Matrix for best model
        self._plot_confusion_matrix()
        
        print("✅ All visualizations saved!")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        metrics = ['test_f1', 'test_roc_auc', 'test_precision', 'test_recall']
        metric_names = ['F1 Score', 'ROC-AUC', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [self.results[model][metric] for model in self.results.keys()]
            models = list(self.results.keys())
            
            bars = axes[i].bar(models, values, alpha=0.7, color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{name} by Model')
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Model comparison plot saved")
    
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name in self.results.keys():
            model_data = self.models[name]
            y_test = model_data['y_test']
            y_pred_proba = model_data['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = self.results[name]['test_roc_auc']
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curves: MemTrax Features → MCI Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ ROC curves plot saved")
    
    def _plot_feature_importance(self, importance_df):
        """Plot feature importance"""
        # Plot top 15 features
        top_features = importance_df.head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 MemTrax Features for MCI Prediction')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Feature importance plot saved")
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix for best model"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
        model_data = self.models[best_model_name]
        
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Healthy', 'MCI'],
                   yticklabels=['Healthy', 'MCI'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Confusion matrix plot saved")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📋 GENERATING FINAL REPORT")
        print("-" * 28)
        
        # Create summary report
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_directory': str(self.data_dir),
                'total_subjects': len(self.combined_data) if self.combined_data is not None else 0,
                'mci_cases': int(self.y.sum()) if self.y is not None else 0,
                'healthy_cases': int((~self.y.astype(bool)).sum()) if self.y is not None else 0,
                'features_used': len(self.X.columns) if self.X is not None else 0
            },
            'model_performance': self.results,
            'data_summary': {
                'memtrax_rows': len(self.memtrax_data) if self.memtrax_data is not None else 0,
                'medical_rows': len(self.medical_data) if self.medical_data is not None else 0,
                'merged_rows': len(self.combined_data) if self.combined_data is not None else 0
            }
        }
        
        # Save JSON report
        json_file = self.output_dir / 'analysis_report.json'
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate text summary
        self._generate_text_summary(report)
        
        print(f"✅ JSON report saved: {json_file}")
    
    def _generate_text_summary(self, report):
        """Generate human-readable summary"""
        summary_file = self.output_dir / 'ANALYSIS_SUMMARY.txt'
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BHR MEMTRAX-MCI PREDICTION ANALYSIS RESULTS\n") 
            f.write("=" * 80 + "\n\n")
            
            # Overview
            f.write("📊 DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analysis Date: {report['analysis_metadata']['timestamp'][:10]}\n")
            f.write(f"Total Subjects: {report['analysis_metadata']['total_subjects']:,}\n")
            f.write(f"MCI Cases: {report['analysis_metadata']['mci_cases']:,}\n")
            f.write(f"Healthy Cases: {report['analysis_metadata']['healthy_cases']:,}\n")
            mci_rate = report['analysis_metadata']['mci_cases'] / (report['analysis_metadata']['mci_cases'] + report['analysis_metadata']['healthy_cases']) * 100
            f.write(f"MCI Prevalence: {mci_rate:.2f}%\n")
            f.write(f"MemTrax Features: {report['analysis_metadata']['features_used']}\n\n")
            
            # Model Results
            f.write("🤖 MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            
            best_model = max(report['model_performance'].keys(), 
                           key=lambda x: report['model_performance'][x]['test_f1'])
            
            f.write(f"Best Model: {best_model}\n\n")
            
            for model_name, metrics in report['model_performance'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  F1 Score: {metrics['test_f1']:.3f}\n")
                f.write(f"  ROC-AUC: {metrics['test_roc_auc']:.3f}\n")
                f.write(f"  Precision: {metrics['test_precision']:.3f}\n")
                f.write(f"  Recall: {metrics['test_recall']:.3f}\n")
                f.write(f"  Cross-Validation F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}\n\n")
            
            # Interpretation
            f.write("🎯 INTERPRETATION\n")
            f.write("-" * 17 + "\n")
            best_f1 = report['model_performance'][best_model]['test_f1']
            best_auc = report['model_performance'][best_model]['test_roc_auc']
            
            if best_f1 > 0.4:
                interpretation = "STRONG predictive power"
            elif best_f1 > 0.3:
                interpretation = "GOOD predictive power"
            elif best_f1 > 0.2:
                interpretation = "MODERATE predictive power"
            else:
                interpretation = "LIMITED predictive power"
            
            f.write(f"MemTrax reaction times show {interpretation} for predicting MCI.\n")
            f.write(f"Best F1 Score: {best_f1:.3f} (range: 0.0-1.0, higher is better)\n")
            f.write(f"Best ROC-AUC: {best_auc:.3f} (range: 0.5-1.0, higher is better)\n\n")
            
            f.write("📁 OUTPUT FILES\n")
            f.write("-" * 15 + "\n")
            f.write("• model_comparison.png - Model performance comparison\n")
            f.write("• roc_curves.png - ROC curves for all models\n")
            f.write("• feature_importance.png - Most predictive MemTrax features\n")
            f.write("• confusion_matrix.png - Prediction accuracy breakdown\n")
            f.write("• feature_importance.csv - Complete feature rankings\n")
            f.write("• analysis_report.json - Complete technical results\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Analysis complete! MemTrax reaction times can predict MCI with ")
            f.write(f"{interpretation.lower()}.\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Text summary saved: {summary_file}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Merge datasets  
            self.merge_datasets()
            
            # Inject demographics before building features/target
            self.add_demographic_features()
            
            # Step 3: Prepare target
            self.prepare_target_variable()
            
            # Step 4: Prepare features
            self.prepare_features()
            
            # Step 5: Train models
            self.train_models()
            
            # Step 6: Create visualizations
            self.create_visualizations()
            
            # Step 7: Generate report
            self.generate_report()
            
            # Final summary
            print("\n" + "=" * 60)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
            best_f1 = self.results[best_model]['test_f1']
            best_auc = self.results[best_model]['test_roc_auc']
            
            print(f"🏆 Best Model: {best_model}")
            print(f"📊 F1 Score: {best_f1:.3f}")
            print(f"📈 ROC-AUC: {best_auc:.3f}")
            print(f"📁 Results saved to: {self.output_dir}/")
            print(f"📋 Summary: {self.output_dir}/ANALYSIS_SUMMARY.txt")
            
            if best_f1 > 0.3:
                print("✅ CONCLUSION: MemTrax reaction times show GOOD predictive power for MCI!")
            elif best_f1 > 0.2:
                print("⚠️  CONCLUSION: MemTrax reaction times show MODERATE predictive power for MCI")
            else:
                print("❌ CONCLUSION: MemTrax reaction times show LIMITED predictive power for MCI")
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function"""
    print("Starting BHR MemTrax-MCI Analysis...")
    
    # Initialize analyzer
    analyzer = BHRMemTraxMCIAnalyzer()
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎯 Analysis completed successfully!")
        print("📊 Check the 'bhr_memtrax_results/' folder for all outputs.")
    else:
        print("\n💥 Analysis failed. Check error messages above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 