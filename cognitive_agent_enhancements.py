"""
Enhanced CDR Prediction Methods for Cognitive Analysis Agent
Based on research findings from top Kaggle notebooks and recent literature
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import logging

class EnhancedCDRPredictor:
    """Enhanced CDR prediction with advanced feature engineering and ensemble methods"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_brain_atrophy_rate(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate brain atrophy rate for longitudinal data
        Research shows:
        - Nondemented: -0.49% ± 0.56% per year
        - Demented: -0.87% ± 0.99% per year (p<0.01)
        """
        self.logger.info("📊 Calculating brain atrophy rates from longitudinal data...")
        
        atrophy_features = []
        
        # Group by subject to calculate atrophy over time
        if 'Subject_ID' in longitudinal_df.columns and 'Visit' in longitudinal_df.columns:
            for subject_id, group in longitudinal_df.groupby('Subject_ID'):
                if len(group) > 1:  # Need at least 2 visits
                    # Sort by visit number
                    group = group.sort_values('Visit')
                    
                    # Calculate nWBV change rate if available
                    if 'nWBV' in group.columns and 'MR Delay' in group.columns:
                        first_nwbv = group['nWBV'].iloc[0]
                        last_nwbv = group['nWBV'].iloc[-1]
                        time_diff = group['MR Delay'].iloc[-1] if group['MR Delay'].iloc[-1] > 0 else 365
                        
                        # Annual atrophy rate
                        annual_atrophy = ((last_nwbv - first_nwbv) / first_nwbv) * (365 / time_diff) * 100
                        
                        atrophy_features.append({
                            'Subject_ID': subject_id,
                            'nWBV_atrophy_rate': annual_atrophy,
                            'baseline_nWBV': first_nwbv,
                            'final_nWBV': last_nwbv,
                            'observation_days': time_diff
                        })
        
        if atrophy_features:
            atrophy_df = pd.DataFrame(atrophy_features)
            self.logger.info(f"   ✅ Calculated atrophy rates for {len(atrophy_df)} subjects")
            
            # Add clinical interpretation
            atrophy_df['atrophy_severity'] = atrophy_df['nWBV_atrophy_rate'].apply(
                lambda x: 'severe' if x < -0.87 else ('mild' if x < -0.49 else 'normal')
            )
            
            return atrophy_df
        else:
            self.logger.warning("   ⚠️ Could not calculate atrophy rates (insufficient longitudinal data)")
            return pd.DataFrame()
    
    def enhance_brain_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance brain volume features based on research findings
        - ASF-eTIV correlation validation (should be negative, t=-88.7, p<0.001)
        - Age-nWBV interaction (t=-36.4, p<0.001)
        - Gender-specific normalization
        """
        self.logger.info("🧠 Enhancing brain volume features with advanced normalization...")
        
        enhanced_df = df.copy()
        
        # 1. Validate ASF-eTIV correlation
        if 'ASF' in df.columns and 'eTIV' in df.columns:
            correlation = df[['ASF', 'eTIV']].dropna().corr().iloc[0, 1]
            self.logger.info(f"   ASF-eTIV correlation: {correlation:.3f} (expected: strongly negative)")
            
            if correlation > -0.7:
                self.logger.warning("   ⚠️ Weak ASF-eTIV correlation detected - normalization may be suboptimal")
            
            # Create validated brain volume metric
            enhanced_df['validated_brain_volume'] = enhanced_df['eTIV'] * enhanced_df['ASF']
        
        # 2. Age-nWBV interaction feature
        if 'Age' in df.columns and 'nWBV' in df.columns:
            # Normalize age and nWBV first
            age_norm = (df['Age'] - df['Age'].mean()) / df['Age'].std()
            nwbv_norm = (df['nWBV'] - df['nWBV'].mean()) / df['nWBV'].std()
            
            # Create interaction term
            enhanced_df['age_nWBV_interaction'] = age_norm * nwbv_norm
            
            # Age-adjusted brain volume
            enhanced_df['age_adjusted_nWBV'] = df['nWBV'] + 0.003 * (75 - df['Age'])  # Adjust to age 75 baseline
            
            self.logger.info("   ✅ Added age-nWBV interaction features")
        
        # 3. Gender-specific brain volume normalization
        if 'Gender' in df.columns and 'nWBV' in df.columns:
            # Calculate gender-specific norms
            if 'Gender_M' in df.columns:
                gender_col = 'Gender_M'
            else:
                gender_col = 'Gender'
                
            for gender in df[gender_col].unique():
                gender_mask = df[gender_col] == gender
                if gender_mask.sum() > 10:  # Need sufficient samples
                    gender_mean = df.loc[gender_mask, 'nWBV'].mean()
                    gender_std = df.loc[gender_mask, 'nWBV'].std()
                    
                    # Z-score within gender
                    enhanced_df.loc[gender_mask, 'nWBV_gender_zscore'] = (
                        (df.loc[gender_mask, 'nWBV'] - gender_mean) / gender_std
                    )
            
            self.logger.info("   ✅ Added gender-specific brain volume normalization")
        
        # 4. CDR-stratified nWBV features (research shows p<0.01 differences)
        if 'CDR' in df.columns and 'nWBV' in df.columns:
            # Calculate deviation from CDR group mean
            for cdr_value in df['CDR'].unique():
                cdr_mask = df['CDR'] == cdr_value
                if cdr_mask.sum() > 5:
                    cdr_mean = df.loc[cdr_mask, 'nWBV'].mean()
                    enhanced_df.loc[cdr_mask, 'nWBV_cdr_deviation'] = df.loc[cdr_mask, 'nWBV'] - cdr_mean
        
        # 5. Brain volume ratio features
        if 'eTIV' in df.columns and 'nWBV' in df.columns:
            # Brain atrophy index
            enhanced_df['brain_atrophy_index'] = 1 - df['nWBV']
            
            # Volume preservation ratio
            enhanced_df['volume_preservation_ratio'] = df['nWBV'] / df['eTIV'].apply(lambda x: x/1500 if x > 0 else 1)
        
        self.logger.info(f"   ✅ Enhanced features: added {len(enhanced_df.columns) - len(df.columns)} new brain volume features")
        
        return enhanced_df
    
    def create_ensemble_model(self, X_train, y_train, X_test=None, y_test=None) -> Dict[str, Any]:
        """
        Create ensemble model combining multiple approaches
        Research shows GBM achieved 91.3% accuracy with socio-demographic + MMSE
        """
        self.logger.info("🚀 Building advanced ensemble model...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Check for XGBoost availability
        try:
            from xgboost import XGBClassifier
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
            self.logger.warning("XGBoost not available - using alternative models")
        
        # Initialize models with research-optimized parameters
        models = []
        
        # 1. Gradient Boosting (91.3% reported accuracy)
        gbm = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        models.append(('gbm', gbm))
        
        # 2. Random Forest (robust baseline)
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        )
        models.append(('rf', rf))
        
        # 3. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            # Ensure proper label encoding for XGBoost
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            
            xgb = XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            models.append(('xgb', xgb))
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        
        # Train and evaluate
        self.logger.info(f"   Training ensemble with {len(models)} models...")
        
        # Cross-validation on training set
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=10, scoring='accuracy')
        
        # Fit the ensemble
        ensemble.fit(X_train, y_train)
        
        results = {
            'ensemble_cv_mean': cv_scores.mean(),
            'ensemble_cv_std': cv_scores.std(),
            'models_included': [name for name, _ in models]
        }
        
        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            test_score = ensemble.score(X_test, y_test)
            results['ensemble_test_accuracy'] = test_score
            
            # Get individual model scores for comparison
            for name, model in models:
                if name == 'xgb' and XGBOOST_AVAILABLE:
                    # Use encoded labels for XGBoost
                    y_test_encoded = le.transform(y_test)
                    model.fit(X_train, y_train_encoded)
                    individual_score = model.score(X_test, y_test_encoded)
                else:
                    model.fit(X_train, y_train)
                    individual_score = model.score(X_test, y_test)
                
                results[f'{name}_test_accuracy'] = individual_score
                self.logger.info(f"   {name.upper()}: {individual_score:.3f}")
        
        self.logger.info(f"   ✅ Ensemble CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        results['model'] = ensemble
        return results
    
    def apply_correlation_based_feature_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.1) -> pd.DataFrame:
        """
        Select features based on correlation with target
        Research suggests MMSE, Age, Education, SES, eTIV, nWBV are most important
        """
        self.logger.info("🔍 Applying correlation-based feature selection...")
        
        # Calculate correlations with target
        correlations = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Handle missing values
                valid_mask = X[col].notna() & y.notna()
                if valid_mask.sum() > 10:
                    corr = X.loc[valid_mask, col].corr(y[valid_mask])
                    correlations[col] = abs(corr)
        
        # Sort by correlation strength
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Always include research-validated features if available
        important_features = ['MMSE', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'Gender_M', 'Gender_F']
        selected_features = [f for f in important_features if f in X.columns]
        
        # Add other correlated features
        for feature, corr in sorted_corrs:
            if feature not in selected_features and corr >= threshold:
                selected_features.append(feature)
        
        # Limit to top features to avoid overfitting
        max_features = min(30, len(X.columns))
        selected_features = selected_features[:max_features]
        
        self.logger.info(f"   Selected {len(selected_features)} features (from {len(X.columns)} original)")
        self.logger.info(f"   Top 5 features by correlation: {[f for f, _ in sorted_corrs[:5]]}")
        
        return X[selected_features]


def integrate_enhancements(cognitive_agent, combined_data: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate all enhancements into the cognitive analysis agent's data
    """
    enhancer = EnhancedCDRPredictor(logger=cognitive_agent.logger)
    
    # 1. Enhance brain volume features
    enhanced_data = enhancer.enhance_brain_volume_features(combined_data)
    
    # 2. Calculate atrophy rates if we have longitudinal data
    if 'Visit' in combined_data.columns:
        atrophy_df = enhancer.calculate_brain_atrophy_rate(combined_data)
        if not atrophy_df.empty:
            # Merge atrophy features
            enhanced_data = enhanced_data.merge(
                atrophy_df[['Subject_ID', 'nWBV_atrophy_rate', 'atrophy_severity']], 
                on='Subject_ID', 
                how='left'
            )
    
    cognitive_agent.logger.info(f"✅ Enhanced dataset: {enhanced_data.shape[0]} rows, {enhanced_data.shape[1]} columns")
    
    return enhanced_data