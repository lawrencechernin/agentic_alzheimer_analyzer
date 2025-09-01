#!/usr/bin/env python3
"""
Modular Feature Engineering Pipeline for Alzheimer's Research
Configurable feature engineering with domain-specific modules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

class FeatureEngineeringPipeline:
    """
    Modular feature engineering pipeline with Alzheimer's-specific modules
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Register available feature engineering modules
        self.modules = {
            'alzheimer_brain_features': self._alzheimer_brain_features,
            'cognitive_assessment_features': self._cognitive_assessment_features,
            'age_stratified_features': self._age_stratified_features,
            'interaction_features': self._interaction_features,
            'temporal_features': self._temporal_features,
            'risk_score_features': self._risk_score_features,
            'biomarker_ratios': self._biomarker_ratios,
            'demographic_features': self._demographic_features
        }
    
    def apply_pipeline(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Apply configured feature engineering pipeline
        """
        enhanced_df = df.copy()
        applied_modules = []
        
        # Get enabled modules from config
        enabled_modules = self.config.get('feature_engineering', {}).get('modules', [])
        
        # Auto-detect if no modules specified
        if not enabled_modules:
            enabled_modules = self._auto_detect_modules(df)
        
        self.logger.info(f"🔧 Applying {len(enabled_modules)} feature engineering modules...")
        
        for module_name in enabled_modules:
            if module_name in self.modules:
                try:
                    self.logger.info(f"   Applying {module_name}...")
                    enhanced_df = self.modules[module_name](enhanced_df, target_col)
                    applied_modules.append(module_name)
                except Exception as e:
                    self.logger.warning(f"   Failed to apply {module_name}: {e}")
        
        original_cols = len(df.columns)
        new_cols = len(enhanced_df.columns)
        self.logger.info(f"✅ Feature engineering complete: {original_cols} → {new_cols} features")
        
        return enhanced_df
    
    def _auto_detect_modules(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect which modules to apply based on available columns
        """
        modules = []
        columns_lower = [col.lower() for col in df.columns]
        
        # Brain imaging features
        brain_indicators = ['etiv', 'nwbv', 'asf', 'volume', 'cortical', 'hippocampal']
        if any(indicator in ' '.join(columns_lower) for indicator in brain_indicators):
            modules.append('alzheimer_brain_features')
            modules.append('biomarker_ratios')
        
        # Cognitive assessment features
        cognitive_indicators = ['mmse', 'adas', 'cdr', 'moca', 'cognitive', 'memory']
        if any(indicator in ' '.join(columns_lower) for indicator in cognitive_indicators):
            modules.append('cognitive_assessment_features')
        
        # Age-related features
        if 'age' in columns_lower:
            modules.append('age_stratified_features')
        
        # Demographics
        demo_indicators = ['gender', 'education', 'ses', 'race', 'ethnicity']
        if any(indicator in ' '.join(columns_lower) for indicator in demo_indicators):
            modules.append('demographic_features')
        
        # Temporal data
        temporal_indicators = ['visit', 'date', 'time', 'delay', 'followup']
        if any(indicator in ' '.join(columns_lower) for indicator in temporal_indicators):
            modules.append('temporal_features')
        
        # Always add interaction and risk features if we have multiple predictors
        if len([col for col in columns_lower if col not in ['subject_id', 'id']]) > 3:
            modules.extend(['interaction_features', 'risk_score_features'])
        
        self.logger.info(f"📊 Auto-detected modules: {modules}")
        return modules
    
    def _alzheimer_brain_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create Alzheimer's-specific brain imaging features
        """
        enhanced_df = df.copy()
        
        # Brain volume features
        if 'eTIV' in df.columns and 'nWBV' in df.columns:
            # Brain atrophy index
            enhanced_df['brain_atrophy_index'] = 1 - df['nWBV']
            
            # Volume preservation ratio
            enhanced_df['volume_preservation_ratio'] = df['nWBV'] / (df['eTIV'] / 1500)
            
            # Brain volume per unit body size (if ASF available)
            if 'ASF' in df.columns:
                enhanced_df['normalized_brain_volume'] = df['eTIV'] * df['ASF']
        
        # ASF-eTIV relationship validation
        if 'ASF' in df.columns and 'eTIV' in df.columns:
            enhanced_df['asf_etiv_product'] = df['ASF'] * df['eTIV']
            
        return enhanced_df
    
    def _cognitive_assessment_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create cognitive assessment-specific features
        """
        enhanced_df = df.copy()
        
        # MMSE-based features
        if 'MMSE' in df.columns:
            # MMSE severity categories
            enhanced_df['mmse_severity'] = pd.cut(df['MMSE'], 
                bins=[0, 10, 20, 24, 27, 30], 
                labels=['severe', 'moderate', 'mild', 'normal', 'high'])
            
            # Age-adjusted MMSE (if age available)
            if 'Age' in df.columns:
                # Expected decline: ~0.3 points per year after 60
                expected_mmse = 30 - np.maximum(0, (df['Age'] - 60) * 0.3)
                enhanced_df['mmse_age_deviation'] = df['MMSE'] - expected_mmse
            
            # Education-adjusted MMSE
            if 'EDUC' in df.columns or 'Education' in df.columns:
                educ_col = 'EDUC' if 'EDUC' in df.columns else 'Education'
                enhanced_df['mmse_per_education_year'] = df['MMSE'] / (df[educ_col] + 1)
        
        # CDR-based features
        if 'CDR' in df.columns:
            enhanced_df['cdr_binary'] = (df['CDR'] > 0).astype(int)
            enhanced_df['cdr_severity_group'] = pd.cut(df['CDR'], 
                bins=[-0.1, 0, 0.5, 2.0], 
                labels=['normal', 'questionable', 'impaired'])
        
        return enhanced_df
    
    def _age_stratified_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create age-stratified features important in Alzheimer's research
        """
        enhanced_df = df.copy()
        
        if 'Age' in df.columns:
            # Age groups relevant to Alzheimer's
            enhanced_df['age_group'] = pd.cut(df['Age'],
                bins=[0, 65, 75, 85, 120],
                labels=['young', 'early_senior', 'senior', 'advanced_senior'])
            
            # Age squared (nonlinear relationship)
            enhanced_df['age_squared'] = df['Age'] ** 2
            
            # Age relative to typical onset
            enhanced_df['age_relative_to_onset'] = df['Age'] - 65
            
            # High-risk age indicator
            enhanced_df['high_risk_age'] = (df['Age'] >= 75).astype(int)
        
        return enhanced_df
    
    def _interaction_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create important interaction features for Alzheimer's research
        """
        enhanced_df = df.copy()
        
        # Age interactions
        if 'Age' in df.columns:
            if 'MMSE' in df.columns:
                enhanced_df['age_mmse_interaction'] = df['Age'] * df['MMSE']
            
            if 'nWBV' in df.columns:
                enhanced_df['age_brain_volume_interaction'] = df['Age'] * df['nWBV']
            
            if 'EDUC' in df.columns or 'Education' in df.columns:
                educ_col = 'EDUC' if 'EDUC' in df.columns else 'Education'
                enhanced_df['age_education_interaction'] = df['Age'] * df[educ_col]
        
        # Gender interactions
        gender_cols = [col for col in df.columns if 'gender' in col.lower() or col in ['Gender_M', 'Gender_F']]
        if gender_cols and 'nWBV' in df.columns:
            gender_col = gender_cols[0]
            enhanced_df['gender_brain_volume_interaction'] = df[gender_col] * df['nWBV']
        
        return enhanced_df
    
    def _temporal_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create temporal features for longitudinal data
        """
        enhanced_df = df.copy()
        
        # Visit-based features
        if 'Visit' in df.columns:
            enhanced_df['visit_squared'] = df['Visit'] ** 2
            enhanced_df['is_baseline'] = (df['Visit'] == 1).astype(int)
            enhanced_df['is_followup'] = (df['Visit'] > 1).astype(int)
        
        # Time delay features
        if 'MR Delay' in df.columns:
            enhanced_df['mr_delay_months'] = df['MR Delay'] / 30.44
            enhanced_df['long_interval'] = (df['MR Delay'] > 400).astype(int)
        
        return enhanced_df
    
    def _risk_score_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create composite risk scores
        """
        enhanced_df = df.copy()
        
        # Basic Alzheimer's risk score
        risk_score = 0
        risk_factors_used = []
        
        if 'Age' in df.columns:
            risk_score += (df['Age'] > 75).astype(int)
            risk_factors_used.append('age')
        
        if 'MMSE' in df.columns:
            risk_score += (df['MMSE'] < 24).astype(int)
            risk_factors_used.append('mmse')
        
        if 'nWBV' in df.columns:
            median_volume = df['nWBV'].median()
            risk_score += (df['nWBV'] < median_volume).astype(int)
            risk_factors_used.append('brain_volume')
        
        if 'EDUC' in df.columns or 'Education' in df.columns:
            educ_col = 'EDUC' if 'EDUC' in df.columns else 'Education'
            risk_score += (df[educ_col] < 12).astype(int)
            risk_factors_used.append('education')
        
        enhanced_df['alzheimer_risk_score'] = risk_score
        enhanced_df['risk_factors_count'] = len(risk_factors_used)
        
        return enhanced_df
    
    def _biomarker_ratios(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create biomarker ratios important in Alzheimer's research
        """
        enhanced_df = df.copy()
        
        # Brain volume ratios
        if 'eTIV' in df.columns and 'nWBV' in df.columns:
            enhanced_df['brain_volume_ratio'] = df['nWBV'] / (df['eTIV'] / 1000)
        
        # Cognitive-to-brain ratios
        if 'MMSE' in df.columns and 'nWBV' in df.columns:
            enhanced_df['cognitive_brain_ratio'] = df['MMSE'] / (df['nWBV'] * 100)
        
        return enhanced_df
    
    def _demographic_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Create demographic features relevant to Alzheimer's research
        """
        enhanced_df = df.copy()
        
        # Education categories
        if 'EDUC' in df.columns or 'Education' in df.columns:
            educ_col = 'EDUC' if 'EDUC' in df.columns else 'Education'
            enhanced_df['education_level'] = pd.cut(df[educ_col],
                bins=[0, 8, 12, 16, 25],
                labels=['low', 'medium', 'high', 'very_high'])
            
            # Cognitive reserve proxy
            enhanced_df['cognitive_reserve'] = (df[educ_col] >= 16).astype(int)
        
        # SES features
        if 'SES' in df.columns:
            enhanced_df['low_ses'] = (df['SES'] <= 2).astype(int)
            enhanced_df['high_ses'] = (df['SES'] >= 4).astype(int)
        
        return enhanced_df


def create_feature_config(dataset_type: str = 'alzheimer') -> Dict[str, Any]:
    """
    Create feature engineering configuration for different dataset types
    """
    configs = {
        'alzheimer': {
            'feature_engineering': {
                'modules': [
                    'alzheimer_brain_features',
                    'cognitive_assessment_features',
                    'age_stratified_features',
                    'interaction_features',
                    'risk_score_features',
                    'biomarker_ratios',
                    'demographic_features'
                ]
            }
        },
        'mild_cognitive_impairment': {
            'feature_engineering': {
                'modules': [
                    'cognitive_assessment_features',
                    'age_stratified_features',
                    'interaction_features',
                    'risk_score_features'
                ]
            }
        },
        'longitudinal': {
            'feature_engineering': {
                'modules': [
                    'alzheimer_brain_features',
                    'cognitive_assessment_features',
                    'temporal_features',
                    'age_stratified_features',
                    'interaction_features',
                    'risk_score_features'
                ]
            }
        }
    }
    
    return configs.get(dataset_type, configs['alzheimer'])


if __name__ == "__main__":
    # Test the pipeline
    print("🧪 Testing Modular Feature Engineering Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'Age': np.random.normal(70, 10, n_samples),
        'MMSE': np.random.normal(25, 5, n_samples),
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.1, n_samples),
        'EDUC': np.random.normal(12, 4, n_samples),
        'SES': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Gender_M': np.random.choice([0, 1], n_samples),
        'CDR': np.random.choice([0, 0.5, 1], n_samples)
    })
    
    # Test pipeline
    config = create_feature_config('alzheimer')
    pipeline = FeatureEngineeringPipeline(config)
    
    enhanced_data = pipeline.apply_pipeline(sample_data, 'CDR')
    
    print(f"✅ Pipeline test complete!")
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Enhanced features: {len(enhanced_data.columns)}")
    print(f"New features added: {len(enhanced_data.columns) - len(sample_data.columns)}")