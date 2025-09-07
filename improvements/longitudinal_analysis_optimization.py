#!/usr/bin/env python3
"""
Longitudinal Analysis Optimization
===================================

Implements advanced strategies for analyzing longitudinal medical data,
based on learnings from BHR MemTrax analysis achieving 0.7+ AUC.

Key Insights:
1. Aggregate across ALL timepoints, not just baseline
2. Create composite scores (e.g., RT/accuracy)
3. Apply quality filters before feature extraction
4. Extract variability and progression features
5. Use subject-level aggregation for stronger signal

This module transforms the agentic analyzer from single-timepoint
analysis (0.59 AUC) to robust longitudinal analysis (0.70+ AUC).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats


class LongitudinalOptimizer:
    """
    Optimizes analysis of longitudinal medical data through intelligent
    aggregation and feature engineering.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Domain-specific composite scores
        self.composite_scores = {
            'cognitive_efficiency': lambda df: df['reaction_time'] / (df['accuracy'] + 0.01),
            'speed_accuracy_tradeoff': lambda df: df['correct_rt'] / (df['correct_pct'] + 0.01),
            'error_consistency': lambda df: df['error_std'] / (df['error_mean'] + 1e-6),
            'performance_stability': lambda df: 1 / (df['performance_std'] + 1e-6)
        }
        
        # Quality criteria for different test types
        self.quality_criteria = {
            'memtrax': {
                'Status': 'Collected',
                'CorrectPCT': (0.60, 1.0),
                'CorrectResponsesRT': (0.5, 2.5),
                'required': ['IncorrectRejectionsN']
            },
            'mmse': {
                'score': (0, 30),
                'completion': 0.80
            },
            'cdr': {
                'score': (0, 3),
                'sob': (0, 18)
            }
        }
    
    def detect_longitudinal_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if dataset has longitudinal structure and analyze it
        """
        self.logger.info("🔍 Detecting longitudinal structure...")
        
        # Find subject identifier column
        subject_cols = ['SubjectCode', 'SubjectID', 'ParticipantID', 'subject_id', 'ID']
        subject_col = None
        for col in subject_cols:
            if col in df.columns:
                subject_col = col
                break
        
        if not subject_col:
            return {'is_longitudinal': False, 'reason': 'No subject identifier found'}
        
        # Find timepoint column
        timepoint_cols = ['TimepointCode', 'Timepoint', 'Visit', 'Wave', 'time', 'TimePoint']
        timepoint_col = None
        for col in timepoint_cols:
            if col in df.columns:
                timepoint_col = col
                break
        
        # Calculate structure metrics
        unique_subjects = df[subject_col].nunique()
        total_records = len(df)
        records_per_subject = total_records / unique_subjects
        
        # Count subjects with multiple records
        subject_counts = df.groupby(subject_col).size()
        multi_record_subjects = (subject_counts > 1).sum()
        
        is_longitudinal = records_per_subject > 1.5 or multi_record_subjects > unique_subjects * 0.3
        
        structure = {
            'is_longitudinal': is_longitudinal,
            'subject_col': subject_col,
            'timepoint_col': timepoint_col,
            'unique_subjects': unique_subjects,
            'total_records': total_records,
            'records_per_subject': records_per_subject,
            'subjects_with_1_record': (subject_counts == 1).sum(),
            'subjects_with_2plus_records': (subject_counts >= 2).sum(),
            'subjects_with_3plus_records': (subject_counts >= 3).sum(),
            'max_records_per_subject': subject_counts.max()
        }
        
        self.logger.info(f"   📊 Structure: {total_records:,} records, {unique_subjects:,} subjects")
        self.logger.info(f"   📈 Avg records/subject: {records_per_subject:.2f}")
        self.logger.info(f"   🔄 Longitudinal: {'Yes' if is_longitudinal else 'No'}")
        
        return structure
    
    def apply_quality_filters(self, df: pd.DataFrame, test_type: str = 'auto') -> pd.DataFrame:
        """
        Apply domain-specific quality filters
        """
        self.logger.info(f"✅ Applying quality filters (type: {test_type})...")
        
        # Auto-detect test type if needed
        if test_type == 'auto':
            if 'CorrectPCT' in df.columns and 'CorrectResponsesRT' in df.columns:
                test_type = 'memtrax'
            elif 'MMSE' in df.columns or 'mmse_total' in df.columns:
                test_type = 'mmse'
            elif 'CDR' in df.columns or 'cdr_global' in df.columns:
                test_type = 'cdr'
        
        if test_type not in self.quality_criteria:
            self.logger.warning(f"   ⚠️ No quality criteria for test type: {test_type}")
            return df
        
        criteria = self.quality_criteria[test_type]
        df_filtered = df.copy()
        initial_count = len(df_filtered)
        
        # Apply filters
        for field, criterion in criteria.items():
            if field == 'required':
                for req_field in criterion:
                    if req_field in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered[req_field].notna()]
            elif field in df_filtered.columns:
                if isinstance(criterion, tuple):
                    df_filtered = df_filtered[
                        df_filtered[field].between(criterion[0], criterion[1])
                    ]
                elif isinstance(criterion, str):
                    df_filtered = df_filtered[df_filtered[field] == criterion]
        
        filtered_count = initial_count - len(df_filtered)
        self.logger.info(f"   ✅ Filtered {filtered_count:,} records ({filtered_count/initial_count*100:.1f}%)")
        self.logger.info(f"   📊 Remaining: {len(df_filtered):,} records")
        
        return df_filtered
    
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific composite scores
        """
        self.logger.info("🧠 Creating composite scores...")
        
        df_enhanced = df.copy()
        
        # Cognitive efficiency score (key feature from BHR analysis)
        if 'CorrectResponsesRT' in df.columns and 'CorrectPCT' in df.columns:
            df_enhanced['CognitiveScore'] = df['CorrectResponsesRT'] / (df['CorrectPCT'] + 0.01)
            self.logger.info("   ✅ Added CognitiveScore (RT/accuracy)")
        
        # Speed-accuracy tradeoff
        if 'correct_rt' in df.columns and 'accuracy' in df.columns:
            df_enhanced['SpeedAccuracyTradeoff'] = df['correct_rt'] / (df['accuracy'] + 0.01)
            self.logger.info("   ✅ Added SpeedAccuracyTradeoff")
        
        # Performance consistency
        if 'correct_std' in df.columns and 'correct_mean' in df.columns:
            df_enhanced['PerformanceConsistency'] = 1 / (df['correct_std'] / (df['correct_mean'] + 1e-6) + 1e-6)
            self.logger.info("   ✅ Added PerformanceConsistency")
        
        return df_enhanced
    
    def aggregate_longitudinal_features(
        self, 
        df: pd.DataFrame, 
        subject_col: str,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate features across timepoints per subject
        """
        self.logger.info("📈 Aggregating longitudinal features...")
        
        if exclude_cols is None:
            exclude_cols = [subject_col, 'TimepointCode', 'Date', 'Status']
        
        # Select numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_aggregate = [col for col in numeric_cols if col not in exclude_cols]
        
        # Define aggregation functions
        agg_funcs = {}
        for col in cols_to_aggregate:
            agg_funcs[col] = ['mean', 'std', 'min', 'max']
            
            # Add trend for temporal columns
            if 'rt' in col.lower() or 'time' in col.lower() or 'score' in col.lower():
                agg_funcs[col].append(lambda x: self._calculate_trend(x))
        
        # Add test count
        first_col = df.columns[0]
        agg_funcs[first_col] = 'count'
        
        # Perform aggregation
        df_aggregated = df.groupby(subject_col).agg(agg_funcs)
        
        # Flatten column names
        df_aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in df_aggregated.columns]
        
        # Rename count column
        count_col = f'{first_col}_count'
        if count_col in df_aggregated.columns:
            df_aggregated.rename(columns={count_col: 'TestCount'}, inplace=True)
        
        # Add coefficient of variation for key features
        for col in cols_to_aggregate:
            if f'{col}_mean' in df_aggregated.columns and f'{col}_std' in df_aggregated.columns:
                df_aggregated[f'{col}_cv'] = (
                    df_aggregated[f'{col}_std'] / (df_aggregated[f'{col}_mean'] + 1e-6)
                )
        
        # Add progression features (max - min)
        for col in cols_to_aggregate:
            if f'{col}_max' in df_aggregated.columns and f'{col}_min' in df_aggregated.columns:
                df_aggregated[f'{col}_range'] = df_aggregated[f'{col}_max'] - df_aggregated[f'{col}_min']
        
        df_aggregated = df_aggregated.reset_index()
        
        self.logger.info(f"   ✅ Aggregated to {len(df_aggregated):,} subjects")
        self.logger.info(f"   📊 Features: {len(cols_to_aggregate)} → {len(df_aggregated.columns)-1}")
        
        return df_aggregated
    
    def _calculate_trend(self, values):
        """Calculate linear trend over time"""
        if len(values) < 2:
            return 0
        try:
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        except:
            return 0
    
    def optimize_for_prediction(
        self, 
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete optimization pipeline for longitudinal data
        """
        self.logger.info("\n🚀 LONGITUDINAL OPTIMIZATION PIPELINE")
        self.logger.info("=" * 50)
        
        # Detect structure
        structure = self.detect_longitudinal_structure(df)
        
        if not structure['is_longitudinal']:
            self.logger.warning("   ⚠️ Not longitudinal data - returning original")
            return df, structure
        
        # Apply quality filters
        df_filtered = self.apply_quality_filters(df)
        
        # Create composite scores
        df_enhanced = self.create_composite_scores(df_filtered)
        
        # Aggregate if longitudinal
        df_final = self.aggregate_longitudinal_features(
            df_enhanced, 
            structure['subject_col']
        )
        
        # Report improvement
        original_features = len(df.columns)
        final_features = len(df_final.columns)
        feature_increase = (final_features / original_features - 1) * 100
        
        optimization_report = {
            'structure': structure,
            'original_shape': df.shape,
            'final_shape': df_final.shape,
            'feature_increase_pct': feature_increase,
            'quality_filtered': len(df) - len(df_filtered),
            'subjects_retained': len(df_final)
        }
        
        self.logger.info(f"\n✅ OPTIMIZATION COMPLETE")
        self.logger.info(f"   Original: {df.shape[0]:,} × {df.shape[1]} features")
        self.logger.info(f"   Final: {df_final.shape[0]:,} × {df_final.shape[1]} features")
        self.logger.info(f"   Feature increase: {feature_increase:+.0f}%")
        
        return df_final, optimization_report


def integrate_with_cognitive_agent():
    """
    Integration code for CognitiveAnalysisAgent
    """
    integration_code = '''
    # Add to CognitiveAnalysisAgent.__init__:
    from improvements.longitudinal_analysis_optimization import LongitudinalOptimizer
    self.longitudinal_optimizer = LongitudinalOptimizer(logger=self.logger)
    
    # Modify data loading to use optimization:
    def _load_and_prepare_data(self):
        # Original loading...
        data = self._load_data()
        
        # Apply longitudinal optimization
        if self.longitudinal_optimizer:
            data_optimized, report = self.longitudinal_optimizer.optimize_for_prediction(data)
            if report['structure']['is_longitudinal']:
                self.logger.info(f"Applied longitudinal optimization: {report['feature_increase_pct']:+.0f}% more features")
                return data_optimized
        
        return data
    '''
    
    return integration_code


if __name__ == "__main__":
    print("Longitudinal Analysis Optimization Module")
    print("=" * 50)
    print("\nKey Features:")
    print("✅ Automatic longitudinal structure detection")
    print("✅ Quality filtering (Ashford criteria)")
    print("✅ Composite score creation (RT/accuracy)")
    print("✅ Multi-statistic aggregation (mean, std, min, max, trend)")
    print("✅ Progression and variability features")
    print("\nExpected Impact:")
    print("📈 AUC improvement: 0.59 → 0.70+ (based on BHR analysis)")
    print("📊 Feature richness: 4-5x increase through aggregation")
    print("🎯 Better signal: Noise reduction through multi-test averaging") 