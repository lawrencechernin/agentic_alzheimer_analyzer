#!/usr/bin/env python3
"""
Enhanced Data Merging for Alzheimer's Research
==============================================

Comprehensive data merging improvements based on real-world lessons from
analyzing Brain Health Registry (BHR) MemTrax-MCI data. Prevents Cartesian
joins and handles longitudinal medical data appropriately.

Key Features:
- Pre-merge structure analysis to detect longitudinal data patterns
- Smart merge strategy selection based on data characteristics  
- Enhanced Cartesian join prevention with domain-specific warnings
- Timepoint-aware merging for medical/cognitive assessment data
- Post-merge validation to ensure data integrity

This module addresses the critical lesson learned when attempting to merge
BHR MemTrax and Medical History data, where a simple SubjectCode merge
created a Cartesian product (1.1M rows instead of expected ~60k subjects).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import warnings
from datetime import datetime


class EnhancedDataMerger:
    """
    Enhanced data merging with longitudinal awareness and Cartesian join prevention
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Domain-specific patterns that indicate longitudinal data
        self.longitudinal_indicators = [
            'timepoint', 'visit', 'wave', 'followup', 'baseline', 
            'daysafter', 'months', 'years', 'time', 'date'
        ]
        
        # Medical dataset patterns that are typically longitudinal
        self.medical_dataset_patterns = [
            'medical', 'memtrax', 'cognitive', 'assessment', 'clinical',
            'bhr', 'adni', 'oasis', 'nacc', 'brain'
        ]
        
        # Common timepoint column names
        self.timepoint_columns = [
            'TimepointCode', 'TimePoint', 'Visit', 'VisitCode', 'Wave', 
            'DaysAfterBaseline', 'Months', 'Years', 'StudyDay'
        ]
    
    def analyze_merge_structure(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              subject_col: str, df1_name: str = 'Dataset1', 
                              df2_name: str = 'Dataset2') -> Dict[str, Any]:
        """
        Analyze data structure before merging to detect potential Cartesian join risks
        
        Returns comprehensive analysis including:
        - Records per subject statistics
        - Longitudinal data detection
        - Available timepoint columns
        - Cartesian join risk assessment
        """
        self.logger.info(f"🔍 Analyzing merge structure: {df1_name} + {df2_name}")
        
        # Basic subject statistics
        records_per_subj_1 = df1.groupby(subject_col).size()
        records_per_subj_2 = df2.groupby(subject_col).size()
        
        analysis = {
            # Basic counts
            'df1_total_rows': len(df1),
            'df2_total_rows': len(df2),
            'df1_subjects': df1[subject_col].nunique(),
            'df2_subjects': df2[subject_col].nunique(),
            'overlapping_subjects': len(set(df1[subject_col]) & set(df2[subject_col])),
            
            # Records per subject analysis
            'df1_avg_records_per_subject': records_per_subj_1.mean(),
            'df2_avg_records_per_subject': records_per_subj_2.mean(),
            'df1_max_records_per_subject': records_per_subj_1.max(),
            'df2_max_records_per_subject': records_per_subj_2.max(),
            'df1_median_records_per_subject': records_per_subj_1.median(),
            'df2_median_records_per_subject': records_per_subj_2.median(),
            
            # Longitudinal detection
            'df1_is_longitudinal': records_per_subj_1.mean() > 1.5,
            'df2_is_longitudinal': records_per_subj_2.mean() > 1.5,
            'df1_highly_longitudinal': records_per_subj_1.mean() > 5.0,
            'df2_highly_longitudinal': records_per_subj_2.mean() > 5.0,
            
            # Dataset names for pattern matching
            'df1_name': df1_name,
            'df2_name': df2_name,
        }
        
        # Detect timepoint columns
        analysis['df1_timepoint_cols'] = [col for col in df1.columns 
                                        if col in self.timepoint_columns or
                                        any(indicator in col.lower() 
                                           for indicator in self.longitudinal_indicators)]
        analysis['df2_timepoint_cols'] = [col for col in df2.columns 
                                        if col in self.timepoint_columns or
                                        any(indicator in col.lower() 
                                           for indicator in self.longitudinal_indicators)]
        
        # Check for matching timepoint columns
        analysis['matching_timepoint_cols'] = list(set(analysis['df1_timepoint_cols']) & 
                                                 set(analysis['df2_timepoint_cols']))
        analysis['has_matching_timepoints'] = len(analysis['matching_timepoint_cols']) > 0
        
        # Cartesian join risk assessment
        predicted_explosion = analysis['df1_avg_records_per_subject'] * analysis['df2_avg_records_per_subject']
        analysis['predicted_records_per_subject'] = predicted_explosion
        analysis['cartesian_risk_level'] = self._assess_cartesian_risk(predicted_explosion)
        
        # Medical dataset detection
        analysis['df1_likely_medical'] = any(pattern in df1_name.lower() 
                                           for pattern in self.medical_dataset_patterns)
        analysis['df2_likely_medical'] = any(pattern in df2_name.lower() 
                                           for pattern in self.medical_dataset_patterns)
        
        # Overall risk assessment
        analysis['high_risk_merge'] = (
            analysis['cartesian_risk_level'] in ['HIGH', 'CRITICAL'] or
            (analysis['df1_is_longitudinal'] and analysis['df2_is_longitudinal'] and 
             not analysis['has_matching_timepoints'])
        )
        
        self._log_structure_analysis(analysis)
        return analysis
    
    def _assess_cartesian_risk(self, predicted_explosion: float) -> str:
        """Assess the level of Cartesian join risk"""
        if predicted_explosion > 50:
            return 'CRITICAL'
        elif predicted_explosion > 10:
            return 'HIGH'
        elif predicted_explosion > 3:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _log_structure_analysis(self, analysis: Dict[str, Any]):
        """Log the structure analysis results"""
        self.logger.info(f"   📊 {analysis['df1_name']}: {analysis['df1_total_rows']:,} rows, "
                        f"{analysis['df1_subjects']:,} subjects "
                        f"({analysis['df1_avg_records_per_subject']:.1f} avg records/subject)")
        
        self.logger.info(f"   📊 {analysis['df2_name']}: {analysis['df2_total_rows']:,} rows, "
                        f"{analysis['df2_subjects']:,} subjects "
                        f"({analysis['df2_avg_records_per_subject']:.1f} avg records/subject)")
        
        self.logger.info(f"   🔗 Overlapping subjects: {analysis['overlapping_subjects']:,}")
        
        if analysis['has_matching_timepoints']:
            self.logger.info(f"   ⏰ Matching timepoint columns: {analysis['matching_timepoint_cols']}")
        else:
            self.logger.warning(f"   ⚠️  No matching timepoint columns found")
        
        risk_level = analysis['cartesian_risk_level']
        if risk_level == 'CRITICAL':
            self.logger.error(f"   🚨 CRITICAL CARTESIAN RISK: ~{analysis['predicted_records_per_subject']:.1f}x explosion predicted!")
        elif risk_level == 'HIGH':
            self.logger.warning(f"   ⚠️  HIGH CARTESIAN RISK: ~{analysis['predicted_records_per_subject']:.1f}x explosion predicted")
        elif risk_level == 'MODERATE':
            self.logger.info(f"   ⚠️  Moderate cartesian risk: ~{analysis['predicted_records_per_subject']:.1f}x growth expected")
        else:
            self.logger.info(f"   ✅ Low cartesian risk: ~{analysis['predicted_records_per_subject']:.1f}x growth expected")
    
    def select_merge_strategy(self, structure_analysis: Dict[str, Any]) -> str:
        """
        Automatically select the best merge strategy based on data structure analysis
        
        Returns:
            'timepoint_match': Use SubjectCode + TimepointCode
            'latest_per_subject': Take latest record per subject, then merge on SubjectCode
            'dedupe_first': Deduplicate both datasets, then merge on SubjectCode  
            'simple_merge': Safe to merge on SubjectCode only
            'abort_merge': Too risky to merge - manual intervention required
        """
        analysis = structure_analysis
        
        # CRITICAL risk - abort merge
        if analysis['cartesian_risk_level'] == 'CRITICAL':
            return 'abort_merge'
        
        # High risk but have matching timepoints - use timepoint matching
        if analysis['high_risk_merge'] and analysis['has_matching_timepoints']:
            return 'timepoint_match'
        
        # High risk without timepoint matching - use latest per subject
        if analysis['high_risk_merge'] and not analysis['has_matching_timepoints']:
            if analysis['df1_is_longitudinal'] or analysis['df2_is_longitudinal']:
                return 'latest_per_subject'
            else:
                return 'dedupe_first'
        
        # Moderate risk - deduplicate first
        if analysis['cartesian_risk_level'] == 'MODERATE':
            return 'dedupe_first'
        
        # Low risk - simple merge is safe
        return 'simple_merge'
    
    def execute_smart_merge(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          subject_col: str, df1_name: str = 'Dataset1', 
                          df2_name: str = 'Dataset2') -> pd.DataFrame:
        """
        Execute smart merge using the optimal strategy based on data structure
        """
        # Step 1: Analyze structure
        analysis = self.analyze_merge_structure(df1, df2, subject_col, df1_name, df2_name)
        
        # Step 2: Select strategy
        strategy = self.select_merge_strategy(analysis)
        
        self.logger.info(f"🎯 Selected merge strategy: {strategy}")
        
        # Step 3: Execute merge based on strategy
        if strategy == 'abort_merge':
            self.logger.error("🚨 MERGE ABORTED - Too high Cartesian join risk!")
            self.logger.error("   Manual intervention required:")
            self.logger.error("   1. Check for duplicate subject IDs")
            self.logger.error("   2. Consider using timepoint-specific merging")
            self.logger.error("   3. Pre-filter to specific time periods")
            raise ValueError(f"Merge aborted due to critical Cartesian join risk "
                           f"(predicted {analysis['predicted_records_per_subject']:.1f}x explosion)")
        
        elif strategy == 'timepoint_match':
            return self._merge_with_timepoints(df1, df2, subject_col, analysis)
        
        elif strategy == 'latest_per_subject':
            return self._merge_latest_per_subject(df1, df2, subject_col, analysis)
        
        elif strategy == 'dedupe_first':
            return self._merge_with_deduplication(df1, df2, subject_col, analysis)
        
        else:  # simple_merge
            return self._simple_merge_with_validation(df1, df2, subject_col, analysis)
    
    def _merge_with_timepoints(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             subject_col: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Merge using SubjectCode + TimepointCode"""
        timepoint_col = analysis['matching_timepoint_cols'][0]  # Use first matching column
        
        self.logger.info(f"   🔗 Merging on [{subject_col}, {timepoint_col}]")
        
        merge_keys = [subject_col, timepoint_col]
        result = df1.merge(df2, on=merge_keys, how='inner', suffixes=('', '_df2'))
        
        # Validate the merge
        self._validate_merge_result(result, analysis, subject_col, 'timepoint_match')
        
        return result
    
    def _merge_latest_per_subject(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                subject_col: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Take latest record per subject, then merge"""
        self.logger.info("   📅 Taking latest record per subject before merge...")
        
        # Get latest record for each dataset
        df1_latest = self._get_latest_per_subject(df1, subject_col, analysis['df1_timepoint_cols'])
        df2_latest = self._get_latest_per_subject(df2, subject_col, analysis['df2_timepoint_cols'])
        
        self.logger.info(f"   📊 {analysis['df1_name']}: {len(df1):,} → {len(df1_latest):,} records")
        self.logger.info(f"   📊 {analysis['df2_name']}: {len(df2):,} → {len(df2_latest):,} records")
        
        # Now merge the latest records
        result = df1_latest.merge(df2_latest, on=subject_col, how='inner', suffixes=('', '_df2'))
        
        # Update analysis for validation
        updated_analysis = analysis.copy()
        updated_analysis['df1_total_rows'] = len(df1_latest)
        updated_analysis['df2_total_rows'] = len(df2_latest)
        updated_analysis['predicted_records_per_subject'] = 1.0  # Should be 1:1 now
        
        self._validate_merge_result(result, updated_analysis, subject_col, 'latest_per_subject')
        
        return result
    
    def _get_latest_per_subject(self, df: pd.DataFrame, subject_col: str, 
                               timepoint_cols: List[str]) -> pd.DataFrame:
        """Get the latest record for each subject"""
        if not timepoint_cols:
            # No timepoint columns - just deduplicate
            return df.drop_duplicates(subset=[subject_col], keep='last')
        
        # Use the first available timepoint column
        timepoint_col = timepoint_cols[0]
        
        # Sort by subject and timepoint, then take last record per subject
        if df[timepoint_col].dtype in ['int64', 'float64']:
            # Numeric timepoint - sort ascending
            df_sorted = df.sort_values([subject_col, timepoint_col], na_position='first')
        else:
            # Try to convert to datetime, fallback to string sort
            try:
                df[f'{timepoint_col}_datetime'] = pd.to_datetime(df[timepoint_col], errors='coerce')
                df_sorted = df.sort_values([subject_col, f'{timepoint_col}_datetime'], na_position='first')
                df_sorted = df_sorted.drop(columns=[f'{timepoint_col}_datetime'])
            except:
                df_sorted = df.sort_values([subject_col, timepoint_col], na_position='first')
        
        return df_sorted.groupby(subject_col).tail(1)
    
    def _merge_with_deduplication(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                subject_col: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Deduplicate both datasets before merging"""
        self.logger.info("   🧹 Deduplicating datasets before merge...")
        
        # Deduplicate both datasets
        df1_dedup = df1.drop_duplicates(subset=[subject_col], keep='last')
        df2_dedup = df2.drop_duplicates(subset=[subject_col], keep='last')
        
        self.logger.info(f"   📊 {analysis['df1_name']}: {len(df1):,} → {len(df1_dedup):,} records")
        self.logger.info(f"   📊 {analysis['df2_name']}: {len(df2):,} → {len(df2_dedup):,} records")
        
        # Merge deduplicated datasets
        result = df1_dedup.merge(df2_dedup, on=subject_col, how='inner', suffixes=('', '_df2'))
        
        # Update analysis for validation
        updated_analysis = analysis.copy()
        updated_analysis['df1_total_rows'] = len(df1_dedup)
        updated_analysis['df2_total_rows'] = len(df2_dedup)
        updated_analysis['predicted_records_per_subject'] = 1.0  # Should be 1:1 now
        
        self._validate_merge_result(result, updated_analysis, subject_col, 'dedupe_first')
        
        return result
    
    def _simple_merge_with_validation(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                    subject_col: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Simple merge with validation"""
        self.logger.info("   🔗 Performing simple merge (low risk detected)")
        
        result = df1.merge(df2, on=subject_col, how='inner', suffixes=('', '_df2'))
        
        self._validate_merge_result(result, analysis, subject_col, 'simple_merge')
        
        return result
    
    def _validate_merge_result(self, result: pd.DataFrame, analysis: Dict[str, Any],
                             subject_col: str, strategy: str):
        """Validate merge result to ensure no Cartesian join occurred"""
        final_rows = len(result)
        final_subjects = result[subject_col].nunique()
        final_records_per_subject = final_rows / final_subjects if final_subjects > 0 else 0
        
        # Expected subjects (intersection of input datasets)
        expected_subjects = analysis['overlapping_subjects']
        
        self.logger.info(f"   ✅ Merge complete using {strategy} strategy:")
        self.logger.info(f"      Final rows: {final_rows:,}")
        self.logger.info(f"      Final subjects: {final_subjects:,}")
        self.logger.info(f"      Records per subject: {final_records_per_subject:.2f}")
        
        # Validation checks
        if final_subjects > expected_subjects * 1.1:
            self.logger.error(f"   🚨 VALIDATION FAILED: Too many subjects ({final_subjects} > {expected_subjects})")
            raise ValueError("Subject count validation failed - possible duplicate entries")
        
        if final_records_per_subject > 10:
            self.logger.error(f"   🚨 VALIDATION FAILED: Too many records per subject ({final_records_per_subject:.1f})")
            raise ValueError("Records per subject validation failed - possible Cartesian join")
        
        if final_records_per_subject > 3:
            self.logger.warning(f"   ⚠️  High records per subject ({final_records_per_subject:.1f}) - review merge logic")
        
        # Success
        if final_subjects == expected_subjects and final_records_per_subject <= 2:
            self.logger.info(f"   🎉 MERGE VALIDATION PASSED - Clean merge achieved!")
        else:
            self.logger.info(f"   ✅ MERGE VALIDATION PASSED - Acceptable merge result")


def create_enhanced_merger(logger: Optional[logging.Logger] = None) -> EnhancedDataMerger:
    """Factory function to create an enhanced data merger"""
    return EnhancedDataMerger(logger=logger)


def smart_merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, subject_col: str,
                        df1_name: str = 'Dataset1', df2_name: str = 'Dataset2',
                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Convenience function for smart dataset merging with Cartesian join prevention
    
    This is the main entry point for enhanced merging based on BHR lesson learned.
    """
    merger = EnhancedDataMerger(logger=logger)
    return merger.execute_smart_merge(df1, df2, subject_col, df1_name, df2_name)
