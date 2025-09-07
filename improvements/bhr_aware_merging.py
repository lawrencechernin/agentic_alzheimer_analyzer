#!/usr/bin/env python3
"""
BHR-Aware Data Merging Strategy
================================

Enhanced merging strategy specifically designed for Brain Health Registry (BHR) 
and similar longitudinal medical datasets, based on successful patterns from
production analysis scripts.

Key Learnings Incorporated:
1. Timepoint-first filtering (filter BEFORE merging)
2. Data quality pre-filtering (Status == 'Collected', RT ranges)
3. Sequential merge pattern (demographics first, then medical)
4. Domain-specific timepoint recognition (m00, m06, m12, etc.)

This module prevents the Cartesian join issues that occurred when naively
merging BHR MemTrax and Medical History data (1.1M rows from ~60k subjects).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import re


class BHRAwareMerger:
    """
    Implements BHR-specific merging strategies learned from successful analyses
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # BHR-specific timepoint patterns
        self.bhr_timepoint_pattern = re.compile(r'^m\d{2}$')  # m00, m06, m12, etc.
        
        # Common timepoint column names in medical datasets
        self.timepoint_columns = [
            'TimepointCode', 'Timepoint', 'Visit', 'VisitCode',
            'Wave', 'TimePoint', 'visitcode', 'timepoint_code'
        ]
        
        # Quality filter columns for cognitive tests
        self.quality_columns = {
            'Status': 'Collected',  # BHR-specific
            'TestStatus': 'Complete',
            'DataQuality': 'Pass'
        }
        
        # Cognitive test quality thresholds
        self.cognitive_thresholds = {
            'CorrectPCT': (0.60, 1.0),  # Min 60% accuracy
            'CorrectResponsesRT': (0.5, 2.5),  # Reasonable RT range
            'accuracy': (0.5, 1.0),
            'reaction_time': (200, 3000)  # milliseconds
        }
        
    def detect_bhr_dataset(self, df: pd.DataFrame, dataset_name: str = "") -> bool:
        """
        Detect if this is a BHR or BHR-like dataset
        """
        # Check for BHR-specific columns
        bhr_indicators = [
            'SubjectCode', 'TimepointCode', 'ParticipantID',
            'CorrectResponsesRT', 'IncorrectResponsesRT',
            'QID1-13', 'QID1-12', 'QID1-5'  # Medical history QIDs
        ]
        
        found_indicators = sum(1 for col in bhr_indicators if col in df.columns)
        
        # Check for BHR timepoint pattern
        has_bhr_timepoints = False
        for col in self.timepoint_columns:
            if col in df.columns:
                sample_values = df[col].dropna().head(100)
                bhr_matches = sample_values.apply(
                    lambda x: bool(self.bhr_timepoint_pattern.match(str(x)))
                ).sum()
                if bhr_matches > len(sample_values) * 0.5:
                    has_bhr_timepoints = True
                    break
        
        # Dataset name check
        name_is_bhr = 'bhr' in dataset_name.lower() or 'brain health' in dataset_name.lower()
        
        return found_indicators >= 2 or has_bhr_timepoints or name_is_bhr
    
    def apply_cognitive_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality filters commonly used in successful BHR analyses
        """
        df_filtered = df.copy()
        initial_count = len(df_filtered)
        
        # Apply status filters
        for col, expected_value in self.quality_columns.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == expected_value]
                self.logger.info(f"   Filtered {col} == '{expected_value}': {len(df_filtered):,} rows remain")
        
        # Apply threshold filters
        for col, (min_val, max_val) in self.cognitive_thresholds.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered[col] >= min_val) & 
                    (df_filtered[col] <= max_val)
                ]
                self.logger.info(f"   Filtered {col} in [{min_val}, {max_val}]: {len(df_filtered):,} rows remain")
        
        # Remove rows with critical missing data
        critical_cols = ['IncorrectRejectionsN', 'CorrectN', 'accuracy']
        for col in critical_cols:
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col].notna()]
        
        final_count = len(df_filtered)
        self.logger.info(f"   ✅ Quality filtering: {initial_count:,} → {final_count:,} rows "
                        f"({final_count/initial_count*100:.1f}% retained)")
        
        return df_filtered
    
    def get_baseline_timepoint(self, df: pd.DataFrame) -> Optional[str]:
        """
        Identify the baseline timepoint value (usually m00 for BHR)
        """
        for col in self.timepoint_columns:
            if col in df.columns:
                values = df[col].value_counts()
                
                # Check for m00 (BHR baseline)
                if 'm00' in values.index:
                    return 'm00'
                
                # Check for other baseline indicators
                for val in values.index[:10]:  # Check top 10 values
                    val_str = str(val).lower()
                    if 'baseline' in val_str or val_str in ['0', '1', 'bl', 'v0', 'v1']:
                        return val
                
                # Use the earliest timepoint
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].min()
                
        return None
    
    def smart_bhr_merge(
        self, 
        memtrax_df: pd.DataFrame,
        medical_df: pd.DataFrame,
        demographics_df: Optional[pd.DataFrame] = None,
        subject_col: str = 'SubjectCode'
    ) -> pd.DataFrame:
        """
        Perform BHR-aware merge following successful patterns
        """
        self.logger.info("\n🧠 BHR-AWARE MERGE STRATEGY")
        self.logger.info("=" * 50)
        
        # Step 1: Detect dataset types
        is_memtrax_bhr = self.detect_bhr_dataset(memtrax_df, "memtrax")
        is_medical_bhr = self.detect_bhr_dataset(medical_df, "medical")
        
        if is_memtrax_bhr or is_medical_bhr:
            self.logger.info("✅ BHR dataset pattern detected - applying specialized merge strategy")
        
        # Step 2: Apply quality filters to MemTrax
        self.logger.info("\n📊 Step 1: Applying quality filters to MemTrax...")
        memtrax_clean = self.apply_cognitive_quality_filters(memtrax_df)
        
        # Step 3: Filter to baseline timepoints
        self.logger.info("\n🕐 Step 2: Filtering to baseline timepoints...")
        
        baseline_tp = self.get_baseline_timepoint(memtrax_clean)
        if baseline_tp:
            timepoint_col = None
            for col in self.timepoint_columns:
                if col in memtrax_clean.columns:
                    timepoint_col = col
                    break
            
            if timepoint_col:
                memtrax_baseline = memtrax_clean[memtrax_clean[timepoint_col] == baseline_tp]
                self.logger.info(f"   MemTrax baseline ({baseline_tp}): {len(memtrax_baseline):,} rows, "
                               f"{memtrax_baseline[subject_col].nunique():,} subjects")
            else:
                memtrax_baseline = memtrax_clean
        else:
            memtrax_baseline = memtrax_clean
            self.logger.info("   No timepoint column found - using all data")
        
        # Do the same for medical data
        baseline_tp_med = self.get_baseline_timepoint(medical_df)
        if baseline_tp_med:
            timepoint_col_med = None
            for col in self.timepoint_columns:
                if col in medical_df.columns:
                    timepoint_col_med = col
                    break
            
            if timepoint_col_med:
                medical_baseline = medical_df[medical_df[timepoint_col_med] == baseline_tp_med]
                self.logger.info(f"   Medical baseline ({baseline_tp_med}): {len(medical_baseline):,} rows, "
                               f"{medical_baseline[subject_col].nunique():,} subjects")
            else:
                medical_baseline = medical_df
        else:
            medical_baseline = medical_df
        
        # Step 4: Sequential merging (demographics first if available)
        if demographics_df is not None:
            self.logger.info("\n👥 Step 3: Adding demographics first...")
            
            # Clean demographics (remove duplicates)
            demo_clean = demographics_df.drop_duplicates(subset=[subject_col])
            
            # Merge with demographics
            merged = memtrax_baseline.merge(
                demo_clean,
                on=subject_col,
                how='inner',
                suffixes=('', '_demo')
            )
            self.logger.info(f"   After demographics: {len(merged):,} rows, "
                           f"{merged[subject_col].nunique():,} subjects")
        else:
            merged = memtrax_baseline
        
        # Step 5: Add medical data
        self.logger.info("\n🏥 Step 4: Adding medical history...")
        
        # Check for Cartesian join risk
        records_per_subject_med = len(medical_baseline) / medical_baseline[subject_col].nunique()
        if records_per_subject_med > 1.5:
            self.logger.warning(f"   ⚠️ Medical data has {records_per_subject_med:.1f} records per subject")
            self.logger.info("   Taking latest record per subject...")
            
            # Take the latest record per subject
            if timepoint_col_med and timepoint_col_med in medical_baseline.columns:
                medical_baseline = medical_baseline.sort_values(timepoint_col_med).groupby(subject_col).last().reset_index()
            else:
                medical_baseline = medical_baseline.groupby(subject_col).last().reset_index()
        
        # Final merge
        final_merged = merged.merge(
            medical_baseline,
            on=subject_col,
            how='inner',
            suffixes=('_memtrax', '_medical')
        )
        
        # Step 6: Validate results
        self.logger.info("\n✅ MERGE COMPLETE")
        self.logger.info(f"   Final dataset: {len(final_merged):,} rows, "
                       f"{final_merged[subject_col].nunique():,} subjects")
        self.logger.info(f"   Records per subject: {len(final_merged) / final_merged[subject_col].nunique():.2f}")
        
        # Warn if potential Cartesian join
        if len(final_merged) > len(memtrax_baseline) * 1.5:
            self.logger.warning(f"   ⚠️ Possible residual Cartesian effect - "
                             f"consider additional filtering")
        
        return final_merged


def integrate_with_cognitive_agent():
    """
    Integration code for CognitiveAnalysisAgent
    """
    integration_code = '''
    # Add to CognitiveAnalysisAgent.__init__:
    from improvements.bhr_aware_merging import BHRAwareMerger
    self.bhr_merger = BHRAwareMerger(logger=self.logger)
    
    # Modify _combine_assessment_datasets method:
    def _combine_assessment_datasets(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Enhanced combination with BHR awareness"""
        
        # Check if this looks like BHR data
        is_bhr = any(self.bhr_merger.detect_bhr_dataset(df) for df in datasets)
        
        if is_bhr and len(datasets) >= 2:
            # Use BHR-aware merging
            memtrax_df = datasets[0]  # Assuming first is cognitive data
            medical_df = datasets[1]  # Assuming second is medical
            demographics_df = datasets[2] if len(datasets) > 2 else None
            
            return self.bhr_merger.smart_bhr_merge(
                memtrax_df, medical_df, demographics_df
            )
        else:
            # Use existing merge logic
            return self._original_combine_assessment_datasets(datasets)
    '''
    
    return integration_code


if __name__ == "__main__":
    # Example usage
    print("BHR-Aware Merging Module")
    print("=" * 50)
    print("\nKey Features:")
    print("✅ Timepoint-first filtering (m00 baseline)")
    print("✅ Quality pre-filtering (Status == 'Collected', RT ranges)")
    print("✅ Sequential merge pattern (demographics → medical)")
    print("✅ Cartesian join prevention")
    print("✅ BHR-specific pattern recognition")
    print("\nIntegration:")
    print(integrate_with_cognitive_agent()) 