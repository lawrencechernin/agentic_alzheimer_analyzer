#!/usr/bin/env python3
"""
BRFSS surveillance dataset adapter: loads configured CSVs and returns a combined frame.
"""
from typing import Dict, Any, List
import os
import glob
import pandas as pd
from .base_adapter import BaseDatasetAdapter


class BrfssAdapter(BaseDatasetAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        dataset_cfg = config.get('dataset', {})
        sources = dataset_cfg.get('data_sources', [])
        self.paths: List[str] = [s.get('path') for s in sources if s.get('type') == 'local_directory']
        self.file_patterns = dataset_cfg.get('file_patterns', {}).get('brfss_surveillance_data', ["*.csv"])

    def is_available(self) -> bool:
        for base in self.paths:
            if not base or not os.path.exists(base):
                continue
            for pattern in self.file_patterns:
                if glob.glob(os.path.join(base, '**', pattern), recursive=True):
                    return True
        return False

    def _discover_files(self) -> List[str]:
        files = []
        for base in self.paths:
            if not base or not os.path.exists(base):
                continue
            for pattern in self.file_patterns:
                files.extend(glob.glob(os.path.join(base, '**', pattern), recursive=True))
        # Deduplicate
        return sorted(list(set(files)))

    def load_combined(self) -> pd.DataFrame:
        files = self._discover_files()
        frames = []
        for fp in files[:10]:  # cap for safety
            try:
                df = pd.read_csv(fp, low_memory=False)
                frames.append(df)
            except Exception:
                continue
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # For BRFSS, keep all important surveillance columns
        if not combined.empty:
            # Keep all BRFSS-specific columns that are important for analysis
            important_cols = [
                'YearStart', 'YearEnd', 'LocationAbbr', 'LocationDesc', 
                'Datasource', 'Class', 'Topic', 'Question',
                'Data_Value', 'Data_Value_Alt', 'Data_Value_Type', 'Data_Value_Unit',
                'Low_Confidence_Limit', 'High_Confidence_Limit',
                'StratificationCategory1', 'Stratification1', 
                'StratificationCategory2', 'Stratification2',
                'Geolocation', 'ClassID', 'TopicID', 'QuestionID', 'LocationID'
            ]
            
            # Keep columns that exist in the dataset
            cols_to_keep = [c for c in important_cols if c in combined.columns]
            
            # Also keep any numeric columns not already in the list
            numeric_cols = combined.select_dtypes(include=[int, float]).columns.tolist()
            for col in numeric_cols:
                if col not in cols_to_keep:
                    cols_to_keep.append(col)
            
            combined = combined[cols_to_keep]
            # Ensure unique column names to avoid DataFrame returns on label selection
            combined = combined.loc[:, ~combined.columns.duplicated()]
        self.combined_data = combined
        return combined

    def data_summary(self) -> Dict[str, Any]:
        total = 0 if self.combined_data is None else len(self.combined_data)
        return {
            'assessments_loaded': [
                {'type': 'brfss_surveillance_data', 'files': None, 'records': total}
            ],
            'total_subjects': total,
            'baseline_subjects': total,
            'preprocessing_steps': [
                "BRFSS adapter: loaded CSV files per config patterns",
                "Merged frames and retained numeric + key surveillance columns",
                "Deduplicated duplicate-named columns"
            ]
        } 