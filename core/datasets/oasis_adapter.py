#!/usr/bin/env python3
"""
OASIS dataset adapter: loads cross-sectional and longitudinal data and combines them.
"""
from typing import Dict, Any
import os
import pandas as pd
from .base_adapter import BaseDatasetAdapter


class OasisAdapter(BaseDatasetAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = "./training_data/oasis/"

    def is_available(self) -> bool:
        return (
            os.path.exists(os.path.join(self.data_path, "oasis_cross-sectional.csv")) and
            os.path.exists(os.path.join(self.data_path, "oasis_longitudinal.csv"))
        )

    def load_combined(self) -> pd.DataFrame:
        cross_df = pd.read_csv(os.path.join(self.data_path, "oasis_cross-sectional.csv"))
        long_df = pd.read_csv(os.path.join(self.data_path, "oasis_longitudinal.csv"))

        cross_df = cross_df.rename(columns={'ID': 'Subject_ID', 'M/F': 'Gender', 'Educ': 'EDUC'})
        long_df = long_df.rename(columns={'Subject ID': 'Subject_ID', 'M/F': 'Gender'})

        common_cols = list(set(cross_df.columns) & set(long_df.columns))
        cross_common = cross_df[common_cols]
        long_common = long_df[common_cols]
        combined = pd.concat([cross_common, long_common], ignore_index=True)

        if 'CDR' in combined.columns:
            combined = combined.dropna(subset=['CDR'])

        # Gentle imputation like current logic
        if 'SES' in combined.columns and combined['SES'].isnull().any():
            from sklearn.impute import SimpleImputer
            mode_imputer = SimpleImputer(strategy='most_frequent')
            combined[['SES']] = mode_imputer.fit_transform(combined[['SES']])
        if 'MMSE' in combined.columns and combined['MMSE'].isnull().any():
            from sklearn.impute import SimpleImputer
            median_imputer = SimpleImputer(strategy='median')
            combined[['MMSE']] = median_imputer.fit_transform(combined[['MMSE']])

        self.combined_data = combined
        return combined

    def data_summary(self) -> Dict[str, Any]:
        total = 0 if self.combined_data is None else len(self.combined_data)
        return {
            'assessments_loaded': [
                {'type': 'brain_imaging_data', 'files': 2, 'records': total},
                {'type': 'clinical_data', 'files': 2, 'records': total}
            ],
            'total_subjects': total,
            'baseline_subjects': total,
            'preprocessing_steps': [
                "OASIS adapter: combined cross-sectional + longitudinal datasets",
                "Harmonized column names",
                "Dropped rows missing CDR",
                "Imputed SES (mode) and MMSE (median) if missing"
            ]
        } 