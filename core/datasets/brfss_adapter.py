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
        # Keep a set of likely analysis columns (numeric)
        if not combined.empty:
            numeric = combined.select_dtypes(include=[int, float])
            # Keep geographic/time columns if present
            keep_cols = [c for c in combined.columns if c in (
                'LocationAbbr', 'LocationDesc', 'YearStart', 'YearEnd', 'Class', 'Topic', 'Question'
            )]
            combined = pd.concat([numeric, combined[keep_cols]], axis=1)
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