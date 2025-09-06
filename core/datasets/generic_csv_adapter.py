#!/usr/bin/env python3
"""
Generic CSV dataset adapter: loads one or more CSV files from configured paths/patterns.
Optimized for simple, single-file datasets (e.g., Kaggle CSVs) without dataset-specific logic.
"""
from typing import Dict, Any, List
import os
import glob
import pandas as pd
import logging
from .base_adapter import BaseDatasetAdapter


class GenericCSVAdapter(BaseDatasetAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        dataset_cfg = config.get('dataset', {})
        sources = dataset_cfg.get('data_sources', []) or []
        # Default to current directory if not specified
        default_paths: List[str] = ["."]
        self.paths: List[str] = [s.get('path') for s in sources if s.get('type') in ("local_directory", "local_file") and s.get('path')] or default_paths
        patterns_cfg = dataset_cfg.get('file_patterns', {}) or {}
        # Prefer a generic key if provided; fallback to common names or any CSV
        self.file_patterns: List[str] = (
            patterns_cfg.get('generic_csv')
            or patterns_cfg.get('csv')
            or patterns_cfg.get('assessment_data')
            or ["*.csv"]
        )
        # Sampling settings for large datasets
        analysis_cfg = config.get('analysis', {}) or {}
        self.use_sampling: bool = bool(analysis_cfg.get('use_sampling', False))
        self.sample_size: int = int(analysis_cfg.get('analysis_sample_size', 5000) or 5000)
        # Logger
        self.logger = logging.getLogger(__name__)

    def is_available(self) -> bool:
        for fp in self._discover_files():
            if os.path.exists(fp):
                return True
        return False

    def _discover_files(self) -> List[str]:
        files: List[str] = []
        for base in self.paths:
            if not base or not os.path.exists(base):
                continue
            if os.path.isfile(base) and base.lower().endswith('.csv'):
                files.append(base)
                continue
            for pattern in self.file_patterns:
                files.extend(glob.glob(os.path.join(base, '**', pattern), recursive=True))
        # Deduplicate and prefer files with 'alzheimer' in name
        unique = sorted(list(set(files)))
        prioritized = [f for f in unique if 'alzheimer' in os.path.basename(f).lower()]
        return prioritized or unique

    def load_combined(self) -> pd.DataFrame:
        files = self._discover_files()
        frames: List[pd.DataFrame] = []
        # Prefer single-file datasets; cap to a few files for safety
        selected_files = files[:1] if files else []
        if len(files) > 1 and not selected_files:
            selected_files = files[:3]
        for fp in selected_files:
            try:
                if self.use_sampling and self.sample_size > 0:
                    df = pd.read_csv(fp, nrows=self.sample_size, low_memory=False)
                    self.logger.info(f"GenericCSVAdapter: loaded sample of {len(df)} rows from {os.path.basename(fp)}")
                else:
                    df = pd.read_csv(fp, low_memory=False)
                    self.logger.info(f"GenericCSVAdapter: loaded {len(df)} rows from {os.path.basename(fp)}")
                frames.append(df)
            except Exception as e:
                self.logger.warning(f"GenericCSVAdapter: failed to load {fp}: {e}")
                continue
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.combined_data = combined
        return combined

    def data_summary(self) -> Dict[str, Any]:
        total = 0 if self.combined_data is None else len(self.combined_data)
        return {
            'assessments_loaded': [
                {'type': 'generic_csv', 'files': 1 if total > 0 else 0, 'records': total}
            ],
            'total_subjects': total,
            'baseline_subjects': total,
            'preprocessing_steps': [
                "Generic CSV adapter: loaded CSV file(s) from configured paths/patterns",
                f"Sampling: {'enabled' if self.use_sampling else 'disabled'} (rows={self.sample_size if self.use_sampling else 'all'})"
            ]
        } 