#!/usr/bin/env python3
"""
ADDI / AD Workbench dataset adapter: loads exported CSVs and combines them safely.
"""
from typing import Dict, Any, List, Optional
import os
import glob
import pandas as pd
import logging
from .base_adapter import BaseDatasetAdapter


class ADDIWorkbenchAdapter(BaseDatasetAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        dataset_cfg = config.get('dataset', {}) or {}
        sources = dataset_cfg.get('data_sources', []) or []
        self.paths: List[str] = [s.get('path') for s in sources if s.get('type') in ("local_directory", "local_file") and s.get('path')] or ["./training_data/addi_workbench_export/"]
        patterns_cfg = dataset_cfg.get('file_patterns', {}) or {}
        # Prefer explicit assessment patterns; fallback to any csv
        self.file_patterns: List[str] = (
            patterns_cfg.get('assessment_data')
            or patterns_cfg.get('generic_csv')
            or ["*.csv"]
        )
        # Candidate subject identifier columns commonly seen across Alzheimer's datasets
        self.subject_id_candidates: List[str] = [
            'RID', 'PTID', 'Subject', 'Subject_ID', 'SubjectID', 'SubjectCode',
            'participant_id', 'subject_id', 'ID', 'Code'
        ]
        # Optional date/time columns to pick baseline or most recent
        self.date_candidates: List[str] = [
            'StatusDateTime', 'ExamDate', 'VisitDate', 'date', 'test_date', 'TimepointDate'
        ]
        # Sampling
        analysis_cfg = config.get('analysis', {}) or {}
        self.use_sampling: bool = bool(analysis_cfg.get('use_sampling', False))
        self.sample_size: int = int(analysis_cfg.get('analysis_sample_size', 20000) or 20000)

    def is_available(self) -> bool:
        for fp in self._discover_files():
            if os.path.exists(fp):
                return True
        return False

    def _discover_files(self) -> List[str]:
        files: List[str] = []
        for base in self.paths:
            if not base:
                continue
            if os.path.isfile(base) and base.lower().endswith('.csv'):
                files.append(base)
                continue
            if os.path.isdir(base):
                for pattern in self.file_patterns:
                    files.extend(glob.glob(os.path.join(base, '**', pattern), recursive=True))
        # Deduplicate
        files = sorted(list(set(files)))
        return files

    def _detect_subject_id_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in self.subject_id_candidates:
            if col in df.columns:
                return col
        # Fallback: heuristic
        for col in df.columns:
            lower = str(col).lower()
            if 'subject' in lower or lower in ('id', 'code'):
                return col
        return None

    def _select_baseline(self, df: pd.DataFrame, subject_col: str) -> pd.DataFrame:
        # Prefer earliest visit per subject using available date columns; otherwise first occurrence
        for date_col in self.date_candidates:
            if date_col in df.columns:
                try:
                    temp = df.copy()
                    temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
                    temp = temp.sort_values([subject_col, date_col], ascending=[True, True])
                    return temp.drop_duplicates(subset=[subject_col], keep='first')
                except Exception:
                    continue
        # No date column: just keep first record per subject
        return df.drop_duplicates(subset=[subject_col], keep='first')

    def load_combined(self) -> pd.DataFrame:
        files = self._discover_files()
        if not files:
            self.combined_data = pd.DataFrame()
            return self.combined_data

        # Load a conservative number of files initially
        frames: List[pd.DataFrame] = []
        for fp in files[:6]:  # limit for safety
            try:
                read_kwargs = {"low_memory": False}
                if self.use_sampling and self.sample_size > 0:
                    read_kwargs["nrows"] = self.sample_size
                df = pd.read_csv(fp, **read_kwargs)
                self.logger.info(f"ADDIWorkbenchAdapter: loaded {'sample of ' + str(len(df)) if 'nrows' in read_kwargs else str(len(df))} rows from {os.path.basename(fp)}")
                subj_col = self._detect_subject_id_column(df)
                if not subj_col:
                    continue
                df = df.dropna(subset=[subj_col])
                df_baseline = self._select_baseline(df, subj_col)
                # Keep only informative columns (drop all-empty columns)
                non_empty_cols = [c for c in df_baseline.columns if df_baseline[c].notnull().any()]
                df_baseline = df_baseline[non_empty_cols]
                frames.append(df_baseline)
            except Exception as e:
                self.logger.warning(f"ADDIWorkbenchAdapter: failed to load {fp}: {e}")
                continue

        if not frames:
            self.combined_data = pd.DataFrame()
            return self.combined_data

        # Harmonize subject column name across frames
        unified_frames: List[pd.DataFrame] = []
        unified_subject_col = 'Subject_ID'
        for df in frames:
            subj_col = None
            for cand in self.subject_id_candidates:
                if cand in df.columns:
                    subj_col = cand
                    break
            if not subj_col:
                continue
            temp = df.rename(columns={subj_col: unified_subject_col})
            unified_frames.append(temp)

        # Merge safely using inner joins; start with the widest frame
        unified_frames.sort(key=lambda d: d.shape[1], reverse=True)
        combined: Optional[pd.DataFrame] = None
        for i, df in enumerate(unified_frames):
            if combined is None:
                combined = df
            else:
                before = len(combined)
                # Deduplicate before merge
                combined = combined.drop_duplicates(subset=[unified_subject_col])
                df = df.drop_duplicates(subset=[unified_subject_col])
                combined = combined.merge(df, on=unified_subject_col, how='inner')
                after = len(combined)
                # If we exploded, fall back to left join to retain cohort without blowing up
                if before and after / before > 10:
                    combined = combined.merge(df, on=unified_subject_col, how='left')
        if combined is None:
            combined = pd.DataFrame()

        self.combined_data = combined
        return combined

    def data_summary(self) -> Dict[str, Any]:
        total = 0 if self.combined_data is None else len(self.combined_data)
        return {
            'assessments_loaded': [
                {'type': 'addi_workbench', 'files': 0 if self.combined_data is None else 1, 'records': total}
            ],
            'total_subjects': total,
            'baseline_subjects': total,
            'preprocessing_steps': [
                "ADDI/Workbench adapter: discovered CSVs, detected subject IDs",
                "Selected baseline per subject using date if available",
                "Merged datasets conservatively to avoid Cartesian joins",
                f"Sampling: {'enabled' if self.use_sampling else 'disabled'} (rows={self.sample_size if self.use_sampling else 'all'})"
            ]
        } 