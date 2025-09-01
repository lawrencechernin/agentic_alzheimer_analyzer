#!/usr/bin/env python3
"""
Base dataset adapter interface for dataset-specific loading and preprocessing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract base for dataset adapters."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.combined_data: Optional[pd.DataFrame] = None

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if files/resources for this adapter are present."""
        raise NotImplementedError

    @abstractmethod
    def load_combined(self) -> pd.DataFrame:
        """Load and return a combined dataframe suitable for downstream analysis."""
        raise NotImplementedError

    @abstractmethod
    def data_summary(self) -> Dict[str, Any]:
        """Return a summary dict (subjects, preprocessing steps, etc.)."""
        raise NotImplementedError 