#!/usr/bin/env python3
from typing import Dict, Any, Optional
from .base_adapter import BaseDatasetAdapter
from .oasis_adapter import OasisAdapter
from .brfss_adapter import BrfssAdapter


def get_adapter(config: Dict[str, Any]) -> Optional[BaseDatasetAdapter]:
    """Return a suitable dataset adapter given config."""
    name = (config.get('dataset', {}) or {}).get('name', '').lower()
    candidates = []
    # Prefer specific mapping by name hints
    if 'oasis' in name:
        candidates.append(OasisAdapter(config))
    if 'brfss' in name or 'healthy_aging' in name:
        candidates.append(BrfssAdapter(config))
    # Fallback: try both
    if not candidates:
        candidates = [OasisAdapter(config), BrfssAdapter(config)]
    for adapter in candidates:
        try:
            if adapter.is_available():
                return adapter
        except Exception:
            continue
    return None 