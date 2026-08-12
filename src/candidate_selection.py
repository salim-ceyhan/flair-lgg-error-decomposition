"""Stable public entry point for the canonical single-FLAIR pipeline.

Historical experiment modules remain in ``stage1_finsler_test`` so previous
analyses stay reproducible. New study code must import through this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "stage1_finsler_test"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import probe_persistence as persistence  # noqa: E402
import probe_realmask_seed as pipeline  # noqa: E402
from brain_roi_tcga import brain_roi_tcga  # noqa: E402

__all__ = ["brain_roi_tcga", "persistence", "pipeline"]
