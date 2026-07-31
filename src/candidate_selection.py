"""Stable public entry point for the canonical single-FLAIR pipeline.

The experiment modules under ``pipeline/`` are kept under their original names
so that the analyses reported in the paper stay reproducible. New code should
import through this module rather than reaching into ``pipeline/`` directly.

Typical use::

    from candidate_selection import pipeline, persistence, brain_roi_tcga

The NewMetric backend is resolved by ``pipeline/probe_realmask_seed.py``. It
looks for a ``facseg`` checkout containing ``src/facseg``; the vendored copy
under ``facseg/`` in this repository satisfies that search, so no environment
variable is required. Setting ``FACSEG_ROOT`` overrides it.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import probe_persistence as persistence  # noqa: E402
import probe_realmask_seed as pipeline  # noqa: E402
from brain_roi_tcga import brain_roi_tcga  # noqa: E402

__all__ = ["brain_roi_tcga", "persistence", "pipeline"]
