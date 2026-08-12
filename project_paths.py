"""Canonical filesystem locations for the Finsler brain-tumour project.

All maintained code should import paths from this module instead of deriving the
repository root from a script's nesting depth or embedding workstation-specific
absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("FINSLER_DATA_ROOT", PROJECT_ROOT / "data"))
SOURCE_ROOT = PROJECT_ROOT / "src"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
STUDIES_ROOT = PROJECT_ROOT / "studies"
RESULTS_ROOT = PROJECT_ROOT / "results"
PAPER_ROOT = PROJECT_ROOT / "paper"
DOCS_ROOT = PROJECT_ROOT / "docs"
ARCHIVE_ROOT = PROJECT_ROOT / "archive"
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def require_directory(path: Path, label: str) -> Path:
    """Return *path* when it exists, otherwise raise an actionable error."""
    if not path.is_dir():
        raise FileNotFoundError(f"{label} dizini bulunamadı: {path}")
    return path
