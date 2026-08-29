"""Create or verify the publication release-integrity manifest.

The manifest binds the canonical implementation, analysis programs, and the
numerical artifacts cited by the manuscript. Historical provenance files are
left immutable; this release-level manifest records the exact distributed
snapshot and fails closed when any tracked byte changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"

CANONICAL_FILES = (
    "src/candidate_selection.py",
    "src/newmetric_corrected.py",
    "stage1_finsler_test/probe_persistence.py",
    "stage1_finsler_test/probe_realmask_seed.py",
    "study/core/build_frozen_candidate_pool.py",
    "study/evaluation/evaluate_lbs_error_decomposition.py",
    "study/evaluation/evaluate_candidate_ceiling_saturation.py",
    "study/analysis/analyze_candidate_ceiling_saturation.py",
    "study/evaluation/evaluate_cross_cohort_boundary_ceiling.py",
    "study/analysis/cluster_bootstrap_cross_cohort_boundary.py",
    "study/analysis/compute_primary_performance_intervals.py",
)

RESULT_FILES = (
    "results/pure_flair_p1_corrected_finsler/candidate_pool/summary.json",
    "results/pure_flair_p1_corrected_finsler/candidate_pool/provenance.json",
    "results/pure_flair_p1_corrected_finsler/lbs_error_decomposition/summary.json",
    "results/pure_flair_p1_corrected_finsler/lbs_error_decomposition/provenance.json",
    "results/pure_flair_p1_corrected_finsler/candidate_ceiling_saturation/summary.json",
    "results/pure_flair_p1_corrected_finsler/candidate_ceiling_saturation/provenance.json",
    "results/pure_flair_p1_corrected_finsler/cross_cohort_boundary_ceiling/summary.json",
    "results/pure_flair_p1_corrected_finsler/cross_cohort_boundary_ceiling/provenance.json",
    "results/pure_flair_p1_corrected_finsler/consolidated_audit.json",
    "results/pure_flair_p1_corrected_finsler/claim_source_matrix_core.csv",
    "results/pure_flair_p1_corrected_finsler/primary_performance_intervals.json",
)

ENVIRONMENT_FILES = (
    ".python-version",
    "requirements.txt",
    "requirements-lock.txt",
    "tools/release_manifest.py",
    "tools/verify_environment.py",
    "tests/test_release_integrity.py",
)


def digest(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required release file is missing: {relative_path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_scope": "BMC Medical Imaging candidate-selection manuscript",
        "canonical_pipeline": "pure_flair_p1_corrected_finsler",
        "hash_algorithm": "sha256-bytes",
        "policy": (
            "Historical per-analysis provenance is preserved. This manifest is "
            "the authoritative integrity record for the distributed release snapshot."
        ),
        "canonical_code": {path: digest(path) for path in CANONICAL_FILES},
        "manuscript_evidence": {path: digest(path) for path in RESULT_FILES},
        "reproduction_environment": {path: digest(path) for path in ENVIRONMENT_FILES},
    }


def write_manifest() -> None:
    MANIFEST.write_text(json.dumps(snapshot(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError("RELEASE_MANIFEST.json is missing; run with --write first")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = snapshot()
    if expected != actual:
        for section in ("canonical_code", "manuscript_evidence", "reproduction_environment"):
            old = expected.get(section, {})
            new = actual.get(section, {})
            for path in sorted(set(old) | set(new)):
                if old.get(path) != new.get(path):
                    print(f"MISMATCH {path}: expected={old.get(path)} actual={new.get(path)}")
        raise SystemExit("Release-integrity verification failed")
    print(
        "Release-integrity verification passed: "
        f"{len(CANONICAL_FILES)} code files, {len(RESULT_FILES)} evidence files, "
        f"and {len(ENVIRONMENT_FILES)} environment files."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the current verified snapshot")
    args = parser.parse_args()
    write_manifest() if args.write else verify_manifest()


if __name__ == "__main__":
    main()
