"""Materialize a verified local cache from a Google Drive streamed dataset.

Google Drive for desktop may intermittently return short reads even when file
metadata reports the complete size. This utility retries NumPy loads, writes each
array atomically to local SSD storage, verifies the result, and publishes a ready
marker only after the complete cohort succeeds.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np

REQUIRED_FILES = ("flair.npy", "mask.npy")

def load_with_retry(path: Path, attempts: int) -> np.ndarray:
    """Stage a Drive file locally before NumPy reads it.

    Windows CopyFile reliably hydrates Drive placeholders, while NumPy's direct
    fromfile path can receive a premature EOF from the virtual filesystem.
    """
    errors = []
    staging_root = default_destination().parent / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        fd, staged_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npy", dir=staging_root)
        os.close(fd)
        staged = Path(staged_name)
        try:
            if os.name == "nt":
                copied = ctypes.windll.kernel32.CopyFileW(str(path), str(staged), False)
                if not copied:
                    raise ctypes.WinError()
            else:
                import shutil
                shutil.copyfile(path, staged)
            if staged.stat().st_size != path.stat().st_size:
                raise OSError(
                    f"Short copy: expected {path.stat().st_size} bytes, "
                    f"received {staged.stat().st_size} bytes"
                )
            array = np.load(staged, allow_pickle=False)
            if array.ndim != 2 or array.size == 0:
                raise ValueError(f"Unexpected array shape: {array.shape}")
            _ = float(array.sum())
            return array.copy()
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            time.sleep(min(0.25 * attempt, 2.0))
        finally:
            staged.unlink(missing_ok=True)
    raise OSError(f"Could not stage and read {path} after {attempts} attempts. Last error: {errors[-1]}")

def atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npy", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.save(temporary, array, allow_pickle=False)
        check = np.load(temporary, allow_pickle=False)
        if check.shape != array.shape or check.dtype != array.dtype or not np.array_equal(check, array):
            raise ValueError(f"Verification failed for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def default_destination() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local / "finsler-tumor-data" / "tcga_lgg_dataset"

def main() -> None:
    parser = argparse.ArgumentParser(description="Create a verified local TCGA-LGG cache.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=default_destination())
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()

    # Match the study pipeline exactly: retain the slice with the largest
    # non-empty reference mask for each patient. The source contains all 1360
    # candidate slices, whereas the frozen evaluation cohort contains 110.
    patients = {}
    source_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
    for case_dir in source_dirs:
        if not all((case_dir / filename).is_file() for filename in REQUIRED_FILES):
            continue
        mask = load_with_retry(case_dir / "mask.npy", args.attempts)
        mask_size = int(mask.sum())
        if mask_size == 0:
            continue
        patient_id = "_".join(case_dir.name.split("_")[:4])
        if patient_id not in patients or mask_size > patients[patient_id][1]:
            patients[patient_id] = (case_dir, mask_size)
    cases = [item[0] for item in sorted(patients.values(), key=lambda item: item[0].name)]
    if len(cases) != 110:
        raise ValueError(
            f"Expected 110 patient-selected cases, found {len(cases)} "
            f"among {len(source_dirs)} source directories"
        )
    marker = args.destination / "_CACHE_READY.json"
    marker.unlink(missing_ok=True)
    manifest = []
    for index, case_dir in enumerate(cases, 1):
        record = {"case_id": case_dir.name, "files": {}}
        arrays = {}
        for filename in REQUIRED_FILES:
            array = load_with_retry(case_dir / filename, args.attempts)
            arrays[filename] = array
            record["files"][filename] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "minimum": float(array.min()), "maximum": float(array.max()),
            }
        if arrays["flair.npy"].shape != arrays["mask.npy"].shape:
            raise ValueError(f"Shape mismatch in {case_dir.name}")
        for filename, array in arrays.items():
            atomic_save(args.destination / case_dir.name / filename, array)
        manifest.append(record)
        if index % 10 == 0 or index == len(cases):
            print(f"Materialized and verified {index}/{len(cases)} cases", flush=True)

    payload = {
        "status": "ready", "case_count": len(cases),
        "source": str(args.source), "destination": str(args.destination),
        "required_files": list(REQUIRED_FILES), "cases": manifest,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified cache ready: {args.destination}")

if __name__ == "__main__":
    main()
