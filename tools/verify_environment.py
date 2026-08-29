"""Verify Python and installed package versions against the publication lock."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def expected_environment() -> tuple[str, dict[str, str]]:
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    packages: dict[str, str] = {}
    for raw_line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            packages[name] = version
    return python_version, packages


def mismatches() -> list[str]:
    expected_python, packages = expected_environment()
    actual_python = ".".join(map(str, sys.version_info[:3]))
    errors = [] if actual_python == expected_python else [
        f"Python: expected {expected_python}, found {actual_python}"
    ]
    for name, expected in packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name}: expected {expected}, package is not installed")
            continue
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    return errors


def main() -> None:
    errors = mismatches()
    if errors:
        raise SystemExit("Environment verification failed:\n- " + "\n- ".join(errors))
    python_version, packages = expected_environment()
    print(f"Environment verification passed: Python {python_version}, {len(packages)} packages.")


if __name__ == "__main__":
    main()
