"""Environment diagnostics for optional workflow dependencies."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from typing import Iterable


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_status(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def _check_r_packages(packages: Iterable[str]) -> None:
    try:
        from rpy2.robjects.packages import PackageNotInstalledError, importr
    except Exception as exc:
        _print_status("rpy2 R package checks", False, str(exc))
        return

    for package_name in packages:
        try:
            importr(package_name)
            _print_status(f"R package '{package_name}'", True)
        except PackageNotInstalledError:
            _print_status(f"R package '{package_name}'", False)
        except Exception as exc:
            _print_status(f"R package '{package_name}'", False, str(exc))


def run_glmm_doctor() -> int:
    """Report whether the host can support pymer4-backed GLMM workflows."""
    print("GLMM doctor")
    print(f"Python: {sys.version.split()[0]}")

    python_modules = [
        "pymer4",
        "polars",
        "rpy2",
        "great_tables",
    ]
    for module_name in python_modules:
        _print_status(f"Python module '{module_name}'", _module_available(module_name))

    rscript_path = shutil.which("Rscript")
    r_path = shutil.which("R")
    _print_status("Rscript executable", rscript_path is not None, rscript_path or "not on PATH")
    _print_status("R executable", r_path is not None, r_path or "not on PATH")

    if _module_available("rpy2") and (rscript_path or r_path):
        _check_r_packages(["base", "stats", "tibble", "lme4", "lmerTest"])
    else:
        print("[INFO] Skipping R package checks because rpy2 or R is unavailable.")

    missing_python = [name for name in python_modules if not _module_available(name)]
    if missing_python or not (rscript_path or r_path):
        return 1
    return 0
