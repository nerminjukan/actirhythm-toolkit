"""High-level workflow wrappers for package and CLI execution."""

from __future__ import annotations

import os
from pathlib import Path


def _set_runtime_env(
    run_version: str,
    config_file: str | Path | None = None,
    data_revised_dir: str | Path | None = None,
    raw_data_file: str | Path | None = None,
    processed_base_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
    hmm_input_file: str | Path | None = None,
    activity_signal: str | None = None,
    min_dwell: int | None = None,
) -> None:
    """Set shared environment variables used by script modules."""
    os.environ["THESIS_RUN_VERSION"] = run_version
    if config_file is not None:
        os.environ["THESIS_CONFIG_FILE"] = str(config_file)
    if data_revised_dir is not None:
        os.environ["THESIS_DATA_REVISED_DIR"] = str(data_revised_dir)
    if raw_data_file is not None:
        os.environ["THESIS_RAW_DATA_FILE"] = str(raw_data_file)
    if processed_base_dir is not None:
        os.environ["THESIS_PROCESSED_BASE_DIR"] = str(processed_base_dir)
    if output_base_dir is not None:
        os.environ["THESIS_OUTPUT_BASE_DIR"] = str(output_base_dir)
    if hmm_input_file is not None:
        os.environ["THESIS_HMM_INPUT_FILE"] = str(hmm_input_file)
    if activity_signal is not None:
        os.environ["THESIS_ACTIVITY_SIGNAL"] = str(activity_signal)
    if min_dwell is not None:
        os.environ["THESIS_MIN_DWELL"] = str(min_dwell)


def _resolve_layout(
    project_root: str | Path,
    run_version: str,
    processed_base_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    processed_base = (
        (root / "data" / "processed") if processed_base_dir is None else Path(processed_base_dir)
    )
    output_base = (
        (root / "past-runs") if output_base_dir is None else Path(output_base_dir)
    )

    if not processed_base.is_absolute():
        processed_base = (root / processed_base).resolve()
    if not output_base.is_absolute():
        output_base = (root / output_base).resolve()

    processed_dir = processed_base / run_version
    output_dir = output_base / run_version / "outputs"
    return {
        "project_root": root,
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "feature_file": processed_dir / "feature_data_improved.parquet",
        "hmm_file": processed_dir / "hmm_results_improved.parquet",
        "summary_file": output_dir / "updated_results.json",
    }


def _print_plan(command: str, layout: dict[str, Path], run_version: str) -> None:
    print(f"[dry-run] Command: {command}")
    print(f"[dry-run] Run version: {run_version}")
    print(f"[dry-run] Project root: {layout['project_root']}")
    if "config_file" in layout:
        config_file = layout["config_file"]
        if config_file is None:
            print("[dry-run] Config file: default internal configuration")
        else:
            print(f"[dry-run] Config file: {config_file}")
    print(f"[dry-run] Processed directory: {layout['processed_dir']}")
    print(f"[dry-run] Output directory: {layout['output_dir']}")


def run_preprocess(
    run_version: str = "v3",
    project_root: str | Path = ".",
    config_file: str | Path | None = None,
    data_revised_dir: str | Path | None = None,
    raw_data_file: str | Path | None = None,
    processed_base_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
    hmm_input_file: str | Path | None = None,
    activity_signal: str | None = None,
    min_dwell: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> None:
    """Run the improved preprocessing + HMM pipeline stage."""
    layout = _resolve_layout(project_root, run_version, processed_base_dir, output_base_dir)

    layout["config_file"] = resolve_config_file(project_root, config_file)

    if dry_run:
        _print_plan("preprocess", layout, run_version)
        print(f"[dry-run] Expected output: {layout['feature_file']}")
        print(f"[dry-run] Expected output: {layout['hmm_file']}")
        return

    if skip_existing and layout["hmm_file"].exists() and layout["feature_file"].exists():
        print("[skip-existing] Preprocess outputs already exist; skipping preprocess stage.")
        return

    _set_runtime_env(
        run_version,
        config_file=layout["config_file"],
        data_revised_dir=data_revised_dir,
        raw_data_file=raw_data_file,
        processed_base_dir=processed_base_dir,
        output_base_dir=output_base_dir,
        hmm_input_file=hmm_input_file,
        activity_signal=activity_signal,
        min_dwell=min_dwell,
    )
    import improved_pipeline

    improved_pipeline.run_improved_pipeline()


def run_analytics(
    run_version: str = "v3",
    project_root: str | Path = ".",
    config_file: str | Path | None = None,
    data_revised_dir: str | Path | None = None,
    raw_data_file: str | Path | None = None,
    processed_base_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
    hmm_input_file: str | Path | None = None,
    activity_signal: str | None = None,
    min_dwell: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> None:
    """Run downstream updated analytics stage."""
    layout = _resolve_layout(project_root, run_version, processed_base_dir, output_base_dir)

    layout["config_file"] = resolve_config_file(project_root, config_file)

    if dry_run:
        _print_plan("analytics", layout, run_version)
        print(f"[dry-run] Required input: {layout['hmm_file']}")
        print(f"[dry-run] Expected output: {layout['summary_file']}")
        return

    if skip_existing and layout["summary_file"].exists():
        print("[skip-existing] Analytics summary already exists; skipping analytics stage.")
        return

    _set_runtime_env(
        run_version,
        config_file=layout["config_file"],
        data_revised_dir=data_revised_dir,
        raw_data_file=raw_data_file,
        processed_base_dir=processed_base_dir,
        output_base_dir=output_base_dir,
        hmm_input_file=hmm_input_file,
        activity_signal=activity_signal,
        min_dwell=min_dwell,
    )
    import run_updated_analysis

    run_updated_analysis.main()


def run_full(
    run_version: str = "v3",
    project_root: str | Path = ".",
    config_file: str | Path | None = None,
    data_revised_dir: str | Path | None = None,
    raw_data_file: str | Path | None = None,
    processed_base_dir: str | Path | None = None,
    output_base_dir: str | Path | None = None,
    hmm_input_file: str | Path | None = None,
    activity_signal: str | None = None,
    min_dwell: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> None:
    """Run the complete two-stage analysis workflow."""
    if dry_run:
        layout = _resolve_layout(project_root, run_version, processed_base_dir, output_base_dir)
        layout["config_file"] = resolve_config_file(project_root, config_file)
        _print_plan("full", layout, run_version)
        print(f"[dry-run] Expected preprocess output: {layout['hmm_file']}")
        print(f"[dry-run] Expected analytics output: {layout['summary_file']}")
        return

    run_preprocess(
        run_version=run_version,
        project_root=project_root,
        config_file=config_file,
        data_revised_dir=data_revised_dir,
        raw_data_file=raw_data_file,
        processed_base_dir=processed_base_dir,
        output_base_dir=output_base_dir,
        hmm_input_file=hmm_input_file,
        activity_signal=activity_signal,
        min_dwell=min_dwell,
        dry_run=False,
        skip_existing=skip_existing,
    )
    run_analytics(
        run_version=run_version,
        project_root=project_root,
        config_file=config_file,
        data_revised_dir=data_revised_dir,
        raw_data_file=raw_data_file,
        processed_base_dir=processed_base_dir,
        output_base_dir=output_base_dir,
        hmm_input_file=hmm_input_file,
        activity_signal=activity_signal,
        min_dwell=min_dwell,
        dry_run=False,
        skip_existing=skip_existing,
    )


def validate_project_root(project_root: str | Path) -> Path:
    """Validate and normalize project root before running workflows."""
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    return root


def resolve_config_file(project_root: str | Path, config_file: str | Path | None = None) -> Path | None:
    """Resolve config path relative to the project root when available."""
    root = Path(project_root).resolve()
    candidate = Path(config_file) if config_file is not None else Path("config.yaml")
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    if not candidate.exists():
        if config_file is not None:
            raise FileNotFoundError(
                f"Config file does not exist: {candidate}. Pass --config with a valid file path."
            )
        return None
    return candidate
