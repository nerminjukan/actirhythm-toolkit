"""Command line interface for actirhythm workflows."""

from __future__ import annotations

import argparse
import os
import sys

from . import doctor, workflows


def _add_common_arguments(parser: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    default_value = argparse.SUPPRESS if suppress_defaults else None
    run_version_default = argparse.SUPPRESS if suppress_defaults else "v3"
    project_root_default = argparse.SUPPRESS if suppress_defaults else "."
    parser.add_argument(
        "--project-root",
        default=project_root_default,
        help="Path to project root containing config.yaml (default: current directory)",
    )
    parser.add_argument(
        "--config",
        default=default_value,
        help="Override YAML config file path relative to project root or absolute path",
    )
    parser.add_argument(
        "--run-version",
        default=run_version_default,
        help="Run version namespace for outputs (default: v3)",
    )
    parser.add_argument(
        "--data-revised-dir",
        default=default_value,
        help="Override revised input directory (default: data-revised)",
    )
    parser.add_argument(
        "--raw-data-file",
        default=default_value,
        help="Fallback raw CSV path when revised directory is missing",
    )
    parser.add_argument(
        "--processed-base-dir",
        default=default_value,
        help="Override processed data base directory (default: data/processed)",
    )
    parser.add_argument(
        "--output-base-dir",
        default=default_value,
        help="Override run outputs base directory (default: past-runs)",
    )
    parser.add_argument(
        "--hmm-input-file",
        default=default_value,
        help="Override HMM parquet input path for analytics stage",
    )
    parser.add_argument(
        "--activity-signal",
        default=default_value,
        help=(
            "Column used as the activity signal (default: ActMindata, the "
            "firmware minutes-active count). The activity_xy/activity_xyz "
            "columns describe posture and are not intensity measures."
        ),
    )
    parser.add_argument(
        "--min-dwell",
        type=int,
        default=default_value,
        help=(
            "Minimum bout length in bins for the dwell-time filter "
            "(default: 2, i.e. 30 min at 15-min sampling)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Print planned inputs/outputs without running analysis",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="Skip a stage when its expected outputs already exist",
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common)

    common_subcommand = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common_subcommand, suppress_defaults=True)

    parser = argparse.ArgumentParser(
        prog="actirhythm",
        description=(
            "Run the accelerometer analysis pipeline as an installable Python package."
        ),
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser(
        "run",
        help="Run preprocess + analytics stages (alias of default command)",
        parents=[common_subcommand],
    )
    subparsers.add_parser(
        "full",
        help="Run preprocess + analytics stages",
        parents=[common_subcommand],
    )
    subparsers.add_parser(
        "preprocess",
        help="Run improved preprocessing + HMM stage",
        parents=[common_subcommand],
    )
    subparsers.add_parser(
        "analytics",
        help="Run updated downstream analytics stage",
        parents=[common_subcommand],
    )
    subparsers.add_parser(
        "glmm-doctor",
        help="Check Python and R prerequisites for pymer4-backed GLMM workflows",
        parents=[common_subcommand],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "glmm-doctor":
        raise SystemExit(doctor.run_glmm_doctor())

    project_root = workflows.validate_project_root(args.project_root)
    os.chdir(project_root)

    common_kwargs = {
        "run_version": args.run_version,
        "project_root": project_root,
        "config_file": args.config,
        "data_revised_dir": args.data_revised_dir,
        "raw_data_file": args.raw_data_file,
        "processed_base_dir": args.processed_base_dir,
        "output_base_dir": args.output_base_dir,
        "hmm_input_file": args.hmm_input_file,
        # Absent when a subcommand parser suppresses its defaults.
        "activity_signal": getattr(args, "activity_signal", None),
        "min_dwell": getattr(args, "min_dwell", None),
        "dry_run": args.dry_run,
        "skip_existing": args.skip_existing,
    }

    if args.command in {None, "run", "full"}:
        workflows.run_full(**common_kwargs)
    elif args.command == "preprocess":
        workflows.run_preprocess(**common_kwargs)
    elif args.command == "analytics":
        workflows.run_analytics(**common_kwargs)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
