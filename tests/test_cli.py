from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src import cli


def test_parser_accepts_shared_options_before_subcommand() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["--dry-run", "--run-version", "v4", "full"])

    assert args.command == "full"
    assert args.dry_run is True
    assert args.run_version == "v4"


def test_parser_accepts_shared_options_after_subcommand() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["analytics", "--dry-run", "--run-version", "v5"])

    assert args.command == "analytics"
    assert args.dry_run is True
    assert args.run_version == "v5"


def test_parser_accepts_no_subcommand_default() -> None:
    parser = cli.build_parser()

    args = parser.parse_args([])

    assert args.command is None


def test_parser_accepts_activity_signal_and_min_dwell() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["preprocess", "--activity-signal", "ActMindata", "--min-dwell", "2"]
    )

    assert args.activity_signal == "ActMindata"
    assert args.min_dwell == 2


def test_main_tolerates_namespace_without_new_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subcommand parsers suppress defaults, so the attributes may be absent."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli.workflows, "validate_project_root", lambda _: Path("."))
    monkeypatch.setattr(cli.workflows, "run_full", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            command=None,
            project_root=".",
            config=None,
            run_version="v6",
            data_revised_dir=None,
            raw_data_file=None,
            processed_base_dir=None,
            output_base_dir=None,
            hmm_input_file=None,
            dry_run=True,
            skip_existing=False,
        ),
    )

    cli.main()

    assert captured["activity_signal"] is None
    assert captured["min_dwell"] is None


def test_main_dispatches_config_to_full_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_file = project_root / "alt-config.yaml"
    config_file.write_text("random_state: 7\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_validate_project_root(value: str) -> Path:
        assert value == str(project_root)
        return project_root

    def fake_run_full(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli.workflows, "validate_project_root", fake_validate_project_root)
    monkeypatch.setattr(cli.workflows, "run_full", fake_run_full)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            command="full",
            project_root=str(project_root),
            config="alt-config.yaml",
            run_version="v7",
            data_revised_dir=None,
            raw_data_file=None,
            processed_base_dir=None,
            output_base_dir=None,
            hmm_input_file=None,
            dry_run=True,
            skip_existing=False,
        ),
    )

    cli.main()

    assert captured["project_root"] == project_root
    assert captured["config_file"] == "alt-config.yaml"
    assert captured["run_version"] == "v7"
    assert captured["dry_run"] is True


def test_glmm_doctor_dispatches_without_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            command="glmm-doctor",
            project_root=".",
            config=None,
            run_version="v3",
            data_revised_dir=None,
            raw_data_file=None,
            processed_base_dir=None,
            output_base_dir=None,
            hmm_input_file=None,
            dry_run=False,
            skip_existing=False,
        ),
    )
    monkeypatch.setattr(cli.doctor, "run_glmm_doctor", lambda: 0)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0


def test_main_defaults_to_full_when_no_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli.workflows, "validate_project_root", lambda _: Path("."))
    monkeypatch.setattr(cli.workflows, "run_full", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            command=None,
            project_root=".",
            config=None,
            run_version="v3",
            data_revised_dir=None,
            raw_data_file=None,
            processed_base_dir=None,
            output_base_dir=None,
            hmm_input_file=None,
            dry_run=True,
            skip_existing=False,
        ),
    )

    cli.main()

    assert captured["run_version"] == "v3"
    assert captured["dry_run"] is True