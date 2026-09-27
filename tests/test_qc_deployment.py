from __future__ import annotations

import numpy as np
import pandas as pd

from src import qc


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=6, freq="15min")
    return pd.DataFrame(
        {
            "subject": ["A"] * 6 + ["B"] * 6,
            "timestamp": list(ts) + list(ts),
            "activity": list(range(6)) * 2,
        }
    )


def test_apply_deployment_windows_trims_at_cutoff() -> None:
    df = _frame()

    out = qc.apply_deployment_windows(df, overrides={"A": "2024-01-01 00:45"})

    assert len(out) == 9
    assert (out[out["subject"] == "A"]["timestamp"].max()
            < pd.Timestamp("2024-01-01 00:45"))
    assert len(out[out["subject"] == "B"]) == 6


def test_apply_deployment_windows_ignores_unknown_subject() -> None:
    df = _frame()

    out = qc.apply_deployment_windows(df, overrides={"Z": "2024-01-01 00:00"})

    assert len(out) == len(df)


def test_apply_deployment_windows_without_required_columns() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})

    out = qc.apply_deployment_windows(df, overrides={"A": "2024-01-01"})

    assert len(out) == 3


def test_assign_subject_codes_orders_by_first_observation() -> None:
    df = _frame()
    df.loc[df["subject"] == "B", "timestamp"] -= pd.Timedelta(days=1)

    out, mapping = qc.assign_subject_codes(df)

    assert mapping == {"B": "S1", "A": "S2"}
    assert set(out["subject_code"]) == {"S1", "S2"}


def test_assign_subject_codes_pads_width() -> None:
    ts = pd.date_range("2024-01-01", periods=12, freq="15min")
    df = pd.DataFrame({"subject": [f"n{i}" for i in range(12)], "timestamp": ts})

    _, mapping = qc.assign_subject_codes(df)

    assert mapping["n0"] == "S01"
    assert mapping["n11"] == "S12"
