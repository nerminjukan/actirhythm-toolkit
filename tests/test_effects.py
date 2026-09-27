from __future__ import annotations

import numpy as np

from src.effects import test_bout_duration_normality


def test_bout_duration_normality_returns_ks_results_for_variable_data() -> None:
    data = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=float)

    result = test_bout_duration_normality(data)

    assert result["n"] == 6
    assert "ks_stat" in result
    assert "ks_p" in result
    assert result["ks_stat"] == result["ks_stat"]
    assert result["ks_p"] == result["ks_p"]


def test_bout_duration_normality_handles_zero_variance_sample() -> None:
    data = np.array([1.0, 1.0, 1.0], dtype=float)

    result = test_bout_duration_normality(data)

    assert result["n"] == 3
    assert result["ks_normal"] is None