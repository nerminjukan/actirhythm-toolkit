from __future__ import annotations

import numpy as np

from src import cosinor


def test_multi_component_amplitudes_are_positive() -> None:
    """A negative fitted amplitude must be folded into the phase, not reported raw."""
    t = np.arange(0, 24, 0.25)
    # Second harmonic is deliberately inverted relative to a naive positive guess.
    y = (
        1.0
        + 0.6 * np.cos(2 * np.pi * t / 24 - 0.5)
        - 0.2 * np.cos(2 * np.pi * t / 12 - 0.3)
    )

    res = cosinor.fit_multi_component_cosinor(t, y, n_components=2, base_period=24.0)

    amps = [c["amplitude"] for c in res["components"]]
    assert all(a >= 0 for a in amps), amps
    assert res["r_squared"] > 0.99


def test_multi_component_acrophase_reports_true_peak() -> None:
    """The reported acrophase should coincide with the waveform maximum."""
    t = np.arange(0, 24, 0.1)
    peak = 18.0
    y = 1.0 + 0.5 * np.cos(2 * np.pi * (t - peak) / 24)

    res = cosinor.fit_multi_component_cosinor(t, y, n_components=1, base_period=24.0)
    acro = res["components"][0]["acrophase_hours"]

    assert abs(acro - peak) < 0.2, acro


def test_single_component_amplitude_is_positive() -> None:
    t = np.arange(0, 24, 0.25)
    y = 1.0 - 0.4 * np.cos(2 * np.pi * t / 24)

    res = cosinor.fit_cosinor(t, y, period=24.0)

    assert res["amplitude"] >= 0
    # Trough at 0 h means the peak sits half a period away.
    assert abs(res["acrophase_hours"] - 12.0) < 0.2
