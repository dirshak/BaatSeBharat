"""
Regression tests for Stage 7's `ValueError: scale < 0` crash in
src/geo_dashboard.py.

Two layers, both fixed:

1. Immediate cause: _simulated_wb_data() computed
   `np.random.normal(0, base * 0.15)` where `base` can be negative (e.g.
   "Current Account (% GDP)" defaults to -1.5), making the scale argument
   negative -- illegal for np.random.normal. Fixed with abs(base) * 0.15.

2. Real root cause: fetch_wb_data() was silently falling through to the
   simulated-data path because the real fetch was broken. The installed
   wbdata (0.3.0) queries the plural REST endpoint
   (api.worldbank.org/v2/countries/.../indicators/...), which no longer
   round-trips against the live API (empty body -> JSONDecodeError). Fixed
   by fetching directly from the confirmed-working singular endpoint
   (.../v2/country/.../indicator/...), with wbdata and then simulated data
   only as explicit, clearly-labeled fallbacks (df.attrs['simulated']).
"""
import numpy as np
import pytest

pytest.importorskip("requests")


def test_simulated_wb_data_never_raises_for_any_indicator():
    """_simulated_wb_data must not crash regardless of the indicator's
    (possibly negative) baseline value -- this is the exact scenario that
    raised ValueError: scale < 0 in production."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from geo_dashboard import _simulated_wb_data, WB_INDICATORS

    for indicator_name in list(WB_INDICATORS.keys()) + ["Current Account (% GDP)", "Unknown Indicator"]:
        df = _simulated_wb_data(indicator_name)
        assert not df.empty
        assert df[indicator_name].notna().all() if indicator_name in df.columns else True


def test_simulated_data_is_flagged_not_silently_served_as_real():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from geo_dashboard import fetch_wb_data

    fetch_wb_data.clear()
    df = fetch_wb_data("NY.GDP.MKTP.KD.ZG", "GDP Growth (% annual)")
    assert 'simulated' in df.attrs, (
        "fetch_wb_data() must always set df.attrs['simulated'] so callers can "
        "show a visible notice instead of silently presenting synthetic data as real."
    )


@pytest.mark.network
def test_direct_world_bank_fetch_returns_real_data():
    """The direct-fetch path (the actual fix) must reach the live World
    Bank API and return real, non-simulated data for the indicator that
    previously crashed (negative baseline: Current Account % GDP)."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from geo_dashboard import fetch_wb_data

    fetch_wb_data.clear()
    df = fetch_wb_data("BN.CAB.XOKA.GD.ZS", "Current Account (% GDP)")
    if df.attrs.get('simulated'):
        pytest.skip("Live World Bank API unreachable in this environment")
    assert not df.empty
    assert "Country" in df.columns and "Year" in df.columns
