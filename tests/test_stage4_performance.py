"""
Regression test for "Stage 4 is too slow".

Profiling (cProfile through streamlit.testing.v1.AppTest) showed Stage 4
("Regime Intelligence") spent ~60% of its 2.21s render time inside
Plotly's fig.add_vrect()/add_shape() -- even with only 36 shape calls,
plotly.basedatatypes._process_multiple_axis_spanning_shapes revalidates
the ENTIRE shapes list on every call (same O(n^2)-per-call cost as the
Stage 3 add_vline bug, just at smaller scale here). On top of that, the
`regime_classifications`/`market_data` full-table reads had no
st.cache_data, so every widget interaction (e.g. changing the ticker
selectbox) re-ran both queries from scratch even though Streamlit reruns
the whole script on every interaction.

Fixed by: building the regime-shading shapes as a single list assigned via
fig.update_layout(shapes=...) instead of looping add_vrect(), replacing the
row-by-row .iloc loop with vectorized run-length encoding, and wrapping the
DB reads in @st.cache_data(ttl=1800).
"""
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

MAX_FIRST_LOAD_SECONDS = 5
MAX_CACHED_RERUN_SECONDS = 2


def test_stage4_regime_intelligence_loads_quickly():
    at = AppTest.from_file('App_v2.py', default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value('4. Regime Intelligence')

    start = time.time()
    at.run()
    first_load = time.time() - start
    assert not at.exception, f"Stage 4 raised: {list(at.exception)}"
    assert first_load < MAX_FIRST_LOAD_SECONDS, (
        f"Stage 4 first load took {first_load:.2f}s (must be < {MAX_FIRST_LOAD_SECONDS}s)"
    )

    if at.selectbox:
        sb = at.selectbox[0]
        other = next((o for o in sb.options if o != sb.value), None)
        if other is not None:
            sb.set_value(other)
            start = time.time()
            at.run()
            rerun_time = time.time() - start
            assert not at.exception, f"Stage 4 raised on rerun: {list(at.exception)}"
            assert rerun_time < MAX_CACHED_RERUN_SECONDS, (
                f"Stage 4 took {rerun_time:.2f}s after a widget change (must be < "
                f"{MAX_CACHED_RERUN_SECONDS}s) -- DB reads should be cached, not "
                "re-queried on every interaction."
            )
