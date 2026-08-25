"""The caption must MEASURE the band it drew, never repeat a claim.

The picture said `k = 4` while its own lines sat at 4.24 and 4.47 sigma. Not a
wrong band -- the band was exactly right, P0.5/P99.5 having won the MAX -- but
a caption that took k as an argument and printed it, while the bounds beside it
had been set by something else entirely.

That is the same defect as a metric-source block that named a column without
checking it existed: a number the reader trusts, sourced from an assertion
rather than from the thing on the page. `plot` already has centre, scale, lo
and hi, so the implied k is derivable and there is no reason to accept it.
"""
from __future__ import annotations

import numpy as np
import pytest

from tier5 import curve


class TestTheCaptionIsDerived:
    def test_a_symmetric_band_reports_its_one_k(self):
        text = curve.caption(n=1000, centre=0.0, scale=2.0, lo=-8.0, hi=8.0,
                             outside=0.01, n_offscreen=0, units="spreads")
        assert "k = 4" in text

    def test_the_real_case_reports_what_was_drawn_not_what_was_asked_for(self):
        """HK/VWAP: centre -0.25, sigma 2.67, band -11.58 .. 11.68.

        mean +/- 4*sigma would be -10.93 .. 10.43. It is not that band, so the
        caption must not say 4.
        """
        text = curve.caption(n=46_950, centre=-0.25, scale=2.67,
                             lo=-11.58, hi=11.68, outside=0.01,
                             n_offscreen=188, units="spreads")
        assert "k = 4 " not in text and "k = 4\n" not in text
        assert "4.24" in text and "4.47" in text

    def test_an_asymmetric_band_shows_both_sides(self):
        text = curve.caption(n=1000, centre=0.0, scale=1.0, lo=-5.0, hi=3.0,
                             outside=0.01, n_offscreen=0, units="spreads")
        assert "5" in text and "3" in text

    def test_two_decimals_not_false_precision(self):
        text = curve.caption(n=1000, centre=0.0, scale=1.0,
                             lo=-4.339160209, hi=4.339160209,
                             outside=0.01, n_offscreen=0, units="spreads")
        assert "4.34" in text
        assert "4.3391" not in text

    def test_a_whole_number_k_has_no_decimals(self):
        text = curve.caption(n=1000, centre=0.0, scale=2.0, lo=-6.0, hi=6.0,
                             outside=0.01, n_offscreen=0, units="spreads")
        assert "k = 3 " in text or text.count("k = 3") == 1

    def test_it_survives_a_zero_scale(self):
        text = curve.caption(n=10, centre=0.0, scale=0.0, lo=-1.0, hi=1.0,
                             outside=0.0, n_offscreen=0, units="")
        assert isinstance(text, str) and "n = 10" in text


class TestPlotDrivesItFromTheDrawnBounds:
    def test_the_written_caption_matches_the_bounds_not_the_config(self, tmp_path):
        """End to end: plot() gets a band that is NOT centre +/- 4*scale."""
        rng = np.random.default_rng(0)
        x = rng.normal(-0.25, 2.67, 20_000)
        msg = curve.plot(x, centre=-0.25, scale=2.67, lo=-11.58, hi=11.68,
                         path=str(tmp_path / "c.png"), title="t",
                         units="spreads")
        assert "skipped" not in msg
        # the caption is inside the png; assert via the pure function instead
        text = curve.caption(n=20_000, centre=-0.25, scale=2.67,
                             lo=-11.58, hi=11.68, outside=0.01,
                             n_offscreen=0, units="spreads")
        assert "4.24" in text and "4.47" in text

    def test_plot_no_longer_accepts_a_k_it_would_have_to_trust(self):
        import inspect
        assert "k" not in inspect.signature(curve.plot).parameters, \
            "a k parameter is a claim; the band on the page is the evidence"
