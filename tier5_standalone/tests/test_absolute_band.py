"""An absolute band: -20 .. +20 spreads, stated rather than fitted.

This is a POLICY band, not an estimate. Zero is the meaningful reference for a
spread-normalised metric -- it means the order matched interval VWAP exactly --
so the bound is symmetric about zero and NOT about the book's own mean. A band
that drifted with the mean would move every time execution got worse, which is
the opposite of what a threshold is for.

The fit still computes and reports centre, scale and what the data would have
given, so the gap between the policy and the book stays visible.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tca import pipeline
from tier5 import config as t5cfg, fit


def _extract(n=24_000, seed=4, sigma_mult=1.0):
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    sp = sigma_mult * (-0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63)))
    sprd = rng.uniform(4.0, 12.0, n)
    return pd.DataFrame({
        "aggrTgtId": np.arange(n),
        "Sym": [f"{i % 300:04d}.HK" for i in range(n)],
        "Date": rng.choice(pd.bdate_range("2025-06-02", periods=261), n),
        "Side": rng.choice(["BUY", "SELL"], n), "Strategy": "VWAP",
        "Sprd": sprd, "Pvwap": (sp + 0.6) * sprd, "ePvwap/Sprd": sp,
        "$Mln": rng.lognormal(1.0, 1.2, n)})


def _fit(tmp_path, df=None, **over):
    import config as appcfg
    prep, _ = pipeline.prepare(_extract() if df is None else df,
                               appcfg.COLUMN_MAP, appcfg.DATA,
                               appcfg.SLIPPAGE_SIGN,
                               pre_transform=appcfg.PRE_TRANSFORM)
    cfg = dataclasses.replace(t5cfg.CONFIG, **over)
    return fit.fit_frame(prep, cfg, bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="t.csv")[0]


class TestItIsShipped:
    def test_config_states_twenty_spreads(self):
        assert t5cfg.BAND_ABS_SPREADS == 20.0

    def test_it_is_the_default(self):
        assert t5cfg.CONFIG.band_abs == t5cfg.BAND_ABS_SPREADS

    def test_a_bare_fit_lands_exactly_on_it(self, tmp_path):
        r = _fit(tmp_path)
        assert r["lo"] == pytest.approx(-20.0)
        assert r["hi"] == pytest.approx(20.0)


class TestItIsAbsoluteNotRelative:
    def test_it_does_not_move_with_the_mean(self, tmp_path):
        """The point of a policy band: a worse book must not widen it."""
        shifted = _extract()
        shifted["ePvwap/Sprd"] = shifted["ePvwap/Sprd"] - 5.0
        shifted["Pvwap"] = (shifted["ePvwap/Sprd"] + 0.6) * shifted["Sprd"]
        a = _fit(tmp_path, _extract())
        b = _fit(tmp_path, shifted)
        assert b["centre"] < a["centre"] - 4.0, "premise: the book moved"
        assert (a["lo"], a["hi"]) == pytest.approx((b["lo"], b["hi"]))

    def test_it_does_not_move_with_the_spread_of_the_book(self, tmp_path):
        tight = _fit(tmp_path, _extract(sigma_mult=0.25))
        wide = _fit(tmp_path, _extract(sigma_mult=2.0))
        assert wide["scale"] > 4 * tight["scale"], "premise: the books differ"
        assert (tight["lo"], tight["hi"]) == pytest.approx((wide["lo"], wide["hi"]))

    def test_it_is_symmetric_about_zero_not_about_the_centre(self, tmp_path):
        r = _fit(tmp_path)
        assert r["centre"] != pytest.approx(0.0), "premise: the book is off zero"
        assert r["lo"] == pytest.approx(-r["hi"])


class TestTheFitStillMeasures:
    def test_centre_and_scale_are_still_computed(self, tmp_path):
        r = _fit(tmp_path)
        assert np.isfinite(r["centre"]) and r["scale"] > 0

    def test_it_reports_the_implied_multiple(self, tmp_path):
        """How many sigma +/-20 actually is on this book -- the number that
        says whether the policy is loose or tight here."""
        r = _fit(tmp_path)
        assert r["abs_k_hi"] == pytest.approx(
            (20.0 - r["centre"]) / r["scale"], rel=1e-6)

    def test_the_in_sample_rate_is_measured_not_assumed(self, tmp_path):
        r = _fit(tmp_path)
        assert 0.0 <= r["flag_rate_pct"] < 5.0


class TestOverridesStillWin:
    @pytest.mark.parametrize("over,expect", [
        ({"band_abs": None, "band_percentile": None, "k_sigma": 3.0}, "fixed"),
        ({"band_abs": None, "band_percentile": None,
          "target_flag_rate": 0.5}, "target_flag_rate"),
        ({"band_abs": None, "band_percentile": None,
          "target_review_count": 2}, "target_review_count"),
    ])
    def test_an_explicit_rule_is_not_swallowed(self, tmp_path, over, expect):
        r = _fit(tmp_path, **over)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == expect
        assert r["hi"] != pytest.approx(20.0)


class TestProvenance:
    def test_the_band_file_says_it_was_stated_not_fitted(self, tmp_path):
        r = _fit(tmp_path)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "absolute"
        assert saved["band_abs"] == pytest.approx(20.0)
        assert saved["lo"] == pytest.approx(-20.0)
