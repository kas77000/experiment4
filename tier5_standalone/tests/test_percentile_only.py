"""A band cut from percentiles alone, with no sigma term at all.

Reachable before this only by setting K_SIGMA = 0 and relying on MAX(mean, P)
collapsing to P -- true, but a trick, and a rule nobody should have to
reverse-engineer from an identity. K_SIGMA = None says it outright.

Worth having as a first-class option because it is the one rule that makes no
distributional assumption whatsoever: no centre, no scale, no implied
symmetry. On a book this far from normal that is a real argument, and the cost
is that the bound is only as stable as the handful of orders beyond it.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tca import pipeline
from tier5 import band, config as t5cfg, fit


def _skewed(n=40_000, seed=4):
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    return (-0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63))
            - np.where(heavy, rng.exponential(1.2, n), 0.0))


class TestRuleBoundsAcceptsNoSigmaTerm:
    def test_k_none_gives_exactly_the_percentiles(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=None, percentile=99.5)
        assert r["hi"] == pytest.approx(float(np.quantile(x, 0.995)))
        assert r["lo"] == pytest.approx(float(np.quantile(x, 0.005)))

    def test_both_sides_say_the_percentile_bound_them(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=None, percentile=99.5)
        assert r["hi_binds"] == "percentile" and r["lo_binds"] == "percentile"

    def test_the_sigma_candidate_is_absent_not_zero(self):
        """Zero would read as 'the band could have been the mean'."""
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=None, percentile=99.5)
        assert not np.isfinite(r["hi_sigma"])
        assert not np.isfinite(r["lo_sigma"])

    def test_it_ignores_the_centre_entirely(self):
        """No centre, no scale, no assumed symmetry -- that is the point."""
        x = _skewed()
        a = band.rule_bounds(x, 0.0, 1.0, k=None, percentile=99.5)
        b = band.rule_bounds(x, 999.0, 77.0, k=None, percentile=99.5)
        assert (a["lo"], a["hi"]) == pytest.approx((b["lo"], b["hi"]))

    def test_it_is_asymmetric_when_the_book_is(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=None, percentile=99.5)
        assert abs(abs(r["lo"]) - abs(r["hi"])) > 0.1


def _extract(n=24_000, seed=4):
    rng = np.random.default_rng(seed)
    sp = _skewed(n, seed)
    sprd = rng.uniform(4.0, 12.0, n)
    return pd.DataFrame({
        "aggrTgtId": np.arange(n),
        "Sym": [f"{i % 300:04d}.HK" for i in range(n)],
        "Date": rng.choice(pd.bdate_range("2025-06-02", periods=261), n),
        "Side": rng.choice(["BUY", "SELL"], n), "Strategy": "VWAP",
        "Sprd": sprd, "Pvwap": (sp + 0.6) * sprd, "ePvwap/Sprd": sp,
        "$Mln": rng.lognormal(1.0, 1.2, n)})


def _fit(tmp_path, **over):
    import config as appcfg
    prep, _ = pipeline.prepare(_extract(), appcfg.COLUMN_MAP, appcfg.DATA,
                               appcfg.SLIPPAGE_SIGN,
                               pre_transform=appcfg.PRE_TRANSFORM)
    cfg = dataclasses.replace(t5cfg.CONFIG, **over)
    return fit.fit_frame(prep, cfg, bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="t.csv")[0]


PCT_ONLY = dict(band_abs=None, k_sigma=None, band_percentile=99.5)


class TestEndToEnd:
    def test_the_band_is_the_percentiles_of_the_metric(self, tmp_path):
        r = _fit(tmp_path, **PCT_ONLY)
        x = np.sort(_skewed(24_000, 4))
        assert r["hi"] == pytest.approx(float(np.quantile(x, 0.995)), rel=1e-6)

    def test_it_delivers_half_a_percent_per_tail(self, tmp_path):
        r = _fit(tmp_path, **PCT_ONLY)
        assert r["flag_rate_pct"] == pytest.approx(1.0, abs=0.05)

    def test_a_tighter_percentile_gives_a_wider_band(self, tmp_path):
        loose = _fit(tmp_path, **{**PCT_ONLY, "band_percentile": 99.0})
        tight = _fit(tmp_path, **{**PCT_ONLY, "band_percentile": 99.9})
        assert tight["hi"] > loose["hi"] and tight["lo"] < loose["lo"]

    def test_the_band_file_records_that_no_sigma_term_was_used(self, tmp_path):
        r = _fit(tmp_path, **PCT_ONLY)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "sigma_or_percentile"
        assert saved["band_rule"]["k_sigma"] is None
        assert saved["band_rule"]["hi_binds"] == "percentile"

    def test_centre_and_scale_are_still_measured(self, tmp_path):
        """Reported for context and needed to rank outliers, not to set bounds."""
        r = _fit(tmp_path, **PCT_ONLY)
        assert np.isfinite(r["centre"]) and r["scale"] > 0


class TestTheCli:
    def test_percentile_flag_switches_everything_else_off(self, tmp_path,
                                                          monkeypatch, capsys):
        import sys
        csv = tmp_path / "y.csv"
        _extract().to_csv(csv, index=False)
        monkeypatch.setattr(sys, "argv",
                            ["fit", "--csv", str(csv), "--percentile", "99.9",
                             "--bands-dir", str(tmp_path / "b"),
                             "--out-dir", str(tmp_path / "o")])
        fit.main()
        saved = json.loads((tmp_path / "b" / "HK" / "VWAP.json").read_text())
        assert saved["band_abs"] is None, "the absolute band was not cleared"
        assert saved["band_rule"]["k_sigma"] is None
        assert saved["band_rule"]["percentile"] == 99.9
