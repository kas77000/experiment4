"""The coverage standard: one number in config.py, no flags, every run.

The point of moving the rule out of the command line is that it stops being
something anybody can forget. These tests pin the rule itself and, more
importantly, the two traps a standing default creates -- a flag that appears
to work but is silently overruled, and a guard that fires on every run.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tca import pipeline
from tier5 import config as t5cfg, fit, normality

COVERAGE = 99.5


def _extract(n=24_000, seed=4):
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    sp = -0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63))
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
    df = _extract() if df is None else df
    prep, _ = pipeline.prepare(df, appcfg.COLUMN_MAP, appcfg.DATA,
                               appcfg.SLIPPAGE_SIGN,
                               pre_transform=appcfg.PRE_TRANSFORM)
    # These exercise the COVERAGE rule, which is now an opt-in: the
    # shipped per-side rule would otherwise take precedence over it.
    over.setdefault("band_percentile", None)
    over.setdefault("target_flag_rate", round(100.0 - 99.5, 10))
    cfg = dataclasses.replace(t5cfg.CONFIG, **over)
    return fit.fit_frame(prep, cfg, bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="t.csv")[0]


class TestTheStandardIsTheDefault:
    def test_config_states_a_coverage_not_a_flag_rate(self):
        """The rule is a coverage; the value is policy and may move."""
        assert 90.0 < COVERAGE < 100.0

    def test_the_coverage_rule_is_opt_in_not_the_default(self):
        """It moved: the shipped default is the per-side MAX rule, and a bare
        CONFIG must not silently mean coverage any more."""
        assert t5cfg.CONFIG.target_flag_rate is None
        assert t5cfg.CONFIG.band_percentile == t5cfg.PERCENTILE_PCT

    def test_a_bare_fit_delivers_the_stated_coverage(self, tmp_path):
        r = _fit(tmp_path)
        # <= not ==: the floor may widen past the target, never below it.
        assert r["flag_rate_pct"] <= 100.0 - COVERAGE + 0.02

    def test_a_bare_fit_is_never_tighter_than_the_floor(self, tmp_path):
        """The whole point: a fat book costs far more than a textbook 3."""
        r = _fit(tmp_path)
        assert r["k_used"] >= t5cfg.K_SIGMA
        assert r["k_used"] > 3.0

    def test_it_is_still_centre_plus_k_scale(self, tmp_path):
        r = _fit(tmp_path)
        assert r["hi"] == pytest.approx(r["centre"] + r["k_used"] * r["scale"])

    def test_changing_the_one_constant_moves_the_band(self, tmp_path):
        tight = _fit(tmp_path, target_flag_rate=100.0 - 99.0)
        wide = _fit(tmp_path, target_flag_rate=100.0 - 99.95)
        assert wide["hi"] > tight["hi"]
        assert wide["lo"] < tight["lo"]


class TestKIfNormal:
    @pytest.mark.parametrize("cov,k", [(99.0, 2.576), (99.73, 3.0),
                                       (99.9, 3.291), (99.95, 3.481)])
    def test_the_textbook_values(self, cov, k):
        assert normality.k_if_normal(cov) == pytest.approx(k, abs=0.005)

    def test_three_sigma_really_is_99_73_percent(self):
        """The identity the whole reframing rests on."""
        assert normality.k_if_normal(99.73) == pytest.approx(3.0, abs=0.005)

    def test_a_nonsense_coverage_is_nan(self):
        assert np.isnan(normality.k_if_normal(100.0))
        assert np.isnan(normality.k_if_normal(0.0))


class TestTheTrapsAStandingDefaultCreates:
    """Both of these were live bugs the moment the default was switched on."""

    def test_explicit_k_is_not_silently_overruled(self, tmp_path, monkeypatch,
                                                  capsys):
        import sys
        csv = tmp_path / "o.csv"
        _extract(n=3000).to_csv(csv, index=False)
        monkeypatch.setattr(sys, "argv",
                            ["fit", "--csv", str(csv), "--k", "3",
                             "--bands-dir", str(tmp_path / "b"),
                             "--out-dir", str(tmp_path / "o")])
        fit.main()
        saved = json.loads((tmp_path / "b" / "HK" / "VWAP.json").read_text())
        assert saved["k_sigma"] == 3.0, "--k lost to the config default"
        assert saved["k_source"] == "fixed"

    def test_target_review_count_does_not_trip_the_exclusivity_guard(
            self, tmp_path, monkeypatch):
        import sys
        csv = tmp_path / "o.csv"
        _extract(n=24_000).to_csv(csv, index=False)
        monkeypatch.setattr(sys, "argv",
                            ["fit", "--csv", str(csv),
                             "--target-review-count", "2",
                             "--bands-dir", str(tmp_path / "b"),
                             "--out-dir", str(tmp_path / "o")])
        fit.main()          # must not raise
        saved = json.loads((tmp_path / "b" / "HK" / "VWAP.json").read_text())
        assert saved["k_source"] == "target_review_count"


class TestProvenance:
    def test_the_band_records_the_standard_it_was_cut_to(self, tmp_path):
        r = _fit(tmp_path)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["coverage_pct"] == pytest.approx(COVERAGE)
        assert saved["k_if_normal"] == pytest.approx(
            normality.k_if_normal(COVERAGE), abs=0.005)
        assert saved["k_sigma"] == pytest.approx(r["k_used"])

    def test_the_gap_between_the_two_ks_is_the_non_normality(self, tmp_path):
        r = _fit(tmp_path)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_sigma"] > saved["k_if_normal"], "fat tail should cost more"
