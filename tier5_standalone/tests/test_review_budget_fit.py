"""The review budget end to end: --target-review-count through fit_frame."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tca import pipeline
from tier5 import budget, cells, config as t5cfg, fit


def _extract(n=24_000, months=12, seed=4):
    """A fat-tailed year, shaped like the real HK/VWAP book."""
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    spreads = -0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63))
    days = pd.bdate_range("2025-06-02", periods=int(21.7 * months))
    sprd = rng.uniform(4.0, 12.0, n)
    return pd.DataFrame({
        "aggrTgtId": np.arange(n),
        "Sym": [f"{i % 300:04d}.HK" for i in range(n)],
        "Date": rng.choice(days, n),
        "Side": rng.choice(["BUY", "SELL"], n),
        "Strategy": "VWAP",
        "Sprd": sprd,
        "Pvwap": (spreads + 0.6) * sprd,
        "ePvwap/Sprd": spreads,
        "$Mln": rng.lognormal(1.0, 1.2, n),
    })


def _prepared(df):
    import config as appcfg
    out, _ = pipeline.prepare(df, appcfg.COLUMN_MAP, appcfg.DATA,
                              appcfg.SLIPPAGE_SIGN,
                              pre_transform=appcfg.PRE_TRANSFORM)
    return out


def _fit(df, tmp_path, **over):
    # A review count overrides the coverage standard that CONFIG now carries;
    # spell that out here so each test states one rule, not two.
    if over.get("target_review_count") is not None:
        over.setdefault("target_flag_rate", None)   # the CLI does this too
    # A plain fit here means "no rule at all", not the shipped per-side one.
    over.setdefault("band_percentile", None)
    over.setdefault("band_abs", None)
    cfg = dataclasses.replace(t5cfg.CONFIG, **over)
    return fit.fit_frame(_prepared(df), cfg,
                         bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="test.csv")


class TestBudgetDrivesK:
    def test_two_a_month_pushes_k_well_past_three(self, tmp_path):
        r = _fit(_extract(), tmp_path, target_review_count=2)[0]
        assert r["k_used"] > 4.0
        assert not r["budget"]["floored"]

    def test_the_band_covers_almost_the_whole_book(self, tmp_path):
        r = _fit(_extract(), tmp_path, target_review_count=2)[0]
        assert r["flag_rate_pct"] < 0.15

    def test_it_delivers_roughly_the_requested_count(self, tmp_path):
        df = _extract()
        r = _fit(df, tmp_path, target_review_count=2)[0]
        per_month = r["flag_rate_pct"] / 100.0 * r["n"] / 12.0
        assert per_month == pytest.approx(2.0, abs=1.0)

    def test_a_bigger_budget_gives_a_tighter_band(self, tmp_path):
        df = _extract()
        few = _fit(df, tmp_path, target_review_count=2)[0]
        many = _fit(df, tmp_path, target_review_count=30)[0]
        assert many["k_used"] < few["k_used"]
        assert many["hi"] < few["hi"] and many["lo"] > few["lo"]

    def test_the_centre_and_scale_do_not_move(self, tmp_path):
        """Only the multiple moves. It is still mu +/- k*sigma."""
        df = _extract()
        plain = _fit(df, tmp_path, target_review_count=None)[0]
        budgeted = _fit(df, tmp_path, target_review_count=2)[0]
        assert budgeted["centre"] == pytest.approx(plain["centre"])
        assert budgeted["scale"] == pytest.approx(plain["scale"])

    def test_the_bound_really_is_centre_plus_k_scale(self, tmp_path):
        r = _fit(_extract(), tmp_path, target_review_count=2)[0]
        assert r["hi"] == pytest.approx(r["centre"] + r["k_used"] * r["scale"])
        assert r["lo"] == pytest.approx(r["centre"] - r["k_used"] * r["scale"])


class TestGuards:
    def test_a_thin_cell_is_held_at_the_floor(self, tmp_path):
        # 600 orders a year cannot supply 24 a year without a 4% flag rate.
        r = _fit(_extract(n=600), tmp_path,
                 target_review_count=2, min_group_n=100)[0]
        assert r["k_used"] == t5cfg.BUDGET_K_FLOOR
        assert r["budget"]["floored"]

    def test_the_floor_never_narrows_the_band(self, tmp_path):
        df = _extract(n=600)
        plain = _fit(df, tmp_path, target_review_count=None,
                     target_flag_rate=None, min_group_n=100,
                     k_sigma=t5cfg.BUDGET_K_FLOOR)[0]
        held = _fit(df, tmp_path, target_review_count=2, min_group_n=100)[0]
        assert held["hi"] == pytest.approx(plain["hi"])

    def test_a_count_overrides_the_coverage_standard(self, tmp_path):
        """Most specific wins, and it is not silent -- main() prints a NOTE."""
        r = _fit(_extract(), tmp_path,
                 target_review_count=2, target_flag_rate=1.0)[0]
        assert r["budget"] is not None, "the count was ignored"
        per_month = r["flag_rate_pct"] / 100.0 * r["n"] / 12.0
        assert per_month == pytest.approx(2.0, abs=1.0)

    def test_a_negative_budget_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            _fit(_extract(n=3000), tmp_path, target_review_count=-1)


class TestProvenance:
    def test_the_band_file_records_the_budget_not_just_the_k(self, tmp_path):
        r = _fit(_extract(), tmp_path, target_review_count=2)[0]
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "target_review_count"
        assert saved["target_review_count"] == 2.0
        assert saved["review_budget"]["n_tail"] > 0
        assert saved["k_sigma"] == pytest.approx(r["k_used"])

    def test_a_plain_fit_still_says_fixed(self, tmp_path):
        r = _fit(_extract(n=3000), tmp_path, target_flag_rate=None)[0]
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "fixed"
        assert saved["target_review_count"] is None
