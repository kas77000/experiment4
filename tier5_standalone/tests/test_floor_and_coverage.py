"""k = max(K_FLOOR sigma, the k that delivers COVERAGE_PCT).

Two bounds, and each catches what the other cannot:

  the COVERAGE widens a band whose tail is fatter than a Gaussian assumes.
  Without it, k = 4 on the real HK book leaves ~0.9% outside, not 0.006%.

  the FLOOR stops a well-behaved cell being handed an absurdly tight band.
  A near-normal cell reaches 99.5% at k = 2.95; shipping that alongside HK's
  4.06 would mean the quietest desk had the harshest threshold.

Taking the max means the band is never tighter than EITHER bound allows.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tca import pipeline
from tier5 import config as t5cfg, fit


def _book(n, w, s1, s2, seed):
    """A normal mixture: `w` of the mass tight, the rest heavy-tailed."""
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= w
    return -0.38 + rng.normal(0.0, np.where(heavy, s2, s1))


def _extract(spreads, seed=4):
    n = len(spreads)
    rng = np.random.default_rng(seed)
    sprd = rng.uniform(4.0, 12.0, n)
    return pd.DataFrame({
        "aggrTgtId": np.arange(n),
        "Sym": [f"{i % 300:04d}.HK" for i in range(n)],
        "Date": rng.choice(pd.bdate_range("2025-06-02", periods=261), n),
        "Side": rng.choice(["BUY", "SELL"], n), "Strategy": "VWAP",
        "Sprd": sprd, "Pvwap": (spreads + 0.6) * sprd,
        "ePvwap/Sprd": spreads, "$Mln": rng.lognormal(1.0, 1.2, n)})


# A sharp spike with a rare, violent tail -- the shape where a fixed multiple
# under-delivers worst, so the coverage bound has to widen past the floor.
FAT = _book(24_000, .85, .50, 4.50, 4)          # coverage binds (k ~ 5.2)
NEAR_NORMAL = _book(8_000, .97, 1.05, 1.6, 6)   # floor binds     (k ~ 2.9)


def _fit(tmp_path, spreads, **over):
    import config as appcfg
    prep, _ = pipeline.prepare(_extract(spreads), appcfg.COLUMN_MAP,
                               appcfg.DATA, appcfg.SLIPPAGE_SIGN,
                               pre_transform=appcfg.PRE_TRANSFORM)
    cfg = dataclasses.replace(t5cfg.CONFIG, **over)
    return fit.fit_frame(prep, cfg, bands_dir=str(tmp_path / "bands"),
                         out_dir=str(tmp_path / "out"), source_csv="t.csv")[0]


class TestTheRuleIsShipped:
    def test_config_states_both_bounds(self):
        assert t5cfg.K_FLOOR == 4.0
        assert t5cfg.COVERAGE_PCT == 99.5

    def test_the_floor_is_the_default_k(self):
        assert t5cfg.CONFIG.k_sigma == t5cfg.K_FLOOR

    def test_a_bare_fit_applies_it_with_no_arguments(self, tmp_path):
        r = _fit(tmp_path, FAT)
        assert r["k_used"] >= t5cfg.K_FLOOR


class TestWhichBoundBinds:
    def test_a_fat_tail_is_widened_past_the_floor(self, tmp_path):
        r = _fit(tmp_path, FAT)
        assert r["k_used"] > t5cfg.K_FLOOR
        assert not r["k_floored"]

    def test_a_near_normal_cell_is_held_at_the_floor(self, tmp_path):
        r = _fit(tmp_path, NEAR_NORMAL)
        assert r["k_used"] == pytest.approx(t5cfg.K_FLOOR)
        assert r["k_floored"]

    def test_the_floor_case_would_have_been_much_tighter(self, tmp_path):
        """The floor is doing real work, not rounding."""
        r = _fit(tmp_path, NEAR_NORMAL)
        assert r["k_from_coverage"] < 3.5

    def test_it_is_never_tighter_than_either_bound(self, tmp_path):
        for spreads in (FAT, NEAR_NORMAL):
            r = _fit(tmp_path, spreads)
            assert r["k_used"] >= t5cfg.K_FLOOR - 1e-9
            assert r["k_used"] >= r["k_from_coverage"] - 1e-9

    def test_the_floored_cell_flags_fewer_than_the_coverage_target(self, tmp_path):
        """A wider band than 99.5% asked for means fewer outside, not more."""
        r = _fit(tmp_path, NEAR_NORMAL)
        assert r["flag_rate_pct"] < 100.0 - t5cfg.COVERAGE_PCT

    def test_the_coverage_cell_delivers_the_target(self, tmp_path):
        r = _fit(tmp_path, FAT)
        assert r["flag_rate_pct"] == pytest.approx(100.0 - t5cfg.COVERAGE_PCT,
                                                   abs=0.05)


class TestStillTheSameMethod:
    def test_bounds_are_centre_plus_minus_k_scale(self, tmp_path):
        for spreads in (FAT, NEAR_NORMAL):
            r = _fit(tmp_path, spreads)
            assert r["hi"] == pytest.approx(r["centre"] + r["k_used"] * r["scale"])
            assert r["lo"] == pytest.approx(r["centre"] - r["k_used"] * r["scale"])

    def test_raising_the_floor_raises_a_floored_cell_only(self, tmp_path):
        low = _fit(tmp_path, NEAR_NORMAL, k_sigma=4.0)
        high = _fit(tmp_path, NEAR_NORMAL, k_sigma=5.0)
        assert high["k_used"] == pytest.approx(5.0)
        assert low["k_used"] == pytest.approx(4.0)

    def test_an_explicit_fixed_k_still_wins(self, tmp_path):
        r = _fit(tmp_path, FAT, target_flag_rate=None, k_sigma=3.0)
        assert r["k_used"] == 3.0


class TestProvenance:
    def test_the_band_records_both_bounds_and_which_bound(self, tmp_path):
        r = _fit(tmp_path, NEAR_NORMAL)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "k_floor"
        assert saved["k_floor"] == pytest.approx(4.0)
        assert saved["k_from_coverage"] == pytest.approx(r["k_from_coverage"])
        assert saved["coverage_pct"] == pytest.approx(99.5)

    def test_a_widened_cell_says_coverage(self, tmp_path):
        r = _fit(tmp_path, FAT)
        saved = json.loads(open(r["band_path"]).read())
        assert saved["k_source"] == "target_flag_rate"
        assert saved["k_sigma"] == pytest.approx(r["k_used"])
