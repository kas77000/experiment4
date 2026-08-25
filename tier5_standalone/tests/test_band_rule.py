"""The shipped rule, stated the way the desk states it:

    hi = MAX(mean + K*sigma,  P(PERCENTILE))
    lo = MIN(mean - K*sigma,  P(100 - PERCENTILE))

Two things distinguish this from a single symmetric k, and both matter on a
real book:

  IT IS PER SIDE. Slippage is skewed. Forcing both tails through one multiple
  makes the band wrong on at least one of them; here each side takes whichever
  of its own two candidates is wider.

  THE PERCENTILE IS A PERCENTILE, not a coverage. P99.5 leaves 0.5% above it
  in the upper tail. A "99.5% coverage" band splits that 0.5% across BOTH
  tails, which is a different and tighter bound (~P99.75).
"""
from __future__ import annotations

import numpy as np
import pytest

from tier5 import band


def _skewed(n=40_000, seed=4):
    """Fat-tailed and genuinely left-skewed, like execution slippage."""
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    return (-0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63))
            - np.where(heavy, rng.exponential(1.2, n), 0.0))


class TestTheRuleItself:
    def test_hi_is_the_max_of_the_two_candidates(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi"] == max(r["hi_sigma"], r["hi_pct"])

    def test_lo_is_the_min_of_the_two_candidates(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["lo"] == min(r["lo_sigma"], r["lo_pct"])

    def test_the_sigma_candidate_is_literally_mean_plus_k_sigma(self):
        """The boss's objection: sigma must be visibly, exactly in there."""
        x = _skewed()
        m, s = float(x.mean()), float(x.std(ddof=1))
        r = band.rule_bounds(x, m, s, k=4.0, percentile=99.5)
        assert r["hi_sigma"] == pytest.approx(m + 4.0 * s)
        assert r["lo_sigma"] == pytest.approx(m - 4.0 * s)

    def test_the_percentile_candidate_is_the_raw_empirical_percentile(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi_pct"] == pytest.approx(float(np.quantile(x, 0.995)))
        assert r["lo_pct"] == pytest.approx(float(np.quantile(x, 0.005)))

    def test_the_band_is_never_tighter_than_either_candidate(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi"] >= r["hi_sigma"] and r["hi"] >= r["hi_pct"]
        assert r["lo"] <= r["lo_sigma"] and r["lo"] <= r["lo_pct"]


class TestPerSideNotSymmetric:
    def test_the_two_sides_can_bind_differently(self):
        """The whole reason for a per-side rule."""
        rng = np.random.default_rng(1)
        # heavy only on the low side: sigma should win high, percentile low
        x = np.concatenate([rng.normal(0, 1, 20_000),
                            -rng.exponential(9.0, 400)])
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi_binds"] != r["lo_binds"]

    def test_on_a_real_shaped_book_sigma_wins_both_sides(self):
        """Worth pinning, because it is the surprise in this rule.

        At k = 4 the sigma term is WIDER than P99.5 on a book like the real
        one -- mean+4sigma = 10.54 against P99.5 = 8.30 -- so MAX() returns
        the sigma term and the percentile never fires. The band comes out
        symmetric despite the rule being per-side: the percentile is a safety
        net here, not an active bound.
        """
        x = _skewed()
        m = float(x.mean())
        r = band.rule_bounds(x, m, float(x.std(ddof=1)), k=4.0, percentile=99.5)
        assert r["hi_binds"] == "sigma" and r["lo_binds"] == "sigma"
        assert r["hi_pct"] < r["hi_sigma"]
        assert (m - r["lo"]) == pytest.approx(r["hi"] - m)

    def test_the_percentile_fires_when_the_tail_is_extreme_enough(self):
        """The net catching something: P99.5 beyond 4 sigma needs a tail heavy
        enough that sigma is dragged up by the very orders it should bound."""
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(0, 1, 40_000),
                            rng.normal(0, 1, 300) + 60])
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi_binds"] == "percentile"

    def test_it_reports_which_candidate_won_on_each_side(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["hi_binds"] in ("sigma", "percentile")
        assert r["lo_binds"] in ("sigma", "percentile")


class TestPercentileIsNotCoverage:
    def test_p99_5_is_looser_than_99_5_percent_coverage(self):
        """P99.5 leaves 0.5% above it; 99.5% coverage leaves 0.25% above."""
        x = _skewed()
        m, s = float(x.mean()), float(x.std(ddof=1))
        p = float(np.quantile(x, 0.995))
        coverage_equivalent = float(np.quantile(x, 0.9975))
        assert p < coverage_equivalent

    def test_the_percentile_term_alone_flags_half_a_percent_per_tail(self):
        x = _skewed()
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=0.0, percentile=99.5)   # sigma term switched off
        assert float(np.mean(x > r["hi"])) == pytest.approx(0.005, abs=0.001)
        assert float(np.mean(x < r["lo"])) == pytest.approx(0.005, abs=0.001)


class TestEdges:
    def test_an_empty_array_is_nan_not_a_crash(self):
        r = band.rule_bounds(np.array([]), 0.0, 1.0, k=4.0, percentile=99.5)
        assert np.isnan(r["hi"]) and np.isnan(r["lo"])

    def test_a_nan_scale_falls_back_to_the_percentile(self):
        x = _skewed(n=1000)
        r = band.rule_bounds(x, float(x.mean()), float("nan"),
                             k=4.0, percentile=99.5)
        assert np.isfinite(r["hi"]) and r["hi_binds"] == "percentile"

    def test_percentiles_outside_the_data_do_not_invert_the_band(self):
        x = _skewed(n=500)
        r = band.rule_bounds(x, float(x.mean()), float(x.std(ddof=1)),
                             k=4.0, percentile=99.5)
        assert r["lo"] < r["hi"]
