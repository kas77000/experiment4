"""The review budget: say how many orders a month you will explain, get k back.

These tests pin the two guards that make a count target safe. Without them a
thin cell asking for 2 a month gets a band NARROWER than 3 sigma, and a band
can be cut on a tail nobody has ever observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tier5 import budget


class TestWindowMonths:
    def test_a_year_of_dates_is_twelve_months(self):
        lo, hi = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-05-31")
        assert budget.window_months(lo, hi) == pytest.approx(12.0, abs=0.1)

    def test_one_month_is_one_month(self):
        lo, hi = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")
        assert budget.window_months(lo, hi) == pytest.approx(1.0, abs=0.05)

    def test_a_single_day_does_not_round_to_zero(self):
        d = pd.Timestamp("2026-07-01")
        assert budget.window_months(d, d) > 0

    def test_no_dates_is_unknown_not_a_guess(self):
        assert budget.window_months(None, None) is None


class TestTargetRate:
    def test_two_a_month_over_a_year_of_47k_orders(self):
        # 2 * 12 = 24 of 46_950
        assert budget.target_rate(46_950, per_month=2, months=12.0) == \
            pytest.approx(24 / 46_950)

    def test_scales_with_the_window_not_just_n(self):
        # Same n, half the window -> each month is twice as busy, so the same
        # 2-a-month budget is a rarer event and the rate halves.
        wide = budget.target_rate(12_000, per_month=2, months=12.0)
        narrow = budget.target_rate(12_000, per_month=2, months=6.0)
        assert narrow == pytest.approx(wide / 2)


def _book(n=46_950, seed=7):
    """A fat-tailed book like the real HK/VWAP one: 64/36 normal mixture."""
    rng = np.random.default_rng(seed)
    heavy = rng.random(n) >= 0.64
    return -0.38 + rng.normal(0.0, np.where(heavy, 4.38, 0.63))


class TestSolve:
    def test_two_a_month_on_a_fat_book_lands_near_five_sigma(self):
        x = _book()
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=12.0, k_floor=3.0)
        assert 4.5 < r["k"] < 6.0
        assert not r["floored"]

    def test_it_covers_almost_everything(self):
        x = _book()
        c, s = x.mean(), x.std(ddof=1)
        r = budget.solve(x, c, s, per_month=2, months=12.0, k_floor=3.0)
        lo, hi = c - r["k"] * s, c + r["k"] * s
        inside = float(np.mean((x >= lo) & (x <= hi)))
        assert inside > 0.999

    def test_it_delivers_the_count_asked_for(self):
        x = _book()
        c, s = x.mean(), x.std(ddof=1)
        r = budget.solve(x, c, s, per_month=2, months=12.0, k_floor=3.0)
        n_out = int(((x < c - r["k"] * s) | (x > c + r["k"] * s)).sum())
        assert n_out / 12 == pytest.approx(2, abs=1)

    def test_a_wider_budget_gives_a_narrower_band(self):
        x = _book()
        c, s = x.mean(), x.std(ddof=1)
        few = budget.solve(x, c, s, per_month=2, months=12.0, k_floor=3.0)
        many = budget.solve(x, c, s, per_month=20, months=12.0, k_floor=3.0)
        assert many["k"] < few["k"]

    # --- guard 1: the floor -------------------------------------------------
    def test_a_thin_cell_is_floored_not_narrowed(self):
        # 400 orders a year: 2 a month is 6% of the book, which is a WIDER
        # rate than k=3 and would hand a small desk a tighter band than a
        # large one. The budget may only widen.
        x = _book(n=400, seed=3)
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=12.0, k_floor=3.0)
        assert r["k"] == 3.0
        assert r["floored"]
        assert "floor" in r["reason"].lower()

    def test_the_floor_is_the_configured_k_not_a_hardcoded_three(self):
        x = _book(n=400, seed=3)
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=12.0, k_floor=4.0)
        assert r["k"] == 4.0

    def test_wanting_more_a_month_than_the_cell_trades_is_floored(self):
        x = _book(n=100, seed=5)
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=50, months=12.0, k_floor=3.0)
        assert r["k"] == 3.0
        assert r["floored"]

    # --- guard 2: tail evidence --------------------------------------------
    def test_it_reports_how_many_orders_back_the_estimate(self):
        x = _book()
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=12.0, k_floor=3.0)
        assert r["n_tail"] == pytest.approx(24, abs=3)

    def test_a_thin_tail_is_flagged_as_thin(self):
        # 3 months of fit: 2 a month is 6 orders, too few to cut a bound on.
        x = _book(n=12_000, seed=9)
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=3.0, k_floor=3.0)
        assert r["thin_tail"]

    def test_a_year_of_fit_is_not_flagged_as_thin(self):
        x = _book()
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=12.0, k_floor=3.0)
        assert not r["thin_tail"]

    # --- guard 3: no dates --------------------------------------------------
    def test_without_a_window_it_refuses_to_guess(self):
        x = _book()
        r = budget.solve(x, x.mean(), x.std(ddof=1),
                         per_month=2, months=None, k_floor=3.0)
        assert r["k"] == 3.0
        assert r["floored"]
        assert "date" in r["reason"].lower()

    def test_an_empty_cell_falls_back_to_the_floor(self):
        r = budget.solve(np.array([]), 0.0, 1.0,
                         per_month=2, months=12.0, k_floor=3.0)
        assert r["k"] == 3.0
        assert r["floored"]


class TestMissBy:
    def test_it_says_how_far_an_order_must_miss_to_be_flagged(self):
        # The sanity check a trader applies: a band of -14..13 around a centre
        # of -0.4 means missing by ~13.6 spreads either way.
        assert budget.miss_to_flag(-14.3, 13.5, -0.38) == pytest.approx(13.88,
                                                                        abs=0.1)
