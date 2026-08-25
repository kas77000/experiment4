"""Choose the review load in ORDERS, and let the data supply k.

    "We can explain two orders a month. Where does the band go?"

That is the question a desk head actually asks, and it is not the question
`target_flag_rate` answers. A rate is only a load once you know the cell's
volume: 0.5% is twenty orders a month on a 47k book and one order a quarter on
a thin one. Setting a single rate across twelve cells therefore hands the busy
desks all the work and the quiet desks a band that never fires.

A count target is volume-aware by construction. Each cell converts its own
budget into its own rate --

    rate = orders_per_month * months_in_fit_window / n

-- and solves for the k that delivers it, so every cell lands on the same
review load rather than the same percentage.

Two guards keep that safe, and both matter more than the arithmetic:

  1. THE FLOOR. On a thin cell the budget can imply a rate WIDER than the
     nominal one -- 2 a month out of 400 orders a year is 6% -- which would
     give the small desk a tighter band than the large one, exactly backwards.
     The budget may only ever widen a band, never narrow it.

  2. TAIL EVIDENCE. A bound placed at the 99.95th percentile is cut on the
     handful of orders beyond it. Twenty-four of them (2 a month over a year)
     is enough to be worth freezing; six is somebody's bad Tuesday. The count
     is reported either way, and flagged when it is too thin to lean on.
"""

from __future__ import annotations

import numpy as np

# Mean Gregorian month. The fit window is a date range, not a count of
# calendar months, so a 380-day extract is 12.5 months rather than 13.
MONTH_DAYS = 30.4375

# Orders beyond the bound below which the estimate is somebody's anecdote.
MIN_TAIL_OBS = 10


def window_months(d_lo, d_hi) -> float | None:
    """Length of the fit window in months, or None when there are no dates.

    None rather than a default: without a window there is no way to turn a
    per-month budget into a rate, and inventing one would silently produce a
    band nobody chose.
    """
    if d_lo is None or d_hi is None:
        return None
    days = (d_hi - d_lo).days + 1          # inclusive: one day is one day
    return max(days / MONTH_DAYS, 1.0 / MONTH_DAYS)


def target_rate(n: int, per_month: float, months: float) -> float:
    """Share of the fit book that `per_month` reviews implies."""
    if n <= 0 or months <= 0:
        return float("nan")
    return float(per_month) * float(months) / float(n)


def miss_to_flag(lo: float, hi: float, centre: float) -> float:
    """How far from typical an order must land to be flagged, in metric units.

    The nearer bound, because that is the one an order hits first. This is the
    number a trader sanity-checks: "you flag me at fourteen spreads?" is a
    conversation, "you flag me at k = 5.2" is not.
    """
    return float(min(centre - lo, hi - centre))


def solve(x, centre: float, scale: float, *, per_month: float,
          months: float | None, k_floor: float,
          min_tail_obs: int = MIN_TAIL_OBS) -> dict:
    """The k that puts `per_month` orders a month outside the band.

    Returns k, the rate it came from, how many fit-book orders back it, and
    -- when the guards fired -- why the answer is the floor instead.
    """
    out = {"k": float(k_floor), "rate": float("nan"), "n_tail": 0,
           "floored": True, "thin_tail": False, "reason": ""}

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(scale) or scale <= 0:
        out["reason"] = "no usable observations; kept the k floor"
        return out

    if months is None:
        out["reason"] = ("the extract has no usable Date column, so a "
                         "per-month budget cannot be converted into a rate; "
                         "kept the k floor")
        return out

    rate = target_rate(x.size, per_month, months)
    out["rate"] = rate
    if not np.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
        out["reason"] = (f"{per_month:g} a month is more than this cell "
                         f"trades ({x.size:,} orders over {months:.1f} "
                         f"months); kept the k floor")
        return out

    d = np.abs(x - centre) / scale
    k = float(np.quantile(d, 1.0 - rate))
    n_tail = int(round(rate * x.size))
    out["n_tail"] = n_tail
    out["thin_tail"] = n_tail < min_tail_obs

    if not np.isfinite(k) or k <= 0:
        out["reason"] = "the quantile did not resolve; kept the k floor"
        return out

    if k <= k_floor:
        # The budget is looser than the floor: honouring it would NARROW the
        # band. Refuse -- see guard 1 in the module docstring.
        out["reason"] = (f"{per_month:g} a month implies {100 * rate:.2f}% of "
                         f"this cell, which needs only k={k:.2f}; held at the "
                         f"k={k_floor:g} floor so a thin cell does not get a "
                         f"tighter band than a busy one")
        return out

    out["k"], out["floored"] = k, False
    return out
