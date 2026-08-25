"""Tier 5 knobs --- a Gaussian band at mu +/- k*sigma.

This is the statistical-process-control control limit: assume the metric is
normally distributed, estimate its centre and scale, and accept everything
within k scales of the centre.

k = 3 is not an arbitrary choice. Under a normal distribution 99.73% of
observations fall within +/- 3 sigma, so picking k is picking a flag rate:
0.27% of orders, about 1 in 370. Whether the book delivers that is measured
rather than assumed -- see normality.py.
"""

from dataclasses import dataclass

from tca import schema

# --- fitted grouping levels, most general first --------------------------
LEVEL_ALL = "ALL"
LEVEL_ALGO = "algo"
LEVEL_ADV = "adv_bucket"
LEVEL_ALGO_ADV = "algo_x_adv_bucket"

# level -> the columns that define a group at that level. The empty tuple is
# the whole book, which is the headline number.
LEVEL_KEYS = {
    LEVEL_ALL: (),
    LEVEL_ALGO: (schema.ALGO,),
    LEVEL_ADV: (schema.ADV_BUCKET,),
    LEVEL_ALGO_ADV: (schema.ALGO, schema.ADV_BUCKET),
}

# --- THE STANDARD --------------------------------------------------------
# The share of orders a band is meant to contain. This is the ONE number the
# whole workflow turns on, and it is set here rather than passed on the command
# line so that every run, every region, every strategy and every refit applies
# the same rule without anybody having to remember a flag.
#
# Read it as the sentence you would say out loud: "our threshold is the 99.9th
# percentile of the last year's execution." That sentence is defensible in a
# way "k = 4.97" is not, and it does not change when the book does.
#
# WHY A COVERAGE AND NOT A SIGMA MULTIPLE. They are the same statement only
# under a normal distribution -- "3 sigma" and "99.73% coverage" are two ways
# of saying one thing, and 99.9% is 3.29 sigma. Execution books are not normal,
# so the two come apart, and only one of them survives the split:
#
#   fix the MULTIPLE at 3.0 -> the coverage floats. On a real HK VWAP book it
#                              lands at 97.6%, and the threshold silently means
#                              something different in every cell.
#   fix the COVERAGE at 99.9% -> the multiple floats. It came out at 4.97 on
#                              that book and 3.29 if the book had been normal.
#                              The rule holds; k reports what it cost.
#
# The second is the systematic one, because the promise is what you defend and
# the multiple is only ever the means. k is now an OUTPUT of every fit, printed
# beside the k a normal book would have needed -- and the gap between those two
# numbers is this book's non-normality, in one figure.
#
# What it costs, on a 47k-order HK VWAP book (sd 2.67 spreads):
#
#     coverage    k if normal    k here    band (spreads)    to review
#      99.00%        2.58         3.61     -9.98 ..  9.27     39/month
#      99.73%        3.00         4.44    -12.19 .. 11.47     11/month
#      99.90%        3.29         4.97    -13.59 .. 12.87      4/month   <- set
#      99.95%        3.48         5.27    -14.40 .. 13.68      2/month
#
# Change THIS ONE NUMBER to move the whole desk. Nothing else needs editing,
# and no command line changes.
COVERAGE_PCT = 99.5

# ...and the band is never tighter than this many scales either way, whatever
# the coverage rule works out to. The shipped rule is the WIDER of the two:
#
#     k = max(K_FLOOR, k that delivers COVERAGE_PCT)
#
# The two bounds catch opposite failures, which is why neither alone is enough:
#
#   COVERAGE widens a band whose tail is fatter than a Gaussian assumes. On the
#     real HK book, k = 4 alone leaves ~0.9% outside rather than the 0.006% it
#     advertises, so a fixed multiple quietly under-delivers exactly where the
#     risk is.
#   THE FLOOR stops a well-behaved cell being handed an absurdly tight band. A
#     near-normal cell reaches 99.5% at k = 2.95, and shipping that beside HK's
#     4.06 would give the quietest desk the harshest threshold -- the opposite
#     of a common standard.
#
# What binds, per cell shape:
#
#     cell                        k@99.5%   floor   k used    binds
#     HK / VWAP  (very fat)          4.06     4.0     4.06    coverage
#     JP / VWAP  (mildly fat)        3.95     4.0     4.00    floor
#     AU / VWAP  (near-normal)       2.95     4.0     4.00    floor
#
# Both numbers are reported per cell, so which one bound is never a guess.
K_FLOOR = 4.0

# classical -> mean / standard deviation   (the method as requested)
# robust    -> median / 1.4826 * MAD       (the same band, tail-resistant)
ESTIMATORS = ("classical", "robust")

# What a band bound MEANS, per metric. Printed next to every lo/hi and stamped
# into the band file, because "-2.31 .. 1.87" is unreadable without it and
# quietly wrong if a reader assumes bps.
UNITS = {
    schema.SLIPPAGE_BPS: "bps",
    schema.PERF_IN_SPREADS: "spreads",
    schema.PERF_NORM: "sigma",
}


def units_of(metric: str) -> str:
    return UNITS.get(metric, "")


@dataclass(frozen=True)
class Tier5Config:
    # --- the band ---------------------------------------------------------
    # How many scales either side of the centre. 3.0 promises 0.27% flagged
    # IF the data is normal, which is the assumption the report tests.
    k_sigma: float = K_FLOOR

    # --- or: pick the review load and let the data supply k ---------------
    # A percentage. When set, k_sigma is IGNORED and each cell solves for the
    # k that puts exactly this share of its fit book outside the band.
    #
    # Why this exists: k = 3 promises 0.27% only under normality, and no
    # execution book is normal. A real one is sharply peaked in the middle and
    # heavy in the tails, so k = 3 routinely flags five to ten times what it
    # advertises. There are two honest responses to that and only two --
    # keep k = 3 and accept the real rate, or state the rate you want and
    # report the k it took. This is the second. What is NOT honest is calling
    # a band "3 sigma" while the tails mean something entirely different.
    #
    # The number is a resourcing decision, not a statistical one: 0.5% of a
    # 47k book is about 20 orders a month to explain. Say what the desk can
    # actually review and set that.
    #
    # Note the fit-book rate then equals the target BY CONSTRUCTION, so it
    # measures nothing. The out-of-sample number from tier5.score is still a
    # measurement, and is the one that carries information.
    target_flag_rate: float | None = round(100.0 - COVERAGE_PCT, 10)

    # --- or: pick the review load in ORDERS and let each cell supply its own
    # k. Orders per cell per MONTH. When set, both k_sigma and
    # target_flag_rate are ignored and every cell converts this budget into
    # its own rate using its own volume and fit window -- see budget.py.
    #
    # Why a count rather than a percentage: a rate is only a workload once you
    # know the volume. 0.5% is twenty orders a month on a 47k book and one a
    # quarter on a thin one, so one rate across twelve cells gives the busy
    # desks all the work. A count gives every desk the same load, which is
    # what "we can explain two a month" actually means.
    #
    # k_sigma becomes a FLOOR: the budget may widen a band, never narrow one.
    # Without that a 400-order cell asking for 2 a month would be handed a
    # 6% flag rate, a tighter band than the busy desk beside it.
    target_review_count: float | None = None

    # --- which metric to band --------------------------------------------
    #   PERF_IN_SPREADS -> slippage / spread          (the default)
    #   SLIPPAGE_BPS    -> raw slippage, in bps
    #   PERF_NORM       -> slippage / sigma_expected
    #
    # Spread-normalised is the default, so lo/hi come out in SPREADS. A 12 bps
    # miss is noise in a wide Indian small cap and a serious miss in a tight
    # Japanese large cap; dividing by the spread puts every name and every
    # region on one scale before the band is fitted, which is what makes a
    # single number defensible across four markets.
    #
    # The extract supplies this column already divided -- see
    # config.METRIC_BY_STRATEGY for which source column each strategy uses.
    metric: str = schema.PERF_IN_SPREADS

    # Which estimator SCORES orders. Both are always computed and reported;
    # this only picks the one the zones are cut on.
    estimator: str = "classical"

    # Which fitted level supplies each order's band. LEVEL_ALL is the request
    # taken literally: one range for the whole year. Anything else falls back
    # to LEVEL_ALL when the matched cell is untrusted.
    score_level: str = LEVEL_ALL

    # --- robustness -------------------------------------------------------
    # Minimum orders before a group's sigma is trusted. A sigma from 44 orders
    # is not a threshold. Thin groups stay in the table marked trusted=False,
    # so they are visible rather than silently absent.
    min_group_n: int = 200

    # Levels to fit. Every level lands in band_table.csv.
    group_levels: tuple = (LEVEL_ALL, LEVEL_ALGO, LEVEL_ADV, LEVEL_ALGO_ADV)

    # --- review queue -----------------------------------------------------
    # Materiality gate. 0 = off, so review_required == flagged.
    min_notional_review: float = 0.0

    # Write outputs/tier5/qq_plot.png when matplotlib is available.
    make_qq_plot: bool = True


CONFIG = Tier5Config()
