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
# The band, stated the way the desk states it:
#
#     hi = MAX(mean + K_SIGMA * sigma,   P(PERCENTILE_PCT))
#     lo = MIN(mean - K_SIGMA * sigma,   P(100 - PERCENTILE_PCT))
#
# Set here rather than passed on the command line, so every run, every region,
# every strategy and every refit applies the same rule without anybody having
# to remember a flag.
#
# The sigma term is LITERAL. centre + k*scale, nothing solved and nothing
# adjusted, so "mean plus four sigma" is visibly what it says on the page.
# An earlier version of this file solved for the k that delivered a coverage
# target, which is a defensible rule and a different one: it printed k = 4.97
# where the desk had asked for four, and read as though sigma had been
# replaced rather than used.
#
# TWO PROPERTIES WORTH KNOWING, both easy to lose in conversation:
#
#   IT IS PER SIDE. Slippage is skewed -- a book misses badly far more often
#   than it beats badly -- so forcing both tails through one multiple makes
#   the band wrong on at least one of them. Each side takes whichever of its
#   own two candidates is wider, and the two can bind differently.
#
#   P99.5 IS NOT "99.5% COVERAGE". P99.5 leaves 0.5% of orders above it in the
#   upper tail alone. A 99.5%-coverage band splits that 0.5% across BOTH
#   tails and therefore sits nearer P99.75. The two read identically in a
#   meeting and differ by a real amount on the page.
#
# WHY K_SIGMA IS NINE AND NOT THREE OR FOUR.
#
# Under a normal distribution 4 sigma covers 99.994% -- about three orders a
# YEAR on a 47k book -- which is why "four sigma" sounds like "everything".
# The real HK VWAP book is not normal, and not by a little:
#
#     K_SIGMA   band (spreads)     coverage    to review
#        3      -8.26 ..  7.76      97.60%     94 / month
#        4     -10.93 .. 10.43      98.66%     52 / month     <- 213x a normal book
#        5     -13.60 .. 13.10      99.33%     26 / month
#        6     -16.27 .. 15.77      99.73%     11 / month
#        7     -18.94 .. 18.44      99.88%      4.6 / month
#        8     -21.61 .. 21.11      99.95%      2.1 / month
#        9     -24.28 .. 23.78      99.98%      0.8 / month    <- set
#
# At k = 4 this book leaves 1.34% outside where a normal leaves 0.006%. The
# multiple that means "essentially everything" HERE is nine, and the reason is
# visible in any curve.png: the observed density peaks around 0.44 against the
# fitted normal's 0.15, so sigma is simultaneously too large for the middle of
# the book and too small for its tails. A sigma multiple calibrated on a
# Gaussian intuition does not survive that shape.
#
# ONE CONSEQUENCE WORTH KNOWING. At K_SIGMA = 9 the sigma term (+/-24) is far
# wider than P99.5 (+/-11.7), so the percentile can never win the MAX and the
# rule is in practice a pure 9-sigma band. The percentile stays as a dormant
# net: it fires only if some cell's tail is so heavy that its own P99.5 lands
# beyond nine sigma. Every fit prints both candidates and the winner per side,
# so whether the net ever fires is a fact on the page, not an assumption.
# None drops the sigma term entirely, leaving a band cut from the percentiles
# alone. That is the only rule here that assumes nothing about the shape of the
# distribution -- no centre, no scale, no implied symmetry -- which on a book
# this far from normal is a genuine argument rather than a shortcut. What it
# costs on the real HK VWAP book, with no sigma term:
#
#     PERCENTILE_PCT   band (spreads)     outside    to review    in sigma
#         99.0         -9.07 ..   8.66     2.00%     78 / month      3.3
#         99.5        -12.06 ..  11.72     1.00%     39 / month      4.5
#         99.9        -17.41 ..  16.75     0.20%      7.8 / month    6.4
#        99.95        -19.46 ..  18.98     0.10%      4.0 / month    7.2
#
# Note P99.95 lands almost exactly on the shipped +/-20 absolute band. The two
# rules agree on this book today and would diverge as it changes: the
# percentile follows the book, the absolute bound does not.
K_SIGMA = 9.0
PERCENTILE_PCT = 99.5

# --- THE SHIPPED BAND ----------------------------------------------------
# An ABSOLUTE band, in the metric's own units: -20 .. +20 spreads.
#
# This is a POLICY, not an estimate, and that is the whole point of it. The
# fitted rules above all move with the book -- a quarter of worse execution
# raises the mean and inflates sigma, so a sigma band silently WIDENS exactly
# when performance deteriorates, and the threshold forgives the drift it was
# meant to catch. An absolute bound cannot do that.
#
# Symmetric about ZERO, not about the book's mean. For a spread-normalised
# metric zero is a real reference point -- the order matched interval VWAP
# exactly -- so "twenty spreads either side of matching the benchmark" is a
# sentence that means something on its own, without knowing what the book did
# last year.
#
# What it costs on the real HK VWAP book (centre -0.36, sigma 2.67): +/-20
# spreads is about 7.6 sigma, which leaves roughly one order a quarter outside.
# The fit still computes centre and scale and prints the implied multiple
# beside the band, so how loose or tight this policy is ON THIS BOOK stays
# visible rather than assumed.
#
# Set to None to fall back to the fitted rule (MAX(mean +/- K_SIGMA*sigma,
# P(PERCENTILE_PCT))) documented above.
BAND_ABS_SPREADS = 20.0

# The floor for the OPT-IN overrides only, and deliberately not K_SIGMA.
#
# k_sigma used to serve both jobs -- the shipped band width and the floor under
# an explicit --target-review-count -- which was harmless while it was 4 and
# actively wrong at 9: a desk asking for two orders a month would have been
# floored to a nine-sigma band and got nothing, with the flag appearing to work.
# An explicit override is a deliberate instruction and must deliver what it
# says; this floor exists only to stop a thin cell talking itself into a band
# NARROWER than a conventional one.
BUDGET_K_FLOOR = 3.0

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
    k_sigma: float | None = K_SIGMA

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
    target_flag_rate: float | None = None

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

    # The percentile half of the fitted rule. None switches it off, leaving a
    # plain mean +/- k*sigma band.
    band_percentile: float | None = PERCENTILE_PCT

    # The absolute band, in metric units. Outranks the fitted rules; None falls
    # back to them.
    band_abs: float | None = BAND_ABS_SPREADS

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
