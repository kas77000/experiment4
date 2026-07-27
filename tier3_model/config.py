"""Tier 3 knobs --- quantile-regression cost model + residual scoring."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier3Config:
    # --- the conditional band --------------------------------------------
    # Quantiles of perf_norm to regress on difficulty. The fitted tau_lo and
    # tau_hi surfaces ARE the threshold; tau_med is the expected cost.
    #
    # Deliberately ASYMMETRIC. Underperformance is what you act on, so the lower
    # gate carries most of the budget; the upper gate exists to surface data and
    # benchmark errors, which are rarer and cheaper to check.
    #
    # THIS IS THE DIAL THAT SETS YOUR FLAG RATE. It is a review-capacity
    # decision, not a discovery: you get back what you dial in, near-exactly.
    # Measured out-of-sample on the live column format:
    #
    #     tau_lo / tau_hi     nominal   flagged
    #     0.02   / 0.99        3.00%     3.12%
    #     0.01   / 0.995       1.50%     1.77%   <- current
    #     0.005  / 0.9975      0.75%     0.89%
    #     0.0025 / 0.999       0.35%     0.62%   <- calibration starts to drift
    #
    # So "only 1.8% of orders fell outside the range" says where the threshold
    # was set, NOT how well anyone traded. The number that means something is
    # the calibration table (realized vs nominal), which says the gate will
    # still hold on next quarter's orders.
    #
    # Below ~0.0025 the far tail has too few observations to fit reliably and
    # realized coverage drifts above nominal -- do not go tighter than that.
    tau_lo: float = 0.01
    tau_med: float = 0.50
    tau_hi: float = 0.995

    # --- how the algo enters the model -----------------------------------
    #   "absorb"  algo dummies IN the model. Each algo is judged against its own
    #             norm, so flags mean "bad for this algo". Systematic algo
    #             underperformance is then caught by the slice t-tests, not here.
    #   "expose"  no algo dummies. A structurally worse algo flags on every
    #             order. Use when you are comparing algos, not orders.
    algo_effect: str = "absorb"

    # Include the size x urgency interaction (sqrt(%ADV) * log(POV)).
    include_interactions: bool = True

    # --- honesty about in-sample optimism ---------------------------------
    # K-fold cross-fitting: each order is scored by a model that never saw it.
    # Set to 1 to disable (faster, but the flag rate will look better than it is).
    n_folds: int = 5

    # Training-set trim: drop this fraction from EACH tail of perf_norm before
    # fitting. Trimmed rows are still SCORED -- this is about not letting data
    # errors drag the surfaces, not about hiding outliers.
    #
    # DEFAULT IS OFF, deliberately. Two reasons:
    #   1. Quantile regression minimizes check loss, so an extreme y has BOUNDED
    #      influence on the fit -- unlike OLS, it does not need trimming to be
    #      robust. Gross data errors are already dropped by pipeline.clean().
    #   2. Trimming the tail biases exactly the quantile you are trying to
    #      estimate. Measured on the demo book, out-of-sample coverage against a
    #      2.0% nominal gate degrades monotonically:
    #          trim 0.000 -> 2.02%   trim 0.002 -> 2.29%
    #          trim 0.001 -> 2.15%   trim 0.005 -> 2.56%
    #      Turn it on only if your extract has errors clean() cannot catch, and
    #      re-read the calibration table afterwards.
    #
    # Whatever you set is clamped to stay well inside the fitted taus.
    fit_trim_quantile: float = 0.0

    # Minimum training rows before the regression is attempted at all.
    min_fit_n: int = 500

    # --- severity tiers ---------------------------------------------------
    # Outside the band at all -> MONITOR (logged, trended, no action).
    # |z| beyond escalate_z   -> ESCALATE (written justification).
    escalate_z: float = 3.0

    # Materiality gate: `flagged` is everything outside the band; an order also
    # needs notional >= this to become `review_required`. The idea is that an
    # order 3 sigma off on a tiny notional is statistically real but costs
    # almost nothing in money terms.
    #
    # DEFAULT 0 = OFF, so review_required == flagged: everything outside the
    # threshold goes in the queue. Turn it on only once you have confirmed what
    # currency `$Mln` is denominated in -- the gate is in the same units as
    # notional, so a value meant as HK$1m would silently cut a large part of the
    # book if the extract is in USD millions.
    min_notional_review: float = 0.0

    # --- backend ----------------------------------------------------------
    #   "auto"      quantile regression if statsmodels imports, else empirical
    #   "quantreg"  force regression (raises if statsmodels is missing)
    #   "empirical" force bucketed percentiles of perf_norm (no statsmodels)
    backend: str = "auto"

    # --- cause attribution percentiles -----------------------------------
    # Evidence thresholds are taken from the book itself, so they travel to new
    # datasets without retuning.
    rev_hi_pct: float = 85.0       # reversion/sigma above this -> own-impact evidence
    pov_hi_pct: float = 75.0       # participation above this (within algo)
    passive_lo_pct: float = 15.0   # passive fill below this (within algo)
    auction_lo_pct: float = 10.0   # auction share below this (within market)
    momentum_hi_pct: float = 90.0  # |momentum|/sigma above this


CONFIG = Tier3Config()
