"""Tier 2 knobs --- percentile bands within peer groups."""

from dataclasses import dataclass
from tca import schema


@dataclass(frozen=True)
class Tier2Config:
    # --- the band ---------------------------------------------------------
    # Percentiles of the metric within each group. Inside -> acceptable;
    # outside -> flagged. Two-sided by construction: below the lower bound is
    # underperformance, above the upper bound is suspiciously good (stale
    # marks, benchmark errors, lucky fills).
    #
    # NOTE the arithmetic: p10/p90 flags ~20% of the book BY CONSTRUCTION.
    # If a human has to justify each flag, that is an unworkable queue --
    # (2, 98) puts you at ~4%, which is the range production exception reports
    # actually run at.
    range_percentiles: tuple = (2.0, 98.0)

    # --- which metric to band --------------------------------------------
    #   PERF_IN_SPREADS -> slippage / spread        (the classic)
    #   PERF_NORM       -> slippage / sigma_expected (spread AND vol-over-horizon)
    # PERF_NORM is strictly better for interval benchmarks worked over hours.
    metric: str = schema.PERF_IN_SPREADS

    # --- robustness -------------------------------------------------------
    # Minimum orders needed to trust a group's band. Thinner groups fall back
    # to the pooled algo x market band, then to a global band.
    min_group_n: int = 200

    # Grouping keys for the most specific level.
    group_keys: tuple = (schema.ALGO, schema.MARKET, schema.ADV_BUCKET)

    # Materiality gate for the review queue.
    min_notional_review: float = 1_000_000.0   # HKD


CONFIG = Tier2Config()
