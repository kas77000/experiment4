"""User-editable configuration.

This is the ONLY file you should need to touch when moving from the synthetic
demo to your real HK VWAP extract. Point COLUMN_MAP at your column names, set
the sign convention, and (optionally) tune the bands.
"""

from dataclasses import dataclass, field
from tca import schema


# ---------------------------------------------------------------------------
# 1) COLUMN MAP  ---  canonical_name -> your_extract_column_name
# ---------------------------------------------------------------------------
# The synthetic generator emits the canonical names directly, so the defaults
# below are identity. For your real data, change the RIGHT-HAND side to match
# your CSV/Parquet headers, e.g.  schema.SLIPPAGE_BPS: "PerfBps".
COLUMN_MAP = {
    schema.ORDER_ID:      "order_id",
    schema.MARKET:        "market",
    schema.ALGO:          "algo",
    schema.BENCHMARK:     "benchmark_type",
    schema.SYMBOL:        "symbol",
    schema.SIDE:          "side",
    schema.SLIPPAGE_BPS:  "slippage_bps",
    schema.SPREAD_BPS:    "spread_bps",
    schema.QUANTITY:      "quantity",
    schema.NOTIONAL:      "notional",
    schema.PCT_ADV:       "pct_adv",
    schema.PARTICIPATION: "participation",
    schema.DURATION_MIN:  "duration_min",
    schema.VOLATILITY:    "volatility",
}


# ---------------------------------------------------------------------------
# 2) SIGN CONVENTION  ---  which direction is "good"?
# ---------------------------------------------------------------------------
# We standardize internally to: HIGHER perf_in_spreads = BETTER (beat benchmark).
#   "positive_is_good": your slippage is already +ve when you beat the benchmark
#   "cost":             your slippage is a COST (+ve = worse); we flip the sign
SLIPPAGE_SIGN = "positive_is_good"


@dataclass(frozen=True)
class Config:
    # --- data quality filters (rows failing these are DROPPED, not flagged) ---
    min_spread_bps: float = 0.1      # spread must be positive & sane
    min_notional: float = 0.0        # drop dust orders below this notional
    max_abs_perf_spreads: float = 25.0  # |perf| beyond this = data error, drop

    # --- difficulty buckets on %ADV ---
    # Orders in the same algo x market but very different sizes are NOT
    # comparable, so we band within size buckets. Edges are in %ADV.
    adv_bucket_edges: tuple = (0.0, 1.0, 5.0, 10.0, 20.0, float("inf"))
    adv_bucket_labels: tuple = ("<1%", "1-5%", "5-10%", "10-20%", ">20%")

    # --- the bands (percentiles of perf_in_spreads within each group) ---
    # RED beyond the outer pair, GREY between, GREEN inside the inner pair.
    # Two-sided by construction: the lower RED tail = underperformance,
    # the upper RED tail = suspiciously good (data/benchmark errors, lucky fills).
    red_percentiles: tuple = (10.0, 90.0)    # outside -> RED (flagged)
    grey_percentiles: tuple = (25.0, 75.0)   # inside  -> GREEN; between -> GREY

    # --- robustness ---
    # Minimum orders needed to trust a group's bands. Groups thinner than this
    # fall back to the pooled algo x market bands, then to a global fallback.
    min_group_n: int = 200

    # Grouping keys for the most specific level.
    group_keys: tuple = (schema.ALGO, schema.MARKET, schema.ADV_BUCKET)


CONFIG = Config()
