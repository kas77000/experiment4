"""Canonical column names used everywhere downstream of the loader.

Your raw extract can call these columns whatever it likes; `config.COLUMN_MAP`
translates *your* names into these. Nothing else in the codebase should ever
reference a vendor-specific column name.

Fields are grouped by how badly we need them:
  ESSENTIAL  - a row is unusable without these
  DIFFICULTY - drive the expected-cost model (Tier 3); optional but valuable
  DIAGNOSTIC - explain *why* an order was bad; optional, used when present
"""

# --- identity / grouping -------------------------------------------------
ORDER_ID = "order_id"
MARKET = "market"          # e.g. "HK"
ALGO = "algo"              # e.g. "VWAP", "VWAP_Passive"
BROKER = "broker"          # optional; enables broker-level slice analysis
BENCHMARK = "benchmark_type"
SYMBOL = "symbol"
SIDE = "side"              # "buy" / "sell" (or +1 / -1)

# --- performance inputs --------------------------------------------------
SLIPPAGE_BPS = "slippage_bps"   # signed performance vs benchmark, in bps
SPREAD_BPS = "spread_bps"       # quoted/effective spread at the time, in bps

# --- difficulty features (bucketing in Tier 2, regression in Tier 3) -----
QUANTITY = "quantity"
NOTIONAL = "notional"
PCT_ADV = "pct_adv"             # order size as % of avg daily volume
PARTICIPATION = "participation" # realized participation rate. Source units are
                                # declared by DataConfig.participation_unit;
                                # normalize_units converts to a FRACTION (0-1).
DURATION_MIN = "duration_min"   # minutes from first to last fill
VOLATILITY = "volatility"       # DAILY volatility. Source units are declared by
                                # DataConfig.volatility_unit; pipeline.normalize_units
                                # converts to BPS, which is what everything downstream
                                # assumes (180 = 1.8%/day).

# --- diagnostic inputs (optional; Tier 3 uses them to attribute cause) ---
REVERSION_BPS = "reversion_bps"        # post-trade reversion. +ve = the price moved
                                       # back AGAINST your trade direction after you
                                       # stopped, i.e. you caused the impact. That is
                                       # a bad sign for execution, not a good one.
PASSIVE_FILL_PCT = "passive_fill_pct"  # fraction of qty filled passively (0-1)
AUCTION_PCT = "auction_pct"            # fraction of qty done in auctions (0-1)
MOMENTUM_BPS = "momentum_bps"          # market drift over the interval, signed vs your side

# --- Tier 4 derived ------------------------------------------------------
# Slippage exactly as the extract reported it, kept because Tier 4 overwrites
# SLIPPAGE_BPS with the de-biased version.
REPORTED_SLIPPAGE_BPS = "reported_slippage_bps"
DILUTION_FACTOR = "dilution_factor"   # 1/(1-participation)
DILUTION_CAPPED = "dilution_capped"   # participation exceeded the cap

# --- derived (produced by the pipeline, never present in raw data) -------
PERF_IN_SPREADS = "perf_in_spreads"      # Tier 1/2 metric: slippage / spread
SIGMA_EXPECTED_BPS = "sigma_expected_bps"  # Tier 3 noise unit: spread & vol-time in quadrature
PERF_NORM = "perf_norm"                  # Tier 3 metric: slippage / sigma_expected
NORM_BASIS = "norm_basis"                # "spread+vol" or "spread_only" (per row)
ADV_BUCKET = "adv_bucket"

# --- demo-only truth labels (synthetic_data emits these; real data won't) -
TRUE_OUTLIER = "_true_outlier"
TRUE_CAUSE = "_true_cause"
DEMO_TRUTH = [TRUE_OUTLIER, TRUE_CAUSE]

# Essential columns a row MUST have to be usable.
ESSENTIAL = [ORDER_ID, MARKET, ALGO, SLIPPAGE_BPS, SPREAD_BPS]

# Columns the Tier 3 cost model will use if they are present.
DIFFICULTY = [PCT_ADV, PARTICIPATION, DURATION_MIN, VOLATILITY, SPREAD_BPS]

# Columns the Tier 3 diagnostics will use if they are present.
DIAGNOSTIC = [REVERSION_BPS, PASSIVE_FILL_PCT, AUCTION_PCT, MOMENTUM_BPS]

# Numeric columns to coerce on load.
NUMERIC = [
    SLIPPAGE_BPS, SPREAD_BPS, QUANTITY, NOTIONAL,
    PCT_ADV, PARTICIPATION, DURATION_MIN, VOLATILITY,
    REVERSION_BPS, PASSIVE_FILL_PCT, AUCTION_PCT, MOMENTUM_BPS,
]
