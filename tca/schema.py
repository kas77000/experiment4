"""Canonical column names used everywhere downstream of the loader.

Your raw extract can call these columns whatever it likes; `config.COLUMN_MAP`
translates *your* names into these. Nothing else in the codebase should ever
reference a vendor-specific column name.
"""

# --- identity / grouping -------------------------------------------------
ORDER_ID = "order_id"
MARKET = "market"          # e.g. "HK"
ALGO = "algo"              # e.g. "VWAP", "VWAP_Passive"
BENCHMARK = "benchmark_type"
SYMBOL = "symbol"
SIDE = "side"              # "buy" / "sell" (or +1 / -1)

# --- performance inputs --------------------------------------------------
SLIPPAGE_BPS = "slippage_bps"   # signed performance vs benchmark, in bps
SPREAD_BPS = "spread_bps"       # the spread measure we normalize by, in bps

# --- difficulty features (used for bucketing now, regression later) ------
QUANTITY = "quantity"
NOTIONAL = "notional"
PCT_ADV = "pct_adv"             # order size as % of avg daily volume
PARTICIPATION = "participation" # realized participation rate (0-1)
DURATION_MIN = "duration_min"
VOLATILITY = "volatility"

# --- derived (produced by the pipeline, never in raw data) ---------------
PERF_IN_SPREADS = "perf_in_spreads"
ADV_BUCKET = "adv_bucket"

# Essential columns a row MUST have to be usable.
ESSENTIAL = [ORDER_ID, MARKET, ALGO, SLIPPAGE_BPS, SPREAD_BPS]

# Numeric columns to coerce on load.
NUMERIC = [
    SLIPPAGE_BPS, SPREAD_BPS, QUANTITY, NOTIONAL,
    PCT_ADV, PARTICIPATION, DURATION_MIN, VOLATILITY,
]
