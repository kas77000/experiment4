"""Shared data contract --- the ONLY file you edit to point this at real data.

Currently wired for an extract with these columns:

    aggrTgtId  -> order_id
    Strategy   -> algo
    Sym        -> market       (last two characters, e.g. "0700 HK" -> "HK")
    Pvwap      -> slippage_bps (slippage vs interval VWAP, in bps)
    Sprd       -> spread_bps   (bps)
    %Adv       -> pct_adv      (percent)
    Vol        -> volatility   (Parkinson daily vol, in PERCENT -> scaled to bps)
    PR         -> participation (percent -> scaled to a fraction)
    Dur        -> duration_min  (minutes: (endtime - starttime) / 60)
    $Mln       -> notional     (x 1,000,000)
    #Shares    -> quantity     (x 1,000)

Run `python check_extract.py your.csv` BEFORE anything else. It reads the file
and tells you what to set for the two things a column name cannot tell you:
SLIPPAGE_SIGN and the unit settings.

Everything here is tier-agnostic. Each tier's own thresholds live in its folder:

    tier1_fixed/config.py        fixed bps / spread-multiple limits
    tier2_percentile/config.py   percentile band settings
    tier3_model/config.py        cost-model + z-score settings
"""

from dataclasses import dataclass

import pandas as pd

from tca import schema


# ---------------------------------------------------------------------------
# 1) PRE-TRANSFORM  ---  anything a rename cannot express
# ---------------------------------------------------------------------------
# Runs on the raw frame BEFORE the column rename below. Unit scaling and derived
# fields go here. Every step is guarded on the source column being present,
# because this also runs against the synthetic demo (which has none of them).

MLN = 1_000_000.0     # $Mln  is in millions
LOT = 1_000.0         # #Shares is in thousands


def PRE_TRANSFORM(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Market is the venue suffix of the ticker: "0700 HK" -> "HK", "7203 JT" -> "JT".
    if "Sym" in df.columns:
        df["_market"] = (df["Sym"].astype(str).str.strip()
                                  .str[-2:].str.upper())
        # Keep the full ticker too -- it is the natural clustering key and makes
        # the per-symbol slice tests possible.
        df["_symbol"] = df["Sym"].astype(str).str.strip()

    if "$Mln" in df.columns:
        df["_notional"] = pd.to_numeric(df["$Mln"], errors="coerce") * MLN

    if "#Shares" in df.columns:
        df["_quantity"] = pd.to_numeric(df["#Shares"], errors="coerce") * LOT

    return df


# ---------------------------------------------------------------------------
# 2) COLUMN MAP  ---  canonical_name -> source column name
# ---------------------------------------------------------------------------
# Names starting with "_" are produced by PRE_TRANSFORM above. Columns your
# extract does not have are simply skipped; see the degradation table at the
# bottom of this file for what each absence costs you.
COLUMN_MAP = {
    schema.ORDER_ID:      "aggrTgtId",
    schema.ALGO:          "Strategy",
    schema.MARKET:        "_market",       # <- derived from Sym
    schema.SYMBOL:        "_symbol",       # <- derived from Sym
    schema.SLIPPAGE_BPS:  "Pvwap",
    schema.SPREAD_BPS:    "Sprd",
    schema.PCT_ADV:       "%Adv",
    schema.VOLATILITY:    "Vol",
    schema.PARTICIPATION: "PR",
    schema.DURATION_MIN:  "Dur",
    schema.NOTIONAL:      "_notional",     # <- $Mln x 1e6
    schema.QUANTITY:      "_quantity",     # <- #Shares x 1e3

    # --- not in your extract yet; left mapped so they light up automatically
    #     the day they appear. See "What to ask for next" in the README.
    schema.BROKER:        "broker",
    schema.SIDE:          "side",
    schema.BENCHMARK:     "benchmark_type",
    schema.REVERSION_BPS:    "reversion_bps",
    schema.PASSIVE_FILL_PCT: "passive_fill_pct",
    schema.AUCTION_PCT:      "auction_pct",
    schema.MOMENTUM_BPS:     "momentum_bps",
}


# ---------------------------------------------------------------------------
# 3) SIGN CONVENTION  ---  which direction is "good"?
# ---------------------------------------------------------------------------
# We standardize internally to: HIGHER = BETTER (beat the benchmark).
#   "positive_is_good": Pvwap is already +ve when you beat interval VWAP
#   "cost":             Pvwap is a COST (+ve = worse); we flip the sign
#
# GET THIS RIGHT. It is the one setting that silently inverts every result --
# the flag rate looks identical either way, but you flag your best orders
# instead of your worst. `check_extract.py` infers it from the data by testing
# whether large orders perform worse (they must).
SLIPPAGE_SIGN = "positive_is_good"


@dataclass(frozen=True)
class DataConfig:
    """Cleaning rules, units and the normalizing noise unit. Shared by all tiers."""

    # --- units in the source extract --------------------------------------
    # Internally volatility is DAILY vol in BPS and pct_adv is a PERCENT.
    #   volatility_unit: "bps" (180) | "pct" (1.8) | "fraction" (0.018)
    #   pct_adv_unit:    "pct" (3.5) | "fraction" (0.035)
    # Wrong settings rescale sigma_expected silently. check_extract.py infers
    # both from the magnitudes in your file and cross-checks them in section 4.
    #
    # `Vol` here is PARKINSON daily volatility in percent, so "pct".
    #
    # Parkinson is a good input for this. It is a high-low RANGE estimator, so
    # (a) it is far more efficient than close-to-close at the same sample size,
    # which matters when you are scaling a per-order band, and (b) it measures
    # INTRADAY range only, excluding overnight gaps -- which is exactly the
    # right exposure for an order worked inside one session against an interval
    # benchmark. Close-to-close vol would import gap risk the order never took.
    #
    # Two known biases, both benign here: Parkinson assumes no drift, and
    # discrete sampling makes the observed range understate the true one, so it
    # runs slightly LOW. Since it is used as a scale and the cost model carries
    # a `log_vol` feature that can re-scale it, a constant proportional bias is
    # absorbed by the fit rather than passed through to the band.
    volatility_unit: str = "pct"
    pct_adv_unit: str = "pct"

    # Participation ("PR") is normalized to a FRACTION internally, unlike the
    # two above -- that is what the code clips and formats against.
    #   participation_unit: "pct" (15 = 15% of volume) | "fraction" (0.15)
    #
    # Worth knowing: the cost model is INVARIANT to this setting, because the
    # feature is log(participation) and a constant scale factor only shifts the
    # intercept. The cause rules rank participation, which is scale-free too.
    # So getting it wrong will not corrupt your thresholds -- it will only make
    # explain_order() print "POV 1520.0%". Set it right anyway.
    participation_unit: str = "pct"

    # --- data quality filters (rows failing these are DROPPED, not flagged) ---
    min_spread_bps: float = 0.1         # spread must be positive & sane
    min_notional: float = 0.0           # drop dust orders below this notional
    max_abs_perf_spreads: float = 25.0  # |slippage/spread| beyond this = data error

    # --- the normalizing noise unit (Tier 3's denominator) -----------------
    # sigma_expected = sqrt( (k_spread*spread)^2 + (w_vol*vol*sqrt(T))^2 )
    #
    # Rationale: for a short, small order the natural scale of slippage is the
    # spread. For an order worked over hours against an interval benchmark the
    # natural scale is volatility over the horizon. Combining in quadrature lets
    # each dominate where it should, with no switch to hand-tune.
    k_spread: float = 0.5               # half-spread as the spread-side scale
    vol_horizon_weight: float = 0.20    # interval VWAP tracks the benchmark, so only
                                        # a fraction of full sigma*sqrt(T) shows up.
                                        # Raise toward 1.0 for ARRIVAL/IS benchmarks.
    # HK continuous session: 09:30-12:00 + 13:00-16:00. If you trade several
    # venues this is a mild misspecification for the others (JT is 300 min, AU
    # 360), but it enters inside a square root and the model's `log_dur` feature
    # absorbs a constant proportional error, so it is not worth a lookup table.
    minutes_per_day: float = 330.0

    # Per-row fallback for orders whose `Dur` is null. Only fills gaps -- real
    # durations from the extract are always used where present. Set to None to
    # leave null-duration rows on spread-only normalization instead.
    default_duration_min: float | None = 120.0

    min_sigma_bps: float = 0.5          # floor, so we never divide by ~0

    # --- difficulty buckets on %ADV (Tier 1 reporting, Tier 2 grouping) ----
    adv_bucket_edges: tuple = (0.0, 1.0, 5.0, 10.0, 20.0, float("inf"))
    adv_bucket_labels: tuple = ("<1%", "1-5%", "5-10%", "10-20%", ">20%")


DATA = DataConfig()


# ---------------------------------------------------------------------------
# WHAT YOUR EXTRACT SUPPORTS TODAY
# ---------------------------------------------------------------------------
# Present:  order_id, algo, market, symbol, slippage, spread, %ADV, vol,
#           participation, duration, notional, quantity
#
# Works fully -- ALL SIX cost-model features are live:
#   Tier 1  all three rules
#   Tier 2  bands on algo x market x %ADV bucket, both metrics
#   Tier 3  cost model on sqrt(%ADV), log(POV), log(duration), log(spread),
#           log(vol) and the size x urgency interaction, + algo & market
#           dummies; true per-order horizon in sigma_expected; cross-fitting;
#           calibration; z-scores; review queue; slice t-tests on algo / market /
#           %ADV bucket and their crosses
#
# Degraded, and by how much:
#   no broker        -> no broker slice test. On the demo book that test is what
#                       finds the single largest systematic effect.
#   no side          -> no side dummy (minor)
#
# NOT AVAILABLE:
#   Cause attribution needs at least one of reversion_bps / passive_fill_pct /
#   auction_pct / momentum_bps. With none of them Tier 3 still gives you the
#   threshold, the z-scores and the slice tests, but every flag comes back
#   "unexplained" -- it can tell you WHICH orders to look at, not WHY.
#   If you can get exactly one column added, ask for post-trade reversion.
