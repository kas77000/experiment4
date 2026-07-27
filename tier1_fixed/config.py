"""Tier 1 knobs --- fixed limits, no fitting, no data required.

Pick ONE rule. Each is a real convention you will find in production best-ex
policies; they differ only in what they divide by.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier1Config:
    # Which fixed rule to apply:
    #   "abs_bps"         |slippage_bps|        > max_abs_bps
    #   "spread_multiple" |slippage/spread|     > max_spread_multiple
    #   "sigma_multiple"  |slippage/sigma_exp|  > max_sigma_multiple
    rule: str = "abs_bps"

    max_abs_bps: float = 50.0          # classic compliance limit
    max_spread_multiple: float = 3.0   # "more than 3 spreads through"
    max_sigma_multiple: float = 3.0    # best of the three: risk-adjusted, still fixed

    # Materiality gate: `flagged` is everything past the limit; an order also
    # needs notional >= this to become `review_required`. 0 = off, so
    # review_required == flagged. See the note in tier3_model/config.py on
    # matching this to the currency your notional is denominated in.
    min_notional_review: float = 0.0


CONFIG = Tier1Config()
