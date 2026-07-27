"""Apply a fixed limit to every order. No fitting, no reference distribution.

This tier exists as the honest baseline. It is what most best-execution policies
still say in writing, it is trivially auditable, and it is statistically wrong in
a specific, measurable way: because the limit does not scale with difficulty, it
flags large/illiquid/volatile orders almost automatically and small/liquid ones
almost never. `run.py` prints the flag rate by %ADV bucket so you can see the
gradient rather than take it on faith.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema

IN_RANGE = "IN_RANGE"
OUT_LOW = "OUT_LOW"      # worse than the limit -> underperformance
OUT_HIGH = "OUT_HIGH"    # better than the limit -> suspiciously good
FLAGGED = {OUT_LOW, OUT_HIGH}

# rule name -> (column holding the test statistic, config attribute for the limit)
RULES = {
    "abs_bps":         (schema.SLIPPAGE_BPS,    "max_abs_bps"),
    "spread_multiple": (schema.PERF_IN_SPREADS, "max_spread_multiple"),
    "sigma_multiple":  (schema.PERF_NORM,       "max_sigma_multiple"),
}


def describe(cfg) -> str:
    """One-line statement of the active rule, for the report header."""
    col, limit_attr = RULES[cfg.rule]
    return f"|{col}| > {getattr(cfg, limit_attr)}  (rule={cfg.rule})"


def score(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tag every order against the fixed limit.

    Adds: rule_stat, rule_limit, zone, flagged, material, review_required.
    """
    if cfg.rule not in RULES:
        raise ValueError(f"Unknown Tier 1 rule {cfg.rule!r}; pick one of {list(RULES)}")

    col, limit_attr = RULES[cfg.rule]
    if col not in df.columns:
        raise ValueError(f"Rule {cfg.rule!r} needs column {col!r}, which is absent.")

    limit = float(getattr(cfg, limit_attr))
    stat = df[col].astype(float)

    out = df.copy()
    out["rule_stat"] = stat
    out["rule_limit"] = limit
    out["zone"] = np.select(
        [stat < -limit, stat > limit],
        [OUT_LOW, OUT_HIGH],
        default=IN_RANGE,
    )
    out["flagged"] = out["zone"].isin(FLAGGED)

    # Tier-comparable severity ranking: 1.0 == exactly at the limit. Lets the
    # top-level comparison hold every tier to the SAME review budget.
    out["rank_stat"] = stat.abs() / limit

    # Materiality: a tiny order is not worth an analyst's afternoon.
    if schema.NOTIONAL in out.columns and cfg.min_notional_review > 0:
        out["material"] = out[schema.NOTIONAL].fillna(0) >= cfg.min_notional_review
    else:
        out["material"] = True
    out["review_required"] = out["flagged"] & out["material"]
    return out


def flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """The diagnosis of this tier: flag rate as a function of order difficulty.

    A well-calibrated threshold produces a roughly FLAT column here. Tier 1 does
    not, and that is the whole argument for the tiers above it.
    """
    g = scored.groupby(schema.ADV_BUCKET, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
    })
    return out.round(2)
