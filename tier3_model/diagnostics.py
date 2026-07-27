"""Attribute a flagged order to a CAUSE, not just a number.

A z-score says an order was bad. It does not say what to change. This module
turns the optional diagnostic columns into a ranked cause, using evidence that
is measured in PERCENTILE units so the different pieces are comparable:

    over_aggressive   post-trade reversion is high for the order's own scale ->
                      you moved the price and it came back. Trade slower.
    spread_bleed      passive fill share far below the algo's norm, with NO
                      reversion -> you paid the spread rather than moved the
                      price. Different fix: post more, cross less.
    missed_close      auction participation near zero where peers do meaningful
                      volume -> in HK the close is a large share of the day.
    adverse_momentum  the market drifted hard over the interval. This is a
                      timing/signal problem, not an algo problem. Weighted
                      lowest for interval-VWAP benchmarks, which are largely
                      immune to drift; raise it for ARRIVAL/IS.
    unexplained       no fingerprint. Check the marks, the benchmark window and
                      the fill records before blaming execution.

The reversion-vs-no-reversion split is the important one and it is why post-trade
marks are worth chasing: without them "we caused impact" and "we were adversely
selected" look identical and have opposite remedies.

Rules are triggered at percentiles OF YOUR OWN BOOK, so nothing needs retuning
when you point this at a different market or year.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tca import schema
from tier3_model import features, scoring

OVER_AGGRESSIVE = "over_aggressive"
SPREAD_BLEED = "spread_bleed"
# Emitted INSTEAD of SPREAD_BLEED when no reversion column is available. Low
# passive fill on its own cannot distinguish "we crossed the spread all day"
# from "we pushed so hard the price ran away from us" -- both leave the same
# footprint, and they have opposite remedies. Claiming spread_bleed here would
# prescribe "post more" for orders whose real fix is "trade slower".
LOW_PASSIVE_UNVERIFIED = "low_passive_unverified"
MISSED_CLOSE = "missed_close"
ADVERSE_MOMENTUM = "adverse_momentum"
UNEXPLAINED = "unexplained"
TOO_GOOD = "suspiciously_good"
NOT_FLAGGED = "-"

# How much to trust each rule when two fire at once. Interval VWAP is largely
# immune to market drift, so momentum is discounted; raise it for arrival/IS.
RULE_WEIGHT = {
    OVER_AGGRESSIVE: 1.0,
    SPREAD_BLEED: 1.0,
    MISSED_CLOSE: 1.0,
    ADVERSE_MOMENTUM: 0.6,
}

REMEDY = {
    OVER_AGGRESSIVE: "Cap participation / lengthen the horizon on this profile.",
    SPREAD_BLEED: "Increase passive posting; review the algo's crossing logic.",
    LOW_PASSIVE_UNVERIFIED: (
        "Passive fill far below this algo's norm -- either it crossed the spread "
        "(post more) or it pushed too hard and the price ran (trade slower). "
        "Add reversion_bps to tell these apart."),
    MISSED_CLOSE: "Route a closing-auction slice; check the auction cut-off.",
    ADVERSE_MOMENTUM: "Execution was reasonable; revisit timing of the decision.",
    UNEXPLAINED: "Verify marks, benchmark window and fill timestamps first.",
    TOO_GOOD: "Verify the benchmark and the marks -- gains this large are usually data.",
}


@dataclass
class CauseModel:
    """Percentile ranks and trigger points learned from the book."""
    available: list = field(default_factory=list)
    ranks: pd.DataFrame = field(default_factory=pd.DataFrame)
    triggers: dict = field(default_factory=dict)


def fit_causes(scored: pd.DataFrame, cfg) -> CauseModel:
    """Compute per-order percentile ranks for every diagnostic signal present."""
    available = [c for c in schema.DIAGNOSTIC if c in scored.columns]
    ranks = pd.DataFrame(index=scored.index)
    triggers = {}

    sig = scored[schema.SIGMA_EXPECTED_BPS]

    if schema.REVERSION_BPS in available:
        rev_norm = scored[schema.REVERSION_BPS] / sig
        ranks["rev"] = rev_norm.rank(pct=True)
        ranks["rev_norm"] = rev_norm
        triggers[OVER_AGGRESSIVE] = cfg.rev_hi_pct / 100.0

    if schema.PASSIVE_FILL_PCT in available:
        # Ranked WITHIN algo: a passive algo at 40% passive is an outlier,
        # an aggressive algo at 40% is normal.
        ranks["passive"] = scored.groupby(schema.ALGO)[schema.PASSIVE_FILL_PCT] \
                                 .rank(pct=True)
        triggers[SPREAD_BLEED] = cfg.passive_lo_pct / 100.0

    if schema.AUCTION_PCT in available:
        ranks["auction"] = scored.groupby(schema.MARKET)[schema.AUCTION_PCT] \
                                 .rank(pct=True)
        triggers[MISSED_CLOSE] = cfg.auction_lo_pct / 100.0

    if schema.MOMENTUM_BPS in available:
        mom_norm = (scored[schema.MOMENTUM_BPS] / sig).abs()
        ranks["momentum"] = mom_norm.rank(pct=True)
        triggers[ADVERSE_MOMENTUM] = cfg.momentum_hi_pct / 100.0

    if schema.PARTICIPATION in scored.columns:
        ranks["pov"] = scored.groupby(schema.ALGO)[schema.PARTICIPATION].rank(pct=True)

    return CauseModel(available=available, ranks=ranks, triggers=triggers)


def _excess(cause_model: CauseModel, scored: pd.DataFrame) -> pd.DataFrame:
    """How far past its trigger each rule fired, in percentile units (0 = not fired).

    Percentile units make the rules directly comparable, so picking the largest
    excess is a defensible way to choose between two competing explanations.
    """
    r = cause_model.ranks
    t = cause_model.triggers
    ex = pd.DataFrame(0.0, index=scored.index,
                      columns=list(RULE_WEIGHT.keys()))

    if OVER_AGGRESSIVE in t and "rev" in r:
        raw = (r["rev"] - t[OVER_AGGRESSIVE]).clip(lower=0)
        # High participation corroborates: you pushed, the price moved.
        if "pov" in r:
            raw = raw * (1.0 + 0.5 * (r["pov"] > 0.5).astype(float))
        ex[OVER_AGGRESSIVE] = raw

    if SPREAD_BLEED in t and "passive" in r:
        raw = (t[SPREAD_BLEED] - r["passive"]).clip(lower=0)
        # Only spread bleed if the price did NOT revert; otherwise it is impact.
        if "rev" in r:
            raw = raw * (r["rev"] < t.get(OVER_AGGRESSIVE, 1.0)).astype(float)
        ex[SPREAD_BLEED] = raw

    if MISSED_CLOSE in t and "auction" in r:
        ex[MISSED_CLOSE] = (t[MISSED_CLOSE] - r["auction"]).clip(lower=0)

    if ADVERSE_MOMENTUM in t and "momentum" in r:
        ex[ADVERSE_MOMENTUM] = (r["momentum"] - t[ADVERSE_MOMENTUM]).clip(lower=0)

    for c in ex.columns:
        ex[c] = ex[c] * RULE_WEIGHT[c]
    return ex


def attribute(scored: pd.DataFrame, cause_model: CauseModel) -> pd.DataFrame:
    """Add `likely_cause`, `cause_strength` and `remedy` to a scored frame."""
    out = scored.copy()
    ex = _excess(cause_model, scored)

    best = ex.idxmax(axis=1)
    strength = ex.max(axis=1)
    cause = best.where(strength > 0, UNEXPLAINED)

    # Only flagged orders get a cause; suspicious GAINS get their own label,
    # because "we beat the benchmark by 6 sigma" is a data question, not an
    # execution one.
    cause = cause.where(out["flagged"], NOT_FLAGGED)
    cause = cause.mask(out["zone"] == scoring.OUT_HIGH, TOO_GOOD)

    # Without post-trade marks, low passive fill is ambiguous between spread
    # cost and own impact. Downgrade the label rather than emit a confident
    # diagnosis the data cannot support -- a wrong remedy is worse than an
    # honest "one of these two".
    if "rev" not in cause_model.ranks.columns:
        cause = cause.replace(SPREAD_BLEED, LOW_PASSIVE_UNVERIFIED)

    out["likely_cause"] = cause
    out["cause_strength"] = np.where(out["flagged"], strength.round(3), np.nan)
    out["remedy"] = out["likely_cause"].map(REMEDY).fillna("")
    return out


def cause_summary(attributed: pd.DataFrame) -> pd.DataFrame:
    """Distribution of causes across the review queue -- the management view."""
    flagged = attributed[attributed["flagged"]]
    if not len(flagged):
        return pd.DataFrame()
    g = flagged.groupby("likely_cause")
    return pd.DataFrame({
        "orders": g.size(),
        "pct_of_flags": (100.0 * g.size() / len(flagged)).round(1),
        "mean_z": g["z"].mean().round(2),
        "mean_shortfall_bps": g["residual_bps"].mean().round(1),
        "total_shortfall_bps_x_notional_m": (
            (flagged["residual_bps"] * flagged.get(
                schema.NOTIONAL, pd.Series(0, index=flagged.index)) / 1e6)
            .groupby(flagged["likely_cause"]).sum().round(0)),
    }).sort_values("orders", ascending=False)


def cause_confusion(attributed: pd.DataFrame) -> pd.DataFrame:
    """Attributed cause vs known cause. Synthetic demo only."""
    if schema.TRUE_CAUSE not in attributed.columns:
        return pd.DataFrame()
    flagged = attributed[attributed["flagged"] & attributed[schema.TRUE_OUTLIER]]
    if not len(flagged):
        return pd.DataFrame()
    return pd.crosstab(flagged[schema.TRUE_CAUSE], flagged["likely_cause"])


# What counts as a correct attribution, for scoring the demo. `benchmark_error`
# is injected WITHOUT any execution fingerprint, so "unexplained" / "check the
# marks" is the right answer for it -- not a miss.
# LOW_PASSIVE_UNVERIFIED counts as correct for BOTH over_aggressive and
# spread_bleed: with no reversion column it is the most specific true statement
# available, and it names both remedies rather than guessing one.
CORRECT_ATTRIBUTION = {
    "over_aggressive": {OVER_AGGRESSIVE, LOW_PASSIVE_UNVERIFIED},
    "spread_bleed": {SPREAD_BLEED, LOW_PASSIVE_UNVERIFIED},
    "missed_close": {MISSED_CLOSE},
    "benchmark_error": {UNEXPLAINED, TOO_GOOD},
}


def cause_accuracy(attributed: pd.DataFrame) -> pd.DataFrame:
    """Per-cause attribution accuracy on flagged true failures. Demo only."""
    if schema.TRUE_CAUSE not in attributed.columns:
        return pd.DataFrame()
    flagged = attributed[attributed["flagged"] & attributed[schema.TRUE_OUTLIER]]
    if not len(flagged):
        return pd.DataFrame()

    rows = []
    for truth, accepted in CORRECT_ATTRIBUTION.items():
        sub = flagged[flagged[schema.TRUE_CAUSE] == truth]
        if not len(sub):
            continue
        hit = sub["likely_cause"].isin(accepted)
        rows.append({
            "true_cause": truth,
            "flagged": len(sub),
            "attributed_correctly": int(hit.sum()),
            "accuracy_pct": round(100.0 * hit.mean(), 1),
            "accepted_labels": "|".join(sorted(accepted)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# single-order narrative
# --------------------------------------------------------------------------

def cost_drivers(row: pd.Series, model_fit, cfg, top: int = 4) -> pd.DataFrame:
    """Which features made this order expensive to begin with.

    Contribution_j = beta_med_j * x_std_j, in perf_norm units, converted to bps.
    Because features are standardized, these are directly comparable and they
    sum (with the intercept) to the expected cost.
    """
    if model_fit.backend != "quantreg":
        return pd.DataFrame()

    spec = model_fit.spec
    X = features.design(row.to_frame().T, spec, cfg)[0]
    beta = model_fit.coefs[model_fit.taus[1]]
    sig = float(row[schema.SIGMA_EXPECTED_BPS])

    contrib = pd.DataFrame({
        "feature": spec.names,
        "value_std": np.round(X, 3),
        "contribution_bps": np.round(beta * X * sig, 1),
    })
    contrib = contrib[contrib["feature"] != "intercept"]
    contrib = contrib.reindex(
        contrib["contribution_bps"].abs().sort_values(ascending=False).index)
    return contrib.head(top).reset_index(drop=True)


def explain_order(row: pd.Series, model_fit, cause_model: CauseModel, cfg) -> str:
    """A paragraph a human can act on, for one flagged order."""
    lines = []
    oid = row.get(schema.ORDER_ID, "?")
    lines.append(f"Order {oid}  |  {row.get(schema.ALGO)} / {row.get(schema.BROKER, 'n/a')}"
                 f"  |  {row.get(schema.SYMBOL, '')}  {row.get(schema.SIDE, '')}")
    lines.append(f"  size {row.get(schema.PCT_ADV, float('nan')):.2f}% ADV"
                 f"   POV {row.get(schema.PARTICIPATION, float('nan')):.1%}"
                 f"   {row.get(schema.DURATION_MIN, float('nan')):.0f} min"
                 f"   spread {row.get(schema.SPREAD_BPS, float('nan')):.1f}bps"
                 f"   vol {row.get(schema.VOLATILITY, float('nan')):.0f}bps/day")
    lines.append(f"  actual {row[schema.SLIPPAGE_BPS]:+.1f} bps"
                 f"   vs expected {row['expected_bps']:+.1f} bps"
                 f"   (band {row['band_lo_bps']:+.1f} .. {row['band_hi_bps']:+.1f})")
    lines.append(f"  shortfall {row['residual_bps']:+.1f} bps"
                 f"   z = {row['z']:+.2f}"
                 f"   -> {row['zone']} / {row['severity']}")

    drivers = cost_drivers(row, model_fit, cfg)
    if len(drivers):
        bits = ", ".join(f"{r.feature} {r.contribution_bps:+.1f}bps"
                         for r in drivers.itertuples())
        lines.append(f"  expected cost driven by: {bits}")

    r = cause_model.ranks
    ev = []
    if "rev" in r:
        ev.append(f"reversion {r.at[row.name, 'rev_norm']:.2f}x sigma "
                  f"(p{100*r.at[row.name, 'rev']:.0f})")
    if "passive" in r and schema.PASSIVE_FILL_PCT in row:
        ev.append(f"passive fill {row[schema.PASSIVE_FILL_PCT]:.0%} "
                  f"(p{100*r.at[row.name, 'passive']:.0f} in algo)")
    if "auction" in r and schema.AUCTION_PCT in row:
        ev.append(f"auction {row[schema.AUCTION_PCT]:.1%} "
                  f"(p{100*r.at[row.name, 'auction']:.0f} in market)")
    if "momentum" in r and schema.MOMENTUM_BPS in row:
        ev.append(f"interval drift {row[schema.MOMENTUM_BPS]:+.0f}bps "
                  f"(p{100*r.at[row.name, 'momentum']:.0f})")
    if ev:
        lines.append("  evidence: " + "; ".join(ev))

    lines.append(f"  LIKELY CAUSE: {row.get('likely_cause', UNEXPLAINED)}")
    lines.append(f"  ACTION: {row.get('remedy', '')}")
    return "\n".join(lines)
