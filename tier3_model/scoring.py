"""Turn predicted quantiles into zones, z-scores and a review queue.

The z-score is the point of the whole tier:

    z = (perf_norm - q_med) / sigma_hat
    sigma_hat = (q_hi - q_lo) / (Phi^-1(tau_hi) - Phi^-1(tau_lo))

i.e. how far this order landed from what was expected of IT, measured in units
of the dispersion expected for IT. A -60bps fill on a 25% ADV name in a volatile
session and a -6bps fill on a liquid name traded over ten minutes can now
produce the same z, which is exactly the comparison a fixed or bucketed
threshold cannot make.

Severity is tiered because a flag costs analyst time:
    OK        inside the band
    MONITOR   outside the band but |z| < escalate_z -> logged and trended
    ESCALATE  |z| >= escalate_z -> written justification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from tca import schema

IN_RANGE = "IN_RANGE"
OUT_LOW = "OUT_LOW"
OUT_HIGH = "OUT_HIGH"
FLAGGED = {OUT_LOW, OUT_HIGH}

OK = "OK"
MONITOR = "MONITOR"
ESCALATE = "ESCALATE"

MIN_SIGMA_HAT = 0.05   # floor, so a degenerate band cannot manufacture huge z


def add_scores(df: pd.DataFrame, preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach predictions, residuals, z, zone, severity and the review gate."""
    out = df.copy()
    out["q_lo"] = preds["q_lo"]
    out["q_med"] = preds["q_med"]
    out["q_hi"] = preds["q_hi"]

    # Normal-equivalent scale from the fitted band width.
    span = norm.ppf(cfg.tau_hi) - norm.ppf(cfg.tau_lo)
    sigma_hat = ((out["q_hi"] - out["q_lo"]) / span).clip(lower=MIN_SIGMA_HAT)
    out["sigma_hat"] = sigma_hat

    perf = out[schema.PERF_NORM]
    out["z"] = (perf - out["q_med"]) / sigma_hat

    # Same quantities back in bps, which is what a human reads.
    sig = out[schema.SIGMA_EXPECTED_BPS]
    out["expected_bps"] = out["q_med"] * sig
    out["band_lo_bps"] = out["q_lo"] * sig
    out["band_hi_bps"] = out["q_hi"] * sig

    # And in SPREADS, which is how a trader reads it ("two spreads through").
    # Note the band is fitted in units of sigma_expected, not of spread -- that
    # is the whole vol-aware argument, and it is why the same band is a
    # different number of spreads on a fast order than on a slow one. These
    # columns are a presentation of the fitted band, not a second threshold.
    spr = out[schema.SPREAD_BPS].replace(0, np.nan)
    out["expected_spreads"] = out["expected_bps"] / spr
    out["band_lo_spreads"] = out["band_lo_bps"] / spr
    out["band_hi_spreads"] = out["band_hi_bps"] / spr
    out["residual_bps"] = out[schema.SLIPPAGE_BPS] - out["expected_bps"]

    # The shortfall in actual money, which is what gets a meeting's attention.
    # Negative = cash lost against the order's own expectation. A -3 sigma miss
    # on a small order and a -1.5 sigma miss on a large one can be ranked
    # against each other here in a way no z-score allows.
    if schema.NOTIONAL in out.columns:
        out["shortfall_ccy"] = out["residual_bps"] / 10_000.0 * out[schema.NOTIONAL]
    else:
        out["shortfall_ccy"] = np.nan

    out["zone"] = np.select(
        [perf < out["q_lo"], perf > out["q_hi"]],
        [OUT_LOW, OUT_HIGH],
        default=IN_RANGE,
    )
    out.loc[out["q_lo"].isna(), "zone"] = "NO_BAND"
    out["flagged"] = out["zone"].isin(FLAGGED)

    # Tier-comparable severity ranking (see tca/evaluate.precision_at_budget).
    out["rank_stat"] = out["z"].abs()

    out["severity"] = np.where(
        ~out["flagged"], OK,
        np.where(out["z"].abs() >= cfg.escalate_z, ESCALATE, MONITOR))

    if schema.NOTIONAL in out.columns and cfg.min_notional_review > 0:
        out["material"] = out[schema.NOTIONAL].fillna(0) >= cfg.min_notional_review
    else:
        out["material"] = True
    out["review_required"] = out["flagged"] & out["material"]
    return out


def _band_stats(g: pd.DataFrame) -> dict:
    """Summarize the per-order bands inside one slice."""
    return {
        "n": len(g),
        "median_spread_bps": g[schema.SPREAD_BPS].median(),
        "expected_bps": g["expected_bps"].median(),
        "band_lo_bps": g["band_lo_bps"].median(),
        "band_hi_bps": g["band_hi_bps"].median(),
        "band_lo_p10": g["band_lo_bps"].quantile(0.10),
        "band_lo_p90": g["band_lo_bps"].quantile(0.90),
        "band_lo_spreads": g["band_lo_spreads"].median(),
        "band_hi_spreads": g["band_hi_spreads"].median(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
    }


def threshold_table(scored: pd.DataFrame, levels=None) -> pd.DataFrame:
    """Make the per-order thresholds legible, at several levels of aggregation.

    Tier 3's threshold is not one number -- every order gets its own band, in
    bps, predicted from its own size, spread, volatility, duration and urgency.
    That is the point, but it leaves nothing to put on a slide. This summarizes
    the realized bands from the whole book down to algo x size bucket, so you
    get a headline figure and the detail underneath it.

    IMPORTANT: the aggregate rows DESCRIBE the fitted thresholds, they are not
    thresholds you should apply. Using the ALL row as the gate for every order
    would put you straight back at Tier 1, since it ignores the difficulty
    adjustment that the whole model exists to make. Scoring always uses each
    order's own band.

    The `band_lo_p10` / `band_lo_p90` columns are what make that concrete: they
    show how much the threshold MOVES inside a single row. Tier 2 would have one
    number there; the spread between them is the resolution Tier 3 buys.
    """
    if levels is None:
        levels = [[], [schema.ALGO], [schema.ADV_BUCKET],
                  [schema.ALGO, schema.ADV_BUCKET]]
        if schema.MARKET in scored.columns and scored[schema.MARKET].nunique() > 1:
            levels.insert(2, [schema.MARKET])

    key_cols = [schema.ALGO, schema.MARKET, schema.ADV_BUCKET]
    key_cols = [c for c in key_cols if c in scored.columns]

    rows = []
    for by in levels:
        by = [b for b in by if b in scored.columns]
        if not by:
            rows.append({"level": "ALL", **{c: None for c in key_cols},
                         **_band_stats(scored)})
            continue
        label = " x ".join(by)
        for keys, g in scored.groupby(by, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append({"level": label, **{c: None for c in key_cols},
                         **dict(zip(by, keys)), **_band_stats(g)})

    out = pd.DataFrame(rows)
    num = [c for c in out.columns if c not in ["level"] + key_cols]
    out[num] = out[num].astype(float).round(2)
    return out


def flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """The calibration proof: flag rate should now be FLAT across difficulty."""
    g = scored.groupby(schema.ADV_BUCKET, dropna=False)
    return pd.DataFrame({
        "n": g.size(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
        "mean_z": g["z"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
        "mean_expected_bps": g["expected_bps"].mean(),
    }).round(2)


def severity_summary(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("severity", dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "pct": (100.0 * g.size() / max(len(scored), 1)).round(2),
        "mean_z": g["z"].mean().round(2),
        "mean_residual_bps": g["residual_bps"].mean().round(1),
    })
    order = [c for c in [ESCALATE, MONITOR, OK] if c in out.index]
    return out.loc[order]
