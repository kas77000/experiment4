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
    out["residual_bps"] = out[schema.SLIPPAGE_BPS] - out["expected_bps"]

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
