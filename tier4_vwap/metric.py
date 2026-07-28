"""The two things Tier 4 changes: what is measured, and what it is divided by.

1. DE-BIAS THE BENCHMARK
------------------------
Against interval VWAP you are part of your own benchmark. Let f be your share of
interval volume (the `PR` column). Splitting the benchmark into your prints and
everyone else's:

    VWAP_total = (1-f) * VWAP_others  +  f * P_you

Substituting and rearranging gives an exact identity, no model involved:

    P_you - VWAP_others  =  (P_you - VWAP_total) / (1 - f)

So the reported slippage UNDERSTATES performance against the rest of the market
by exactly 1/(1-f). At 30% participation, a reported -20 bps is really -29 bps.
The bigger the order, the more the benchmark is its own prints -- which is why
large VWAP orders can look acceptable against interval VWAP while being nothing
of the sort. They are grading their own homework.

This reframes participation entirely. For a VWAP algo it is not an urgency
choice and not a cost driver -- both of which the Tier 3 impact model assumed.
It is the dilution factor, and it belongs in the metric rather than the model.

2. SCALE BY TRACKING ERROR, NOT IMPACT
--------------------------------------
The slippage of a VWAP order against interval VWAP is an identity:

    slippage  =  - SUM_t (w_t - v_t)(P_t - VWAP)

with w_t your share of your own order in bucket t and v_t the market's share of
interval volume. Match the curve (w_t = v_t) and slippage is EXACTLY zero, at
any size. The error is a covariance between schedule deviation and the intraday
price path -- so its natural scale is

    sigma_track  =  d * sigma_intraday * sqrt(T/S)

where d is the (unobserved) normalized schedule deviation, absorbed into a
constant. Note what is NOT here: sqrt(%ADV). The square-root impact law is the
right shape for an arrival-price benchmark and the wrong one for this.

The spread term is kept but demoted -- a VWAP algo still pays spread on each
child order -- so the tracking term dominates for anything worked over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema


def apply(df: pd.DataFrame, cfg, data_cfg) -> tuple[pd.DataFrame, list[str]]:
    """Rewrite slippage, sigma and perf_norm in VWAP-native terms.

    Overwrites SLIPPAGE_BPS / SIGMA_EXPECTED_BPS / PERF_NORM in place so every
    downstream stage -- cost model, scoring, diagnostics, persistence -- works
    unchanged and reports in "vs the rest of the market" terms. The untouched
    original is kept in REPORTED_SLIPPAGE_BPS.

    Returns (frame, notes) where notes are things worth printing.
    """
    out = df.copy()
    notes = []

    out[schema.REPORTED_SLIPPAGE_BPS] = out[schema.SLIPPAGE_BPS]

    # --- 1) de-bias -------------------------------------------------------
    if cfg.debias_benchmark and schema.PARTICIPATION in out.columns:
        f_raw = pd.to_numeric(out[schema.PARTICIPATION], errors="coerce")
        # Guard the division. Beyond the cap the correction explodes (f=0.9
        # multiplies by 10) and the participation figure itself is usually
        # unreliable up there, so cap and mark rather than trust it.
        f = f_raw.clip(lower=0.0, upper=cfg.max_dilution_participation)
        capped = (f_raw > cfg.max_dilution_participation).fillna(False)
        f = f.fillna(0.0)

        out[schema.DILUTION_FACTOR] = 1.0 / (1.0 - f)
        out[schema.DILUTION_CAPPED] = capped
        out[schema.SLIPPAGE_BPS] = out[schema.REPORTED_SLIPPAGE_BPS] * out[schema.DILUTION_FACTOR]

        n_cap = int(capped.sum())
        median_f = float(f_raw.median())
        notes.append(
            f"de-biased by 1/(1-PR): median participation {median_f:.1%} "
            f"-> median correction x{1/(1-median_f):.3f}")
        if n_cap:
            notes.append(
                f"{n_cap:,} orders had participation above the "
                f"{cfg.max_dilution_participation:.0%} cap and were limited to "
                f"x{1/(1-cfg.max_dilution_participation):.2f} (see dilution_capped)")
    else:
        out[schema.DILUTION_FACTOR] = 1.0
        out[schema.DILUTION_CAPPED] = False
        if cfg.debias_benchmark:
            notes.append(
                "NO participation column -- benchmark NOT de-biased. Tier 4 loses "
                "its main advantage over Tier 3 without it.")

    # --- 2) tracking-error scale -----------------------------------------
    spread_term = cfg.k_spread * out[schema.SPREAD_BPS]

    have_vol = schema.VOLATILITY in out.columns
    if have_vol:
        if schema.DURATION_MIN in out.columns:
            dur = out[schema.DURATION_MIN]
            if data_cfg.default_duration_min:
                dur = dur.fillna(data_cfg.default_duration_min)
        elif data_cfg.default_duration_min:
            dur = pd.Series(float(data_cfg.default_duration_min), index=out.index)
        else:
            dur = None
    else:
        dur = None

    if dur is not None:
        horizon = (dur / data_cfg.minutes_per_day).clip(lower=0.0)
        track_term = cfg.k_track * out[schema.VOLATILITY] * np.sqrt(horizon)
        usable = track_term.notna() & (track_term > 0)
        track_term = track_term.where(usable, 0.0)
    else:
        track_term = pd.Series(0.0, index=out.index)
        usable = pd.Series(False, index=out.index)
        notes.append("NO volatility/duration -- falling back to a spread-only scale.")

    sigma = np.sqrt(spread_term.astype(float) ** 2 + track_term.astype(float) ** 2)
    out[schema.SIGMA_EXPECTED_BPS] = sigma.clip(lower=data_cfg.min_sigma_bps)
    out[schema.NORM_BASIS] = np.where(usable, "spread+track", "spread_only")
    out[schema.PERF_NORM] = out[schema.SLIPPAGE_BPS] / out[schema.SIGMA_EXPECTED_BPS]
    return out, notes


def dilution_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """How much the de-biasing moved things, by size bucket.

    The gradient is the argument: correction grows with order size, so the
    orders Tier 3 was most likely to wave through are the ones most flattered
    by the contaminated benchmark.
    """
    if schema.DILUTION_FACTOR not in scored.columns:
        return pd.DataFrame()

    g = scored.groupby(schema.ADV_BUCKET, dropna=False)
    return pd.DataFrame({
        "n": g.size(),
        "median_participation": g[schema.PARTICIPATION].median().round(4),
        "median_correction": g[schema.DILUTION_FACTOR].median().round(3),
        "reported_bps": g[schema.REPORTED_SLIPPAGE_BPS].median().round(2),
        "debiased_bps": g[schema.SLIPPAGE_BPS].median().round(2),
        "understated_by_bps": (g[schema.SLIPPAGE_BPS].median()
                               - g[schema.REPORTED_SLIPPAGE_BPS].median()).round(2),
    })
