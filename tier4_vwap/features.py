"""The Tier 4 design matrix --- curve difficulty, not impact.

Same API as tier3_model.features, so the shared cost model runs against either.

What changed and why:

    DROPPED  log_pov, sqrt_adv x log_pov
             Participation is an OUTPUT of the volume curve for a VWAP algo,
             not an urgency decision. Modelling it as a cost driver imports a
             POV / implementation-shortfall framing that does not apply -- and
             it is now used in the metric instead, as the dilution factor.

    ADDED    session_coverage = duration / session length
             How much of the day the order spanned. An order worked across the
             whole session tracks the volume curve almost by construction; one
             squeezed into twenty minutes carries far more idiosyncratic
             curve risk, because a single misjudged bucket is a large share of
             the schedule. This is the observable proxy for tracking error.

    DEMOTED  sqrt_adv
             Kept, but with no interaction and no POV partner. If the VWAP
             argument holds, its coefficient should collapse once the benchmark
             is de-biased. That is a prediction the output lets you check.

    KEPT     log_dur, log_spread, log_vol, algo / market / side

What is deliberately NOT a feature: %POST and the auction share. Those are
execution CHOICES, not order difficulty. Putting them in the expectation would
let an algo that crosses the spread all day lower its own bar and stop flagging
-- the same trap as absorbing the algo effect. They stay on the diagnostic side,
where they explain a flag instead of excusing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema
from tier3_model.features import FeatureSpec, coverage  # noqa: F401  (shared)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _raw_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Session coverage is computed against the same session length the metric
    # uses, so the two stay consistent. Read from the shared data config.
    import config as root_config
    session = root_config.DATA.minutes_per_day

    if schema.DURATION_MIN in df.columns:
        dur = _num(df, schema.DURATION_MIN)
        if root_config.DATA.default_duration_min:
            dur = dur.fillna(root_config.DATA.default_duration_min)
        out["session_coverage"] = (dur / session).clip(lower=1e-3, upper=1.5)
        out["log_dur"] = np.log(dur.clip(lower=1e-2))

    if cfg.include_size and schema.PCT_ADV in df.columns:
        out["sqrt_adv"] = np.sqrt(_num(df, schema.PCT_ADV).clip(lower=0) / 100.0)
    if schema.SPREAD_BPS in df.columns:
        out["log_spread"] = np.log(_num(df, schema.SPREAD_BPS).clip(lower=1e-2))
    if schema.VOLATILITY in df.columns:
        out["log_vol"] = np.log(_num(df, schema.VOLATILITY).clip(lower=1e-2))

    return out.replace([np.inf, -np.inf], np.nan)


def fit_spec(df: pd.DataFrame, cfg) -> FeatureSpec:
    raw = _raw_features(df, cfg)
    keep = [c for c in raw.columns if raw[c].notna().any()]

    spec = FeatureSpec(numeric=keep)
    for c in keep:
        med = float(raw[c].median())
        col = raw[c].fillna(med)
        sd = float(col.std(ddof=0))
        spec.medians[c] = med
        spec.means[c] = float(col.mean())
        spec.stds[c] = sd if sd > 1e-9 else 1.0

    if cfg.algo_effect == "absorb" and schema.ALGO in df.columns:
        spec.algo_levels = sorted(df[schema.ALGO].dropna().astype(str).unique())[1:]
    if schema.MARKET in df.columns:
        spec.market_levels = sorted(df[schema.MARKET].dropna().astype(str).unique())[1:]
    spec.has_side = schema.SIDE in df.columns
    return spec


def design(df: pd.DataFrame, spec: FeatureSpec, cfg) -> np.ndarray:
    raw = _raw_features(df, cfg)
    n = len(df)
    cols = [np.ones(n)]

    for c in spec.numeric:
        col = raw[c] if c in raw.columns else pd.Series(np.nan, index=df.index)
        col = col.fillna(spec.medians[c]).to_numpy(dtype=float)
        cols.append((col - spec.means[c]) / spec.stds[c])

    for a in spec.algo_levels:
        cols.append((df[schema.ALGO].astype(str) == a).to_numpy(dtype=float))
    for m in spec.market_levels:
        cols.append((df[schema.MARKET].astype(str) == m).to_numpy(dtype=float))
    if spec.has_side:
        side = df[schema.SIDE].astype(str).str.lower()
        cols.append(side.isin(["buy", "b", "1"]).to_numpy(dtype=float))

    return np.column_stack(cols)
