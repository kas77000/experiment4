"""Fit a Gaussian band per group, and score orders against it.

Two estimators are computed for every group and both are reported:

    classical:  centre = mean(x)     scale = std(x, ddof=1)
    robust:     centre = median(x)   scale = 1.4826 * MAD(x)

The 1.4826 is 1 / Phi^-1(0.75), which makes the scaled MAD a consistent
estimator of sigma *under normality*. That is the point of showing both: on
genuinely normal data they agree, so any gap between them IS the
non-normality, expressed in the band's own units. On a fat-tailed book the
classical scale is inflated by the very outliers the band exists to catch,
and the two rows say so numerically.

The band is symmetric around the centre by construction. Slippage is not
symmetric, so the two tails are not equally well served. That is a property
of the method, not a bug; normality.required_k() quantifies it per tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema
from tier5 import config as t5cfg

# 1 / Phi^-1(0.75). Makes MAD a consistent estimator of sigma under normality.
MAD_TO_SIGMA = 1.4826

# Zone labels, shared vocabulary with the other tiers.
IN_RANGE = "IN_RANGE"    # inside the band -> acceptable
OUT_LOW = "OUT_LOW"      # below the band: underperformance -> flag / justify
OUT_HIGH = "OUT_HIGH"    # above the band: suspiciously good -> flag / justify
NO_BAND = "NO_BAND"      # no trusted group, or the metric is missing
FLAGGED = {OUT_LOW, OUT_HIGH}

# Column order of the band table.
BAND_COLS = [
    "level", schema.ALGO, schema.ADV_BUCKET, "n",
    "centre_classical", "scale_classical", "lo_classical", "hi_classical",
    "centre_robust", "scale_robust", "lo_robust", "hi_robust",
]

_EMPTY = {
    "n": 0,
    "centre_classical": np.nan, "scale_classical": np.nan,
    "lo_classical": np.nan, "hi_classical": np.nan,
    "centre_robust": np.nan, "scale_robust": np.nan,
    "lo_robust": np.nan, "hi_robust": np.nan,
}


def estimates(x, k: float) -> dict:
    """Both estimators and both bands for one group's metric array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return dict(_EMPTY)

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * MAD_TO_SIGMA

    return {
        "n": n,
        "centre_classical": mean, "scale_classical": sd,
        "lo_classical": mean - k * sd, "hi_classical": mean + k * sd,
        "centre_robust": med, "scale_robust": mad,
        "lo_robust": med - k * mad, "hi_robust": med + k * mad,
    }


def fit(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return a tidy band table with one row per (level, group).

    Every group appears, including groups too thin to trust -- those are
    marked trusted=False rather than dropped, so a thin cell is visible.
    """
    metric = cfg.metric
    if metric not in df.columns:
        raise ValueError(f"Tier 5 needs column {metric!r}, which is absent.")

    rows = []
    for level in cfg.group_levels:
        keys = t5cfg.LEVEL_KEYS[level]
        if not keys:
            row = {"level": level, schema.ALGO: None, schema.ADV_BUCKET: None}
            row.update(estimates(df[metric].to_numpy(), cfg.k_sigma))
            rows.append(row)
            continue
        for gk, g in df.groupby(list(keys), dropna=False, observed=False):
            gk = gk if isinstance(gk, tuple) else (gk,)
            row = {"level": level, schema.ALGO: None, schema.ADV_BUCKET: None}
            row.update(dict(zip(keys, gk)))
            row.update(estimates(g[metric].to_numpy(), cfg.k_sigma))
            rows.append(row)

    out = pd.DataFrame(rows)[BAND_COLS]
    out["trusted"] = out["n"] >= cfg.min_group_n
    return out.reset_index(drop=True)


def classify(x: float, lo: float, hi: float) -> str:
    """Map a metric value to a zone given one band's bounds."""
    if not np.isfinite(x):
        return NO_BAND
    if x < lo:
        return OUT_LOW
    if x > hi:
        return OUT_HIGH
    return IN_RANGE


class BandModel:
    """Holds the fitted table and scores orders against one chosen estimator."""

    def __init__(self, table: pd.DataFrame, cfg):
        if cfg.estimator not in t5cfg.ESTIMATORS:
            raise ValueError(f"Unknown estimator {cfg.estimator!r}; "
                             f"pick one of {list(t5cfg.ESTIMATORS)}")
        if cfg.score_level not in t5cfg.LEVEL_KEYS:
            raise ValueError(f"Unknown score_level {cfg.score_level!r}; "
                             f"pick one of {list(t5cfg.LEVEL_KEYS)}")

        self.table = table
        self.cfg = cfg
        est = cfg.estimator
        self.centre_col = f"centre_{est}"
        self.scale_col = f"scale_{est}"
        self.lo_col = f"lo_{est}"
        self.hi_col = f"hi_{est}"

        trusted = table[table["trusted"]]
        self._by_level = {}
        for level in t5cfg.LEVEL_KEYS:
            keys = t5cfg.LEVEL_KEYS[level]
            sub = trusted[trusted["level"] == level]
            self._by_level[level] = {
                tuple(r[k] for k in keys): r for _, r in sub.iterrows()
            }

    def _lookup(self, algo, adv_bucket):
        """The chosen level if it is trusted, else the global band."""
        vals = {schema.ALGO: algo, schema.ADV_BUCKET: adv_bucket}
        for level in (self.cfg.score_level, t5cfg.LEVEL_ALL):
            keys = t5cfg.LEVEL_KEYS[level]
            key = tuple(vals[k] for k in keys)
            band = self._by_level.get(level, {}).get(key)
            if band is not None:
                return band, level
        return None, None

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a prepared frame. Adds the band, the zone and rank_stat."""
        metric = self.cfg.metric
        k = self.cfg.k_sigma
        recs = []
        for _, r in df.iterrows():
            band, level = self._lookup(r.get(schema.ALGO),
                                       r.get(schema.ADV_BUCKET))
            x = float(r[metric]) if pd.notna(r[metric]) else np.nan
            if band is None or not np.isfinite(x):
                recs.append((NO_BAND, None, np.nan, np.nan,
                             np.nan, np.nan, False, np.nan))
                continue
            centre = float(band[self.centre_col])
            scale = float(band[self.scale_col])
            lo = float(band[self.lo_col])
            hi = float(band[self.hi_col])
            zone = classify(x, lo, hi)
            # 1.0 == exactly at the limit. Same convention the other tiers use,
            # which is what lets run.py hold every method to one review budget.
            denom = k * scale
            rank = abs(x - centre) / denom if denom > 0 else np.nan
            recs.append((zone, level, centre, scale, lo, hi,
                         zone in FLAGGED, rank))

        out = df.copy()
        cols = ["zone", "band_level", "band_centre", "band_scale",
                "band_lo", "band_hi", "flagged", "rank_stat"]
        if not recs:
            for c in cols:
                out[c] = pd.Series(dtype="float64")
        else:
            for c, vals in zip(cols, zip(*recs)):
                out[c] = list(vals)

        if schema.NOTIONAL in out.columns and self.cfg.min_notional_review > 0:
            out["material"] = (out[schema.NOTIONAL].fillna(0)
                               >= self.cfg.min_notional_review)
        else:
            out["material"] = True
        out["review_required"] = out["flagged"] & out["material"]
        return out


def flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """Flag rate vs order difficulty.

    A global Gaussian band is one number for the whole book, so this column
    is expected to slope the way Tier 1's did: it does not adjust for
    difficulty. Scoring at a finer level flattens it.
    """
    g = scored.groupby(schema.ADV_BUCKET, dropna=False, observed=False)
    return pd.DataFrame({
        "n": g.size(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
    }).round(2)
