"""Fit percentile bands per group, and score new orders against them.

Fitting is done at three nested levels so thin slices degrade gracefully:
    1. bucketed:  algo x market x adv_bucket   (most specific)
    2. pooled:    algo x market                (fallback)
    3. global:    all orders                   (last resort)
Scoring picks the most specific level that has >= min_group_n orders.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tca import schema

# Zone labels
GREEN = "GREEN"
GREY_LOW = "GREY_LOW"
GREY_HIGH = "GREY_HIGH"
RED_LOW = "RED_LOW"     # underperformance -> flag
RED_HIGH = "RED_HIGH"   # suspiciously good -> flag
FLAGGED = {RED_LOW, RED_HIGH}


def _band_row(perf: np.ndarray, cfg) -> dict:
    """Compute the five cut points + summary stats for one group's perf array."""
    lo_red, hi_red = cfg.red_percentiles
    lo_grey, hi_grey = cfg.grey_percentiles
    q = np.nanpercentile(perf, [lo_red, lo_grey, 50.0, hi_grey, hi_red])
    med = float(np.nanmedian(perf))
    mad = float(np.nanmedian(np.abs(perf - med)))  # robust dispersion
    return {
        "n": int(np.sum(~np.isnan(perf))),
        "q_red_lo": float(q[0]),
        "q_grey_lo": float(q[1]),
        "q_median": float(q[2]),
        "q_grey_hi": float(q[3]),
        "q_red_hi": float(q[4]),
        "mad_spreads": mad,
    }


def fit(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return a tidy threshold table with one row per (level, group)."""
    rows = []

    # Level 1: bucketed
    for keys, g in df.groupby(list(cfg.group_keys), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {"level": "bucketed"}
        row.update(dict(zip(cfg.group_keys, keys)))
        row.update(_band_row(g[schema.PERF_IN_SPREADS].to_numpy(), cfg))
        rows.append(row)

    # Level 2: pooled (algo x market)
    for (algo, market), g in df.groupby([schema.ALGO, schema.MARKET], dropna=False):
        row = {"level": "pooled", schema.ALGO: algo, schema.MARKET: market,
               schema.ADV_BUCKET: None}
        row.update(_band_row(g[schema.PERF_IN_SPREADS].to_numpy(), cfg))
        rows.append(row)

    # Level 3: global
    row = {"level": "global", schema.ALGO: None, schema.MARKET: None,
           schema.ADV_BUCKET: None}
    row.update(_band_row(df[schema.PERF_IN_SPREADS].to_numpy(), cfg))
    rows.append(row)

    cols = ["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET, "n",
            "q_red_lo", "q_grey_lo", "q_median", "q_grey_hi", "q_red_hi",
            "mad_spreads"]
    out = pd.DataFrame(rows)[cols]
    out["trusted"] = out["n"] >= cfg.min_group_n
    return out.sort_values(["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET]) \
              .reset_index(drop=True)


def classify(perf: float, band: pd.Series) -> str:
    """Map a normalized performance to a zone given one band row."""
    if perf < band["q_red_lo"]:
        return RED_LOW
    if perf > band["q_red_hi"]:
        return RED_HIGH
    if perf < band["q_grey_lo"]:
        return GREY_LOW
    if perf > band["q_grey_hi"]:
        return GREY_HIGH
    return GREEN


class ThresholdModel:
    """Holds the fitted table and scores individual orders with level fallback."""

    def __init__(self, table: pd.DataFrame, cfg):
        self.table = table
        self.cfg = cfg
        t = table[table["trusted"]]
        self._bucketed = {
            (r[schema.ALGO], r[schema.MARKET], r[schema.ADV_BUCKET]): r
            for _, r in t[t["level"] == "bucketed"].iterrows()
        }
        self._pooled = {
            (r[schema.ALGO], r[schema.MARKET]): r
            for _, r in t[t["level"] == "pooled"].iterrows()
        }
        gl = t[t["level"] == "global"]
        self._global = gl.iloc[0] if len(gl) else None

    def _lookup(self, algo, market, bucket):
        """Most-specific trusted band available, plus which level matched."""
        b = self._bucketed.get((algo, market, bucket))
        if b is not None:
            return b, "bucketed"
        p = self._pooled.get((algo, market))
        if p is not None:
            return p, "pooled"
        if self._global is not None:
            return self._global, "global"
        return None, None

    def score_order(self, algo, market, slippage_bps, spread_bps,
                    pct_adv=None) -> dict:
        """Score a single order. Returns zone, normalized perf, and matched band."""
        perf = slippage_bps / spread_bps
        bucket = self._bucket_for(pct_adv)
        band, level = self._lookup(algo, market, bucket)
        if band is None:
            return {"zone": "NO_BAND", "perf_in_spreads": perf,
                    "band_level": None, "flagged": False}
        zone = classify(perf, band)
        return {
            "zone": zone,
            "perf_in_spreads": perf,
            "band_level": level,
            "band_adv_bucket": band[schema.ADV_BUCKET],
            "median_spreads": band["q_median"],
            "flagged": zone in FLAGGED,
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized-ish scoring of a prepared frame (already has perf + bucket)."""
        recs = []
        for _, r in df.iterrows():
            band, level = self._lookup(r[schema.ALGO], r[schema.MARKET],
                                       r[schema.ADV_BUCKET])
            if band is None:
                recs.append(("NO_BAND", level, False))
                continue
            z = classify(r[schema.PERF_IN_SPREADS], band)
            recs.append((z, level, z in FLAGGED))
        out = df.copy()
        out["zone"], out["band_level"], out["flagged"] = zip(*recs)
        return out

    def _bucket_for(self, pct_adv):
        if pct_adv is None or pd.isna(pct_adv):
            return "unknown"
        b = pd.cut([pct_adv], bins=list(self.cfg.adv_bucket_edges),
                   labels=list(self.cfg.adv_bucket_labels), include_lowest=True)[0]
        return b if pd.notna(b) else "unknown"
