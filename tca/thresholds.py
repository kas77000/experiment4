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
IN_RANGE = "IN_RANGE"    # inside the threshold range -> acceptable
OUT_LOW = "OUT_LOW"      # below range: underperformance -> flag / justify
OUT_HIGH = "OUT_HIGH"    # above range: suspiciously good -> flag / justify
FLAGGED = {OUT_LOW, OUT_HIGH}


def _band_row(perf: np.ndarray, cfg) -> dict:
    """Compute the range bounds + summary stats for one group's perf array."""
    lo, hi = cfg.range_percentiles
    q = np.nanpercentile(perf, [lo, 50.0, hi])
    med = float(np.nanmedian(perf))
    mad = float(np.nanmedian(np.abs(perf - med)))  # robust dispersion
    return {
        "n": int(np.sum(~np.isnan(perf))),
        "q_lo": float(q[0]),
        "q_median": float(q[1]),
        "q_hi": float(q[2]),
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
            "q_lo", "q_median", "q_hi", "mad_spreads"]
    out = pd.DataFrame(rows)[cols]
    out["trusted"] = out["n"] >= cfg.min_group_n
    return out.sort_values(["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET]) \
              .reset_index(drop=True)


def classify(perf: float, band: pd.Series) -> str:
    """Map a normalized performance to a zone given one band row.

    Inside the range is acceptable; anything outside is flagged and must be
    justified.
    """
    if perf < band["q_lo"]:
        return OUT_LOW
    if perf > band["q_hi"]:
        return OUT_HIGH
    return IN_RANGE


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
