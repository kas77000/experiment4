"""Fit percentile bands per peer group, and score orders against them.

This is the vendor-style approach (Abel Noser / Virtu-ITG universe rankings),
implemented against your own history by default: the reference distribution is a
peer group of comparable orders, and your order's position in it is the verdict.

Fitting is done at three nested levels so thin slices degrade gracefully:
    1. bucketed:  algo x market x adv_bucket   (most specific)
    2. pooled:    algo x market                (fallback)
    3. global:    all orders                   (last resort)
Scoring picks the most specific level that has >= min_group_n orders.

The reference book does not have to be your own: `fit()` takes whatever frame
you hand it, so passing a peer/street extract reproduces the vendor product --
you are then ranked against the market rather than against your own history,
which is the point of buying one of those services.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema

# Zone labels
IN_RANGE = "IN_RANGE"    # inside the band -> acceptable
OUT_LOW = "OUT_LOW"      # below the band: underperformance -> flag / justify
OUT_HIGH = "OUT_HIGH"    # above the band: suspiciously good -> flag / justify
NO_BAND = "NO_BAND"      # no trusted reference group at any level
FLAGGED = {OUT_LOW, OUT_HIGH}


def _band_row(perf: np.ndarray, cfg) -> dict:
    """Compute the band bounds + summary stats for one group's metric array."""
    lo, hi = cfg.range_percentiles
    n = int(np.sum(~np.isnan(perf)))
    if n == 0:
        return {"n": 0, "q_lo": np.nan, "q_median": np.nan,
                "q_hi": np.nan, "mad": np.nan}
    q = np.nanpercentile(perf, [lo, 50.0, hi])
    med = float(np.nanmedian(perf))
    mad = float(np.nanmedian(np.abs(perf - med)))  # robust dispersion
    return {
        "n": n,
        "q_lo": float(q[0]),
        "q_median": float(q[1]),
        "q_hi": float(q[2]),
        "mad": mad,
    }


def fit(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return a tidy band table with one row per (level, group).

    `df` is the REFERENCE book -- your own history, or a peer universe.
    """
    metric = cfg.metric
    rows = []

    # Level 1: bucketed
    for keys, g in df.groupby(list(cfg.group_keys), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {"level": "bucketed"}
        row.update(dict(zip(cfg.group_keys, keys)))
        row.update(_band_row(g[metric].to_numpy(), cfg))
        rows.append(row)

    # Level 2: pooled (algo x market)
    for (algo, market), g in df.groupby([schema.ALGO, schema.MARKET], dropna=False):
        row = {"level": "pooled", schema.ALGO: algo, schema.MARKET: market,
               schema.ADV_BUCKET: None}
        row.update(_band_row(g[metric].to_numpy(), cfg))
        rows.append(row)

    # Level 3: global
    row = {"level": "global", schema.ALGO: None, schema.MARKET: None,
           schema.ADV_BUCKET: None}
    row.update(_band_row(df[metric].to_numpy(), cfg))
    rows.append(row)

    cols = ["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET, "n",
            "q_lo", "q_median", "q_hi", "mad"]
    out = pd.DataFrame(rows)[cols]
    out["trusted"] = out["n"] >= cfg.min_group_n
    return out.sort_values(["level", schema.ALGO, schema.MARKET, schema.ADV_BUCKET]) \
              .reset_index(drop=True)


def classify(perf: float, band: pd.Series) -> str:
    """Map a metric value to a zone given one band row."""
    if not np.isfinite(perf):
        return NO_BAND
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
        """Score a single order (spread-normalized metric only)."""
        perf = slippage_bps / spread_bps
        bucket = self._bucket_for(pct_adv)
        band, level = self._lookup(algo, market, bucket)
        if band is None:
            return {"zone": NO_BAND, "metric": perf,
                    "band_level": None, "flagged": False}
        zone = classify(perf, band)
        return {
            "zone": zone,
            "metric": perf,
            "band_level": level,
            "band_adv_bucket": band[schema.ADV_BUCKET],
            "band_lo": float(band["q_lo"]),
            "band_median": float(band["q_median"]),
            "band_hi": float(band["q_hi"]),
            "flagged": zone in FLAGGED,
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a prepared frame (already carries the metric + adv bucket)."""
        metric = self.cfg.metric
        recs = []
        for _, r in df.iterrows():
            band, level = self._lookup(r[schema.ALGO], r[schema.MARKET],
                                       r[schema.ADV_BUCKET])
            if band is None:
                recs.append((NO_BAND, None, np.nan, np.nan, np.nan, False))
                continue
            z = classify(r[metric], band)
            recs.append((z, level, float(band["q_lo"]), float(band["q_median"]),
                         float(band["q_hi"]), z in FLAGGED))

        out = df.copy()
        (out["zone"], out["band_level"], out["band_lo"], out["band_med"],
         out["band_hi"], out["flagged"]) = zip(*recs)

        # Tier-comparable severity ranking: distance from the group median in
        # half-band-widths, so the top-level comparison can hold every tier to
        # the SAME review budget instead of comparing different queue sizes.
        half_width = ((out["band_hi"] - out["band_lo"]) / 2.0).replace(0, np.nan)
        out["rank_stat"] = (out[metric] - out["band_med"]).abs() / half_width

        if schema.NOTIONAL in out.columns and self.cfg.min_notional_review > 0:
            out["material"] = out[schema.NOTIONAL].fillna(0) >= self.cfg.min_notional_review
        else:
            out["material"] = True
        out["review_required"] = out["flagged"] & out["material"]
        return out

    def _bucket_for(self, pct_adv):
        if pct_adv is None or pd.isna(pct_adv):
            return "unknown"
        import config as root_config
        edges = root_config.DATA.adv_bucket_edges
        labels = root_config.DATA.adv_bucket_labels
        b = pd.cut([pct_adv], bins=list(edges), labels=list(labels),
                   include_lowest=True)[0]
        return b if pd.notna(b) else "unknown"


def flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """Flag rate vs difficulty -- should be much flatter than Tier 1."""
    g = scored.groupby(schema.ADV_BUCKET, dropna=False)
    return pd.DataFrame({
        "n": g.size(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
    }).round(2)
