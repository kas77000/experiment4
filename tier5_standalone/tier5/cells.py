"""Where a row belongs, and where its outputs go.

Both fit and score derive cells through this module and nothing else. That is
what guarantees the band written for HK/VWAP is the one score looks up for
those same rows -- if the two sides derived cells independently they would
drift apart the first time a strategy name changed case.

Region is the Sym suffix, already turned into schema.MARKET by
config.PRE_TRANSFORM. Strategy is the Strategy column. Neither is ever typed on
a command line, so a run cannot be mislabelled.
"""

from __future__ import annotations

import os

import pandas as pd

from tca import schema

UNKNOWN_REGION = "UNKNOWN"
UNKNOWN_STRATEGY = "UNKNOWN"


def _region_series(df: pd.DataFrame) -> pd.Series:
    if schema.MARKET not in df.columns:
        return pd.Series(UNKNOWN_REGION, index=df.index, dtype="object")
    return (df[schema.MARKET].astype(str).str.strip().str.upper()
            .replace({"": UNKNOWN_REGION, "NAN": UNKNOWN_REGION,
                      "NONE": UNKNOWN_REGION}))


def _strategy_series(df: pd.DataFrame) -> pd.Series:
    if schema.ALGO not in df.columns:
        return pd.Series(UNKNOWN_STRATEGY, index=df.index, dtype="object")
    return (df[schema.ALGO].astype(str).str.strip()
            .replace({"": UNKNOWN_STRATEGY, "nan": UNKNOWN_STRATEGY,
                      "None": UNKNOWN_STRATEGY}))


def cells(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Every (region, strategy) present, with its rows. Sorted for stable output."""
    region = _region_series(df)
    strategy = _strategy_series(df)
    out = []
    for (r, s), g in df.groupby([region, strategy], dropna=False, observed=False):
        out.append((str(r), str(s), g))
    return sorted(out, key=lambda t: (t[0], t[1]))


def date_range(df: pd.DataFrame):
    """(min, max) order date, or (None, None) when there is no usable date."""
    if schema.ORDER_DATE not in df.columns:
        return None, None
    d = pd.to_datetime(df[schema.ORDER_DATE], errors="coerce").dropna()
    if d.empty:
        return None, None
    return d.min(), d.max()


def period_label(df: pd.DataFrame) -> str | None:
    """'2026-07' for a single month, '2025-06_2026-05' for a span, None if no dates."""
    lo, hi = date_range(df)
    if lo is None:
        return None
    if (lo.year, lo.month) == (hi.year, hi.month):
        return lo.strftime("%Y-%m")
    return f"{lo.strftime('%Y-%m')}_{hi.strftime('%Y-%m')}"


def safe(name) -> str:
    """A filesystem-safe token. Strategy names arrive from vendor extracts."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return cleaned.strip("_") or "UNKNOWN"


def band_path(bands_dir: str, region: str, strategy: str) -> str:
    return os.path.join(bands_dir, safe(region), safe(strategy) + ".json")


def out_dir(root: str, kind: str, period: str, region: str, strategy: str) -> str:
    return os.path.join(root, kind, safe(period), safe(region), safe(strategy))


def windows_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    """Do two date windows intersect? False whenever either is unknown.

    Scoring a period the band was fitted on is leakage, and it is the one
    mistake that makes the whole out-of-sample exercise meaningless.
    """
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return False
    return bool(a_lo <= b_hi and b_lo <= a_hi)
