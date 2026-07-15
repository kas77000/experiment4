"""Load -> clean -> derive metric -> bucket. Produces an analysis-ready frame."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tca import schema


@dataclass
class CleanReport:
    """What the cleaning step dropped, so nothing disappears silently."""
    rows_in: int
    dropped_missing: int
    dropped_bad_spread: int
    dropped_dust: int
    dropped_data_error: int
    rows_out: int

    def as_text(self) -> str:
        return (
            f"  rows in:            {self.rows_in:,}\n"
            f"  - missing essential:{self.dropped_missing:,}\n"
            f"  - bad/zero spread:  {self.dropped_bad_spread:,}\n"
            f"  - dust (< min_notional): {self.dropped_dust:,}\n"
            f"  - |perf| data error:{self.dropped_data_error:,}\n"
            f"  rows out:           {self.rows_out:,}"
        )


def load_orders(df_raw: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Rename vendor columns to canonical names and coerce types.

    `column_map` is canonical -> source. Unknown canonical targets are skipped
    so partial extracts still load.
    """
    rename = {src: canon for canon, src in column_map.items() if src in df_raw.columns}
    df = df_raw.rename(columns=rename).copy()

    for col in schema.NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_essential = [c for c in schema.ESSENTIAL if c not in df.columns]
    if missing_essential:
        raise ValueError(
            f"Extract is missing essential columns after mapping: {missing_essential}. "
            f"Check COLUMN_MAP in config.py."
        )
    return df


def _apply_sign(df: pd.DataFrame, sign_convention: str) -> pd.Series:
    """Return signed slippage where HIGHER = BETTER, regardless of source convention."""
    s = df[schema.SLIPPAGE_BPS]
    if sign_convention == "positive_is_good":
        return s
    if sign_convention == "cost":
        return -s
    raise ValueError(f"Unknown SLIPPAGE_SIGN: {sign_convention!r}")


def clean(df: pd.DataFrame, cfg, sign_convention: str) -> tuple[pd.DataFrame, CleanReport]:
    """Drop unusable rows. Returns (clean_df, report).

    Note: we drop *data errors* and *dust*, NOT legitimate performance outliers —
    those are what the thresholds exist to flag.
    """
    n0 = len(df)

    df = df.copy()
    df[schema.SLIPPAGE_BPS] = _apply_sign(df, sign_convention)

    # 1) missing essentials
    before = len(df)
    df = df.dropna(subset=schema.ESSENTIAL)
    dropped_missing = before - len(df)

    # 2) bad / non-positive spread (can't normalize by it)
    before = len(df)
    df = df[df[schema.SPREAD_BPS] >= cfg.min_spread_bps]
    dropped_bad_spread = before - len(df)

    # 3) dust orders
    dropped_dust = 0
    if schema.NOTIONAL in df.columns and cfg.min_notional > 0:
        before = len(df)
        df = df[df[schema.NOTIONAL].fillna(np.inf) >= cfg.min_notional]
        dropped_dust = before - len(df)

    # 4) implausible normalized performance -> data error
    perf = df[schema.SLIPPAGE_BPS] / df[schema.SPREAD_BPS]
    before = len(df)
    df = df[perf.abs() <= cfg.max_abs_perf_spreads]
    dropped_data_error = before - len(df)

    report = CleanReport(
        rows_in=n0,
        dropped_missing=dropped_missing,
        dropped_bad_spread=dropped_bad_spread,
        dropped_dust=dropped_dust,
        dropped_data_error=dropped_data_error,
        rows_out=len(df),
    )
    return df.reset_index(drop=True), report


def add_metric(df: pd.DataFrame) -> pd.DataFrame:
    """The core normalized metric: performance expressed in units of spread."""
    df = df.copy()
    df[schema.PERF_IN_SPREADS] = df[schema.SLIPPAGE_BPS] / df[schema.SPREAD_BPS]
    return df


def add_buckets(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Assign each order to a %ADV difficulty bucket (or 'unknown' if no %ADV)."""
    df = df.copy()
    if schema.PCT_ADV in df.columns:
        b = pd.cut(
            df[schema.PCT_ADV],
            bins=list(cfg.adv_bucket_edges),
            labels=list(cfg.adv_bucket_labels),
            include_lowest=True,
        )
        df[schema.ADV_BUCKET] = b.astype("object").where(b.notna(), "unknown")
    else:
        df[schema.ADV_BUCKET] = "unknown"
    return df


def prepare(df_raw: pd.DataFrame, column_map: dict, cfg, sign_convention: str):
    """Full pipeline: load -> clean -> metric -> buckets. Returns (df, CleanReport)."""
    df = load_orders(df_raw, column_map)
    df, report = clean(df, cfg, sign_convention)
    df = add_metric(df)
    df = add_buckets(df, cfg)
    return df, report
