"""Load -> clean -> derive metrics -> bucket. Produces an analysis-ready frame.

Two performance metrics come out of here, because the three tiers need
different ones:

    perf_in_spreads = slippage_bps / spread_bps          (Tier 1, Tier 2)
    perf_norm       = slippage_bps / sigma_expected_bps  (Tier 3)

`sigma_expected_bps` combines the spread and the volatility-over-horizon in
quadrature, so a 6-hour order and a 15-minute order with the same spread are no
longer scored on the same scale.
"""

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
    dropped_no_metric: int = 0     # supplied metric column present but null
    norm_full_pct: float = 0.0     # % of rows using spread+vol normalization

    def as_text(self) -> str:
        return (
            f"  rows in:                 {self.rows_in:,}\n"
            f"  - missing essential:     {self.dropped_missing:,}\n"
            f"  - bad/zero spread:       {self.dropped_bad_spread:,}\n"
            f"  - dust (< min_notional): {self.dropped_dust:,}\n"
            f"  - |perf| data error:     {self.dropped_data_error:,}\n"
            f"  - no metric value:       {self.dropped_no_metric:,}\n"
            f"  rows out:                {self.rows_out:,}\n"
            f"  vol-normalized:          {self.norm_full_pct:.1f}%"
            f"  (rest fall back to spread-only)"
        )


def load_orders(df_raw: pd.DataFrame, column_map: dict,
                pre_transform=None) -> pd.DataFrame:
    """Rename vendor columns to canonical names and coerce types.

    `column_map` is canonical -> source. Unknown canonical targets are skipped
    so partial extracts still load.

    `pre_transform` runs BEFORE the rename and is where anything a rename cannot
    express belongs: unit scaling ($mln -> currency, lots -> shares), deriving a
    field from another (market from a ticker suffix), splitting composite
    columns. Keep it defensive -- it also runs on the synthetic demo, which has
    none of those source columns.
    """
    df = df_raw.copy()
    if pre_transform is not None:
        df = pre_transform(df)

    rename = {src: canon for canon, src in column_map.items() if src in df.columns}
    df = df.rename(columns=rename).copy()

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


def normalize_units(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Convert volatility and %ADV into the units the rest of the code assumes.

    Internally: volatility is DAILY vol in bps, pct_adv is a PERCENT (3.5 = 3.5%).
    Extracts disagree about both, and getting either wrong silently rescales
    `sigma_expected` -- which shows up as a miscalibrated band, not as an error.
    `check_extract.py` infers the right settings from your file.
    """
    df = df.copy()

    if schema.VOLATILITY in df.columns:
        scale = {"bps": 1.0, "pct": 100.0, "fraction": 10_000.0}
        if cfg.volatility_unit not in scale:
            raise ValueError(f"Unknown volatility_unit {cfg.volatility_unit!r}; "
                             f"pick one of {list(scale)}")
        df[schema.VOLATILITY] = df[schema.VOLATILITY] * scale[cfg.volatility_unit]

    if schema.PCT_ADV in df.columns:
        scale = {"pct": 1.0, "fraction": 100.0}
        if cfg.pct_adv_unit not in scale:
            raise ValueError(f"Unknown pct_adv_unit {cfg.pct_adv_unit!r}; "
                             f"pick one of {list(scale)}")
        df[schema.PCT_ADV] = df[schema.PCT_ADV] * scale[cfg.pct_adv_unit]

    # Participation is normalized to a FRACTION (0-1), unlike the two above,
    # because that is what the code formats and clips against.
    if schema.PARTICIPATION in df.columns:
        scale = {"pct": 0.01, "fraction": 1.0}
        if cfg.participation_unit not in scale:
            raise ValueError(f"Unknown participation_unit {cfg.participation_unit!r}; "
                             f"pick one of {list(scale)}")
        df[schema.PARTICIPATION] = df[schema.PARTICIPATION] * scale[cfg.participation_unit]

    return df


def _sign_factor(sign_convention: str) -> float:
    """+1 if higher is already better, -1 if the extract reports a cost."""
    if sign_convention == "positive_is_good":
        return 1.0
    if sign_convention == "cost":
        return -1.0
    raise ValueError(f"Unknown SLIPPAGE_SIGN: {sign_convention!r}")


def clean(df: pd.DataFrame, cfg, sign_convention: str) -> tuple[pd.DataFrame, CleanReport]:
    """Drop unusable rows. Returns (clean_df, report).

    Note: we drop *data errors* and *dust*, NOT legitimate performance outliers --
    those are what the thresholds exist to flag.
    """
    n0 = len(df)

    df = df.copy()
    sign = _sign_factor(sign_convention)
    df[schema.SLIPPAGE_BPS] = df[schema.SLIPPAGE_BPS] * sign
    # A supplied spread-normalised metric comes out of the same system as the
    # raw slippage, so one sign convention governs both. Flipping only one of
    # them would leave the band and the diagnostic columns disagreeing about
    # which tail is the bad one.
    supplied_metric = schema.PERF_IN_SPREADS in df.columns
    if supplied_metric:
        df[schema.PERF_IN_SPREADS] = (
            pd.to_numeric(df[schema.PERF_IN_SPREADS], errors="coerce") * sign)

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

    # 4) implausible normalized performance -> data error. Measured on the
    #    metric that will actually be banded, so the guard and the band are
    #    never pointed at two different quantities.
    perf = (df[schema.PERF_IN_SPREADS] if supplied_metric
            else df[schema.SLIPPAGE_BPS] / df[schema.SPREAD_BPS])
    before = len(df)
    df = df[perf.abs().fillna(0.0) <= cfg.max_abs_perf_spreads]
    dropped_data_error = before - len(df)

    # 5) a supplied metric that is null -- the strategy's source column was
    #    absent from the export, or blank on these rows. Such an order cannot
    #    be banded or scored, and dropping it without a count is exactly the
    #    silent shrinkage this report exists to prevent.
    dropped_no_metric = 0
    if supplied_metric:
        before = len(df)
        df = df[df[schema.PERF_IN_SPREADS].notna()]
        dropped_no_metric = before - len(df)

    report = CleanReport(
        rows_in=n0,
        dropped_missing=dropped_missing,
        dropped_bad_spread=dropped_bad_spread,
        dropped_dust=dropped_dust,
        dropped_data_error=dropped_data_error,
        dropped_no_metric=dropped_no_metric,
        rows_out=len(df),
    )
    return df.reset_index(drop=True), report


def add_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1/2 metric: performance expressed in units of spread.

    If the extract already supplies it -- COLUMN_MAP points PERF_IN_SPREADS at a
    column such as `ePvwap/Sprd`, which is pre-divided at source -- that column
    is kept exactly as it arrived. Dividing an already-normalised number by the
    spread a second time is the one mistake here that produces output which
    looks entirely plausible: the band still fits, the curve still looks like a
    curve, and every bound is wrong by a factor of the spread.
    """
    df = df.copy()
    if schema.PERF_IN_SPREADS not in df.columns:
        df[schema.PERF_IN_SPREADS] = df[schema.SLIPPAGE_BPS] / df[schema.SPREAD_BPS]
    return df


def add_sigma_expected(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tier 3 noise unit: spread and volatility-over-horizon, added in quadrature.

        sigma_expected = sqrt( (k_spread*spread)^2 + (w_vol*vol*sqrt(T))^2 )

    T is the order's duration as a fraction of a trading session. Rows lacking
    volatility or duration keep the spread term alone and are tagged
    norm_basis="spread_only", so a partial extract still scores -- just with a
    cruder scale on those rows.
    """
    df = df.copy()
    spread_term = cfg.k_spread * df[schema.SPREAD_BPS]

    if schema.VOLATILITY in df.columns:
        if schema.DURATION_MIN in df.columns:
            dur = df[schema.DURATION_MIN]
            if cfg.default_duration_min:
                dur = dur.fillna(cfg.default_duration_min)
        elif cfg.default_duration_min:
            # No duration column at all. Assuming one horizon for every order is
            # crude, but it is NOT the same as dropping volatility: vol still
            # varies order to order, so a volatile name is correctly given a
            # wider band than a quiet one. Replace with real durations when you
            # can get them -- and see the note in config.py before trusting it.
            dur = pd.Series(float(cfg.default_duration_min), index=df.index)
        else:
            dur = None
    else:
        dur = None

    if dur is not None:
        horizon = (dur / cfg.minutes_per_day).clip(lower=0.0)
        vol_term = cfg.vol_horizon_weight * df[schema.VOLATILITY] * np.sqrt(horizon)
        usable = vol_term.notna() & (vol_term > 0)
        vol_term = vol_term.where(usable, 0.0)
    else:
        vol_term = pd.Series(0.0, index=df.index)
        usable = pd.Series(False, index=df.index)

    sigma = np.sqrt(spread_term.astype(float) ** 2 + vol_term.astype(float) ** 2)
    df[schema.SIGMA_EXPECTED_BPS] = sigma.clip(lower=cfg.min_sigma_bps)
    df[schema.NORM_BASIS] = np.where(usable, "spread+vol", "spread_only")
    return df


def add_norm_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 3 metric: slippage in units of its own expected noise. Roughly unit-scale."""
    df = df.copy()
    df[schema.PERF_NORM] = df[schema.SLIPPAGE_BPS] / df[schema.SIGMA_EXPECTED_BPS]
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


def prepare(df_raw: pd.DataFrame, column_map: dict, cfg, sign_convention: str,
            pre_transform=None):
    """Full pipeline: load -> units -> clean -> both metrics -> buckets.

    Returns (df, CleanReport). The frame it produces feeds all three tiers, so
    every tier scores exactly the same rows -- which is what makes the
    comparison in run.py meaningful.
    """
    df = load_orders(df_raw, column_map, pre_transform=pre_transform)
    df = normalize_units(df, cfg)
    df, report = clean(df, cfg, sign_convention)
    df = add_metric(df)
    df = add_sigma_expected(df, cfg)
    df = add_norm_metric(df)
    df = add_buckets(df, cfg)

    if len(df):
        report.norm_full_pct = 100.0 * (df[schema.NORM_BASIS] == "spread+vol").mean()
    return df, report
