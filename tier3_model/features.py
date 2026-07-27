"""Build the design matrix for the cost model.

Feature choices are not arbitrary -- they follow the shape of the empirical
market-impact literature:

    sqrt_adv    sqrt(%ADV). The square-root law (Almgren et al.; Kissell's
                I-Star) is the most reproducible regularity in execution:
                impact grows with the SQUARE ROOT of size, not linearly.
    log_pov     participation rate. Urgency: trading 30% of volume costs more
                per share than trading 3%.
    log_dur     horizon. Longer working -> more exposure, less impact.
    log_spread  liquidity tier left over after sigma_expected normalization.
    log_vol     volatility regime left over after normalization.
    sqrt_adv x log_pov   size and urgency interact: a big order traded fast is
                worse than the two effects added.

Numeric features are standardized (mean 0, sd 1) using TRAINING statistics only,
which keeps the regression well conditioned and makes coefficient magnitudes
directly comparable when attributing cost drivers in diagnostics.py.

Everything degrades: a feature whose source column is missing is dropped from
the spec entirely; individual NaNs are imputed with the training median.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tca import schema

INTERACTION = "sqrt_adv_x_pov"


@dataclass
class FeatureSpec:
    """Everything needed to rebuild the same design matrix on new data."""
    numeric: list[str] = field(default_factory=list)
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)
    algo_levels: list[str] = field(default_factory=list)
    market_levels: list[str] = field(default_factory=list)
    has_side: bool = False

    @property
    def names(self) -> list[str]:
        cols = ["intercept"] + list(self.numeric)
        cols += [f"algo={a}" for a in self.algo_levels]
        cols += [f"market={m}" for m in self.market_levels]
        if self.has_side:
            cols.append("side_buy")
        return cols


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of a column. Single-row frames (from `row.to_frame().T`) come
    back as object dtype, so coerce rather than assume."""
    return pd.to_numeric(df[col], errors="coerce")


def _raw_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Derive the numeric feature columns that the source data supports."""
    out = pd.DataFrame(index=df.index)

    if schema.PCT_ADV in df.columns:
        out["sqrt_adv"] = np.sqrt(_num(df, schema.PCT_ADV).clip(lower=0) / 100.0)
    if schema.PARTICIPATION in df.columns:
        out["log_pov"] = np.log(_num(df, schema.PARTICIPATION).clip(lower=1e-4))
    if schema.DURATION_MIN in df.columns:
        out["log_dur"] = np.log(_num(df, schema.DURATION_MIN).clip(lower=1e-2))
    if schema.SPREAD_BPS in df.columns:
        out["log_spread"] = np.log(_num(df, schema.SPREAD_BPS).clip(lower=1e-2))
    if schema.VOLATILITY in df.columns:
        out["log_vol"] = np.log(_num(df, schema.VOLATILITY).clip(lower=1e-2))

    if cfg.include_interactions and {"sqrt_adv", "log_pov"} <= set(out.columns):
        out[INTERACTION] = out["sqrt_adv"] * out["log_pov"]

    return out.replace([np.inf, -np.inf], np.nan)


def fit_spec(df: pd.DataFrame, cfg) -> FeatureSpec:
    """Learn imputation medians, standardization stats and dummy levels."""
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
        levels = sorted(df[schema.ALGO].dropna().astype(str).unique())
        spec.algo_levels = levels[1:]      # drop first -> baseline in intercept

    if schema.MARKET in df.columns:
        levels = sorted(df[schema.MARKET].dropna().astype(str).unique())
        spec.market_levels = levels[1:]

    spec.has_side = schema.SIDE in df.columns
    return spec


def design(df: pd.DataFrame, spec: FeatureSpec, cfg) -> np.ndarray:
    """Materialize the design matrix for `df` under an already-fitted spec."""
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


def coverage(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """How much of each difficulty input actually arrived. Printed every run."""
    rows = []
    for col in schema.DIFFICULTY:
        present = col in df.columns
        pct = 100.0 * df[col].notna().mean() if present else 0.0
        rows.append({"column": col, "present": present, "non_null_pct": round(pct, 1)})
    return pd.DataFrame(rows)
