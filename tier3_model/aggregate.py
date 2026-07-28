"""Slice-level significance testing --- where the systematic problems actually live.

A single order's z is mostly noise: the market moves over the working horizon and
you get what you get. Averaged over a few hundred orders that noise cancels and a
real effect becomes visible:

    t = mean_z / (sd_z / sqrt(n))

A broker that is 0.18 sigma worse per order is invisible to ANY single-order
threshold -- it barely shifts the tail rate -- but over 2,500 orders it is a
t-stat around -9. That asymmetry is the reason the exception report and the slice
report are two different products, and why only one of them finds root causes.

Because we test many slices at once, raw p-values overstate significance: test 40
slices at 5% and you expect two false positives for free. `slice_stats` therefore
reports Benjamini-Hochberg q-values (false discovery rate) alongside p, and the
verdict column uses q.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tca import schema

# Slices worth testing by default; missing columns are skipped silently.
DEFAULT_DIMENSIONS = [
    [schema.ALGO],
    [schema.BROKER],
    [schema.ADV_BUCKET],
    [schema.ALGO, schema.ADV_BUCKET],
    [schema.BROKER, schema.ALGO],
]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values. Standard step-up procedure."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return q

    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def slice_stats(scored: pd.DataFrame, by, min_n: int = 30) -> pd.DataFrame:
    """Mean z with a t-test, per slice of `by`."""
    by = [b for b in ([by] if isinstance(by, str) else by) if b in scored.columns]
    if not by:
        return pd.DataFrame()

    g = scored.groupby(by, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "mean_z": g["z"].mean(),
        "sd_z": g["z"].std(ddof=1),
        "mean_residual_bps": g["residual_bps"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
    })
    out = out[out["n"] >= min_n].copy()
    if not len(out):
        return out

    out["se"] = out["sd_z"] / np.sqrt(out["n"])
    out["t_stat"] = out["mean_z"] / out["se"].replace(0, np.nan)
    out["p_value"] = 2 * stats.t.sf(out["t_stat"].abs(), df=out["n"] - 1)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())

    out["verdict"] = np.select(
        [(out["q_value"] < 0.01) & (out["mean_z"] < 0),
         (out["q_value"] < 0.05) & (out["mean_z"] < 0),
         (out["q_value"] < 0.05) & (out["mean_z"] > 0)],
        ["UNDERPERFORMS (strong)", "UNDERPERFORMS", "OUTPERFORMS"],
        default="no evidence",
    )
    out["dimension"] = " x ".join(by)
    cols = ["dimension", "n", "mean_z", "se", "t_stat", "p_value", "q_value",
            "mean_residual_bps", "flag_rate_pct", "verdict"]
    return out[cols].sort_values("t_stat")


def slice_report(scored: pd.DataFrame, dimensions=None, min_n: int = 30) -> pd.DataFrame:
    """Run every dimension and stack the results into one table."""
    dims = dimensions or DEFAULT_DIMENSIONS
    frames = []
    for by in dims:
        s = slice_stats(scored, by, min_n=min_n)
        if len(s):
            frames.append(s.reset_index())
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["mean_z", "se", "t_stat", "mean_residual_bps", "flag_rate_pct"]:
        out[c] = out[c].round(3)
    for c in ["p_value", "q_value"]:
        out[c] = out[c].map(lambda v: f"{v:.2e}" if pd.notna(v) and v < 1e-3
                            else (round(v, 4) if pd.notna(v) else v))
    return out


def dispersion_stats(scored: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    """Test whether a slice is more INCONSISTENT than the rest of the book.

    This is the test that matters for schedule-following algos, and a mean test
    cannot substitute for it. Bad volume-curve tracking does not bias a VWAP
    order in a direction -- it widens the distribution. The order lands
    somewhere random on a wider spread, so on average it looks fine.

    Measured on the VWAP demo book, the injected curve-drift cohort has
    mean z = -0.21 (small) but sd z = 1.36 against 0.49 for clean orders --
    nearly 3x the dispersion, almost no shift in the mean. At broker level the
    sloppiest broker gives t = -1.96 on the mean, which does NOT survive FDR
    correction across ~40 slices, but Levene p = 1.3e-04 on the variance, which
    clearly does.

    Levene rather than an F-test, because it uses absolute deviations from the
    group centre and so does not assume normality -- and z-scores from a
    quantile fit have heavier tails than a normal.
    """
    by = [b for b in ([by] if isinstance(by, str) else by) if b in scored.columns]
    if not by:
        return pd.DataFrame()

    z_all = scored["z"]
    rows = []
    for keys, g in scored.groupby(by, dropna=False):
        if len(g) < min_n:
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        rest = z_all[~z_all.index.isin(g.index)].dropna()
        gz = g["z"].dropna()
        if len(gz) < min_n or len(rest) < min_n:
            continue
        try:
            _, p = stats.levene(gz, rest)
        except Exception:
            p = np.nan
        rest_sd = float(rest.std(ddof=1))
        rows.append({
            **dict(zip(by, keys)),
            "n": len(gz),
            "sd_z": float(gz.std(ddof=1)),
            "mean_z": float(gz.mean()),
            "mean_abs_z": float(gz.abs().mean()),
            "sd_ratio": float(gz.std(ddof=1) / rest_sd) if rest_sd else np.nan,
            "flag_rate_pct": 100.0 * g["flagged"].mean(),
            "p_value": p,
        })

    out = pd.DataFrame(rows)
    if not len(out):
        return out

    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["verdict"] = np.select(
        [(out["q_value"] < 0.01) & (out["sd_ratio"] > 1),
         (out["q_value"] < 0.05) & (out["sd_ratio"] > 1),
         (out["q_value"] < 0.05) & (out["sd_ratio"] < 1)],
        ["INCONSISTENT (strong)", "INCONSISTENT", "more consistent"],
        default="no evidence")
    out["dimension"] = " x ".join(by)
    cols = ["dimension"] + by + ["n", "sd_z", "sd_ratio", "mean_z", "mean_abs_z",
                                 "flag_rate_pct", "p_value", "q_value", "verdict"]
    return out[cols].sort_values("sd_ratio", ascending=False)


def dispersion_report(scored: pd.DataFrame, dimensions=None,
                      min_n: int = 100) -> pd.DataFrame:
    """Run the consistency test across every dimension and stack the results."""
    dims = dimensions or [[schema.ALGO], [schema.BROKER], [schema.ADV_BUCKET],
                          [schema.BROKER, schema.ALGO]]
    frames = []
    for by in dims:
        s = dispersion_stats(scored, by, min_n=min_n)
        if len(s):
            frames.append(s.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["sd_z", "sd_ratio", "mean_z", "mean_abs_z", "flag_rate_pct"]:
        out[c] = out[c].round(3)
    for c in ["p_value", "q_value"]:
        out[c] = out[c].map(lambda v: f"{v:.2e}" if pd.notna(v) and v < 1e-3
                            else (round(v, 4) if pd.notna(v) else v))
    return out


def significant(slices: pd.DataFrame) -> pd.DataFrame:
    """Only the slices that survive FDR correction --- your actual findings list."""
    if not len(slices):
        return slices
    return slices[slices["verdict"] != "no evidence"]
