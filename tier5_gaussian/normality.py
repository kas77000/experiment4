"""Does 3 sigma mean what it says on this book?

The band in band.py is only as good as the normality assumption underneath
it. k = 3 promises 0.27% flagged; whether the book delivers that is an
empirical question, and this module answers it three ways, in increasing
order of how convincing they are to somebody who does not want to read
statistics:

  1. coverage_table  -- promised vs delivered at k = 1, 2, 3, 4
  2. required_k      -- the k that WOULD deliver 0.27% here, per tail
  3. shape_stats     -- skew, excess kurtosis, D'Agostino K2
  4. qq_plot         -- the picture

scipy and matplotlib are both optional here and neither is in the scoring
path: without scipy the K2 test is skipped and everything else still runs,
without matplotlib the plot is skipped. Both skips are reported.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tca import schema
from tier5_gaussian import band, config as t5cfg

NORMAL_K = (1.0, 2.0, 3.0, 4.0)

# Two-sided tail mass a 3-sigma band promises under normality.
NOMINAL_OUTSIDE = 0.0027


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def promised_inside(k: float) -> float:
    """P(|Z| <= k) for a standard normal, in closed form."""
    return math.erf(k / math.sqrt(2.0))


def coverage_table(x, centre: float, scale: float,
                   ks=NORMAL_K) -> pd.DataFrame:
    """Promised vs delivered coverage at each k. The clearest exhibit."""
    x = _finite(x)
    if x.size == 0 or not np.isfinite(scale) or scale <= 0:
        return pd.DataFrame()
    d = np.abs(x - centre) / scale
    rows = []
    for k in ks:
        p_in = promised_inside(k)
        a_in = float(np.mean(d <= k))
        p_out, a_out = 1.0 - p_in, 1.0 - a_in
        rows.append({
            "k": k,
            "promised_inside_pct": 100.0 * p_in,
            "actual_inside_pct": 100.0 * a_in,
            "promised_outside_pct": 100.0 * p_out,
            "actual_outside_pct": 100.0 * a_out,
            "ratio": (a_out / p_out) if p_out > 0 else np.nan,
            "n_outside": int(round(a_out * x.size)),
        })
    return pd.DataFrame(rows)


def required_k(x, centre: float, scale: float,
               target_outside: float = NOMINAL_OUTSIDE) -> dict:
    """The k this book would need to actually flag `target_outside`.

    k_symmetric is the honest single answer. k_lo and k_hi decompose it per
    tail, which is where the asymmetry a symmetric band cannot express shows
    up: if they differ a lot, no single k serves both tails.
    """
    x = _finite(x)
    if x.size == 0 or not np.isfinite(scale) or scale <= 0:
        return {"k_symmetric": np.nan, "k_lo": np.nan, "k_hi": np.nan}
    d = np.abs(x - centre) / scale
    half = target_outside / 2.0
    return {
        "k_symmetric": float(np.quantile(d, 1.0 - target_outside)),
        "k_lo": float((centre - np.quantile(x, half)) / scale),
        "k_hi": float((np.quantile(x, 1.0 - half) - centre) / scale),
    }


def shape_stats(x) -> dict:
    """Skew, excess kurtosis and D'Agostino K2.

    The p-value carries almost no information at this sample size -- with n in
    the thousands every formal normality test rejects, because it is testing
    'exactly normal' and nothing real ever is. The effect sizes and the
    coverage table are the evidence; the test is here because someone asks.
    """
    x = _finite(x)
    s = pd.Series(x)
    out = {
        "n": int(x.size),
        "skew": float(s.skew()) if x.size > 2 else np.nan,
        "excess_kurtosis": float(s.kurt()) if x.size > 3 else np.nan,
        "dagostino_k2": np.nan,
        "p_value": np.nan,
        "test_note": "",
    }
    if x.size < 20:
        out["test_note"] = "n too small for K2"
        return out
    try:
        from scipy import stats as sps
    except ImportError:
        out["test_note"] = "scipy not installed -- K2 skipped"
        return out
    stat, p = sps.normaltest(x)
    out["dagostino_k2"] = float(stat)
    out["p_value"] = float(p)
    return out


def evidence(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row of evidence per group, for the ALL and algo levels only.

    The adv_bucket and cross levels are excluded on purpose: their thin cells
    cannot support a tail estimate, and the table would be longer than it is
    informative.
    """
    metric = cfg.metric
    est = cfg.estimator
    groups = [(t5cfg.LEVEL_ALL, None, df)]
    if schema.ALGO in df.columns:
        for algo, g in df.groupby(schema.ALGO, dropna=False, observed=False):
            groups.append((t5cfg.LEVEL_ALGO, algo, g))

    rows = []
    for level, algo, g in groups:
        x = g[metric].to_numpy()
        e = band.estimates(x, cfg.k_sigma)
        centre, scale = e[f"centre_{est}"], e[f"scale_{est}"]
        row = {"level": level, schema.ALGO: algo, "n": e["n"],
               "centre": centre, "scale": scale,
               "lo": e[f"lo_{est}"], "hi": e[f"hi_{est}"]}

        cov = coverage_table(x, centre, scale, ks=(cfg.k_sigma,))
        if len(cov):
            row["promised_outside_pct"] = float(cov.iloc[0]["promised_outside_pct"])
            row["actual_outside_pct"] = float(cov.iloc[0]["actual_outside_pct"])
            row["ratio"] = float(cov.iloc[0]["ratio"])
        else:
            row.update({"promised_outside_pct": np.nan,
                        "actual_outside_pct": np.nan, "ratio": np.nan})

        row.update(required_k(x, centre, scale))
        st = shape_stats(x)
        row.update({"skew": st["skew"],
                    "excess_kurtosis": st["excess_kurtosis"],
                    "p_value": st["p_value"]})
        rows.append(row)

    return pd.DataFrame(rows)


def qq_plot(x, path: str, title: str) -> str:
    """Write a normal QQ plot. Returns the line to print."""
    x = _finite(x)
    if x.size == 0:
        return "  QQ plot skipped (no finite values)."
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sps
    except ImportError as exc:
        return f"  QQ plot skipped ({exc.name} not installed)."

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    sps.probplot(x, dist="norm", plot=ax)
    ax.set_title(title)
    ax.get_lines()[0].set_markersize(2.0)
    ax.get_lines()[0].set_alpha(0.35)
    ax.set_xlabel("normal theoretical quantiles")
    ax.set_ylabel("observed quantiles")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return (f"  Wrote {path}\n"
            f"  A straight line means normal. The curl at the ends is the fat tail.")
