"""Freeze a fitted band to disk, and load it back to score a later period.

This is what makes the exercise worth doing. Fitting and scoring the same book
tells you almost nothing: 1.7% flags because 1.7% was *defined* as flagged, and
the band was dragged toward the very outliers it then counts. Freeze the band
and apply it unchanged to orders it has never seen and the flag rate becomes a
measurement -- if July flags 4% against a band that flagged 1.5% on the fit
year, something real changed.

Both estimators are stored even though only one scores, so switching estimator
later does not require a refit. A reference snapshot of the fit book travels
with the band so `drift_report` can separate "the market moved" from "execution
degraded".
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from tca import schema
from tier5 import cells, config as t5cfg, normality

FORMAT_VERSION = 1

# Config fields that affect scoring and must travel with the band. Anything not
# listed falls back to the current default on load.
SCORING_FIELDS = ["k_sigma", "metric", "estimator", "min_notional_review"]

# Features whose medians are snapshotted for the drift report.
REFERENCE_FEATURES = [schema.SPREAD_BPS, schema.PCT_ADV,
                      schema.VOLATILITY, schema.DURATION_MIN]

DRIFT_MEDIAN_PCT = 25.0     # a feature median moving more than this is called out


def _f(v):
    """JSON cannot hold NaN portably. None is honest about a missing number."""
    v = float(v)
    return v if np.isfinite(v) else None


def _reference(df: pd.DataFrame, cfg, est: dict, flag_rate_pct: float) -> dict:
    x = df[cfg.metric].to_numpy()
    e = cfg.estimator
    req = normality.required_k(x, est[f"centre_{e}"], est[f"scale_{e}"])
    shape = normality.shape_stats(x)
    medians = {}
    for col in REFERENCE_FEATURES:
        if col in df.columns:
            med = pd.to_numeric(df[col], errors="coerce").median()
            if np.isfinite(med):
                medians[col] = float(med)
    return {
        "flag_rate_pct": float(flag_rate_pct),
        "skew": _f(shape["skew"]),
        "excess_kurtosis": _f(shape["excess_kurtosis"]),
        "k_required": _f(req["k_symmetric"]),
        "k_required_lo": _f(req["k_lo"]),
        "k_required_hi": _f(req["k_hi"]),
        "feature_medians": medians,
    }


def save(est: dict, cfg, path: str, *, region: str, strategy: str,
         source_csv: str, period: str | None, df: pd.DataFrame,
         flag_rate_pct: float, budget: dict | None = None) -> str:
    """Write one frozen band to JSON."""
    e = cfg.estimator
    d_lo, d_hi = cells.date_range(df)
    payload = {
        "format_version": FORMAT_VERSION,
        "fitted_at": dt.datetime.now().isoformat(timespec="seconds"),
        "region": region,
        "strategy": strategy,
        "source_csv": source_csv,
        "fit_period": period,
        "fit_date_min": d_lo.strftime("%Y-%m-%d") if d_lo is not None else None,
        "fit_date_max": d_hi.strftime("%Y-%m-%d") if d_hi is not None else None,
        "metric": cfg.metric,
        # What lo/hi are counted in. Stamped so the artefact is readable on its
        # own six months from now, and so a band frozen in bps is recognisable
        # next to one frozen in spreads.
        "metric_units": t5cfg.units_of(cfg.metric),
        "estimator": cfg.estimator,
        "k_sigma": float(cfg.k_sigma),
        # How k was arrived at. A band at k = 5.9 must not read as somebody's
        # arbitrary guess six months from now: either it is the config default
        # or it is the k that delivered a stated review load on this cell.
        "k_source": ("target_review_count"
                     if getattr(cfg, "target_review_count", None) is not None
                     else "target_flag_rate"
                     if cfg.target_flag_rate is not None else "fixed"),
        "target_flag_rate": (float(cfg.target_flag_rate)
                             if cfg.target_flag_rate is not None else None),
        # The standard this band was cut to, and the k a NORMAL book would
        # have needed for the same promise. Stored together because the pair
        # is the answer to "why isn't it 3?" a year from now, when nobody
        # remembers what the tail looked like.
        "coverage_pct": (round(100.0 - float(cfg.target_flag_rate), 10)
                         if cfg.target_flag_rate is not None else None),
        "k_if_normal": (normality.k_if_normal(100.0 - cfg.target_flag_rate)
                        if cfg.target_flag_rate is not None else None),
        "target_review_count": (float(cfg.target_review_count)
                                if getattr(cfg, "target_review_count", None)
                                is not None else None),
        # What the budget actually cost on THIS cell: the rate it implied, how
        # many fit-book orders sit beyond the bound, and whether the floor had
        # to catch it. Without this a k of 5.29 is unreproducible.
        "review_budget": budget,
        "n": int(est["n"]),
        # the pair that scores
        "centre": _f(est[f"centre_{e}"]), "scale": _f(est[f"scale_{e}"]),
        "lo": _f(est[f"lo_{e}"]), "hi": _f(est[f"hi_{e}"]),
        # both, always, so switching estimator needs no refit
        "centre_classical": _f(est["centre_classical"]),
        "scale_classical": _f(est["scale_classical"]),
        "lo_classical": _f(est["lo_classical"]),
        "hi_classical": _f(est["hi_classical"]),
        "centre_robust": _f(est["centre_robust"]),
        "scale_robust": _f(est["scale_robust"]),
        "lo_robust": _f(est["lo_robust"]),
        "hi_robust": _f(est["hi_robust"]),
        "scoring_config": {f: getattr(cfg, f) for f in SCORING_FIELDS},
        "reference": _reference(df, cfg, est, flag_rate_pct),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load(path: str, base_cfg):
    """Read a frozen band back. Returns (band, cfg, reference)."""
    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)

    if p.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written by format version {p.get('format_version')}, "
            f"this code reads format version {FORMAT_VERSION}. Refit rather "
            f"than guess.")

    cfg = dataclasses.replace(base_cfg, **p["scoring_config"])
    return p, cfg, p.get("reference", {})


def drift_report(df: pd.DataFrame, scored: pd.DataFrame, reference: dict,
                 cfg) -> tuple[pd.DataFrame, list[str]]:
    """Has the new book moved away from the one the band was fitted on?

    A frozen band decays silently. This separates the two reasons a flag rate
    can move:

      the market changed   -> feature medians shifted (wider spreads, bigger
                              orders, higher volatility). Recalibrate.
      execution changed    -> features look the same but the rate moved.
                              That is a real finding, act on it.
    """
    rows, warnings = [], []

    fitted = reference.get("flag_rate_pct", float("nan"))
    realized = 100.0 * scored["flagged"].mean() if len(scored) else float("nan")
    rows.append({"check": "flag rate %", "fit_book": round(fitted, 2),
                 "new_book": round(realized, 2),
                 "change_pct": (round(100.0 * (realized - fitted) / fitted, 1)
                                if fitted else float("nan"))})

    for col, fit_med in (reference.get("feature_medians") or {}).items():
        if col not in df.columns:
            continue
        new_med = float(pd.to_numeric(df[col], errors="coerce").median())
        pct = (100.0 * (new_med - fit_med) / fit_med) if fit_med else float("nan")
        rows.append({"check": f"median {col}", "fit_book": round(fit_med, 3),
                     "new_book": round(new_med, 3), "change_pct": round(pct, 1)})
        if np.isfinite(pct) and abs(pct) > DRIFT_MEDIAN_PCT:
            warnings.append(
                f"{col} median moved {pct:+.0f}% against the fit book -- the "
                f"band was not fitted on orders like these. Consider refitting.")

    if np.isfinite(realized) and np.isfinite(fitted) and fitted > 0:
        if realized > 2.5 * fitted:
            warnings.append(
                f"Flag rate {realized:.2f}% is well above the {fitted:.2f}% the "
                f"band produced on its own fit book. Either execution genuinely "
                f"degraded (a finding) or the regime moved (refit). The feature "
                f"rows above tell you which.")
        elif realized < 0.4 * fitted:
            warnings.append(
                f"Flag rate {realized:.2f}% is far below the {fitted:.2f}% on the "
                f"fit book. The band has gone slack and is no longer catching much.")

    return pd.DataFrame(rows), warnings
