"""Freeze a fitted threshold to disk, and load it back to score future orders.

This is what makes the whole exercise worth doing. Fitting and scoring the same
book tells you almost nothing -- 1.5% flags because 1.5% was *defined* as
flagged. The value comes from freezing the fitted surface and applying it,
unchanged, to orders it has never seen. Then the flag rate becomes a
measurement rather than a definition: if next quarter flags 4% against a gate
set at 1.5%, something real changed.

Everything needed to rebuild the design matrix identically goes in the file --
not just the coefficients. The standardization means and standard deviations,
the imputation medians and the dummy levels are all part of the model. Saving
coefficients alone would silently rescale every feature the next time you ran
it, and nothing would error.

Also stored is a reference snapshot of the training book, so `drift_report`
can tell you whether a change in the flag rate is your execution or the market.
"""

from __future__ import annotations
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from tca import schema
from tier3_model import cost_model, features

FORMAT_VERSION = 1

# Config fields that affect scoring and must travel with the model. Anything not
# listed falls back to the current default on load.
SCORING_FIELDS = ["tau_lo", "tau_med", "tau_hi", "escalate_z",
                  "min_notional_review", "algo_effect", "include_interactions"]


def _reference(df: pd.DataFrame, scored: pd.DataFrame, cfg) -> dict:
    """Snapshot of the training book, for later drift comparison."""
    ref = {
        "n_orders": int(len(df)),
        "flag_rate_pct": float(100 * scored["flagged"].mean()),
        "nominal_flag_rate_pct": float(100 * (cfg.tau_lo + (1 - cfg.tau_hi))),
        "perf_norm_median": float(df[schema.PERF_NORM].median()),
        "perf_norm_sd": float(df[schema.PERF_NORM].std()),
        "feature_medians": {},
        "algo_levels": [],
        "market_levels": [],
    }
    for col in schema.DIFFICULTY:
        if col in df.columns:
            ref["feature_medians"][col] = float(pd.to_numeric(
                df[col], errors="coerce").median())
    if schema.ALGO in df.columns:
        ref["algo_levels"] = sorted(df[schema.ALGO].dropna().astype(str).unique())
    if schema.MARKET in df.columns:
        ref["market_levels"] = sorted(df[schema.MARKET].dropna().astype(str).unique())
    return ref


def save(model: cost_model.ModelFit, cfg, path: str,
         df: pd.DataFrame | None = None,
         scored: pd.DataFrame | None = None) -> str:
    """Write the fitted threshold to JSON."""
    spec = model.spec
    payload = {
        "format_version": FORMAT_VERSION,
        "fitted_at": dt.datetime.now().isoformat(timespec="seconds"),
        "backend": model.backend,
        "taus": list(model.taus),
        "n_train": model.n_train,
        "pseudo_r1": {str(k): v for k, v in model.pseudo_r1.items()},
        "feature_names": spec.names,
        "coefficients": {str(t): list(map(float, model.coefs[t]))
                         for t in model.taus} if model.backend == "quantreg" else {},
        "spec": {
            "numeric": list(spec.numeric),
            "means": {k: float(v) for k, v in spec.means.items()},
            "stds": {k: float(v) for k, v in spec.stds.items()},
            "medians": {k: float(v) for k, v in spec.medians.items()},
            "algo_levels": list(spec.algo_levels),
            "market_levels": list(spec.market_levels),
            "has_side": bool(spec.has_side),
        },
        "empirical": (model.empirical.to_dict("records")
                      if model.empirical is not None else None),
        "scoring_config": {f: getattr(cfg, f) for f in SCORING_FIELDS},
    }
    if df is not None and scored is not None:
        payload["training_reference"] = _reference(df, scored, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load(path: str, base_cfg):
    """Read a frozen threshold back. Returns (ModelFit, cfg, reference)."""
    import dataclasses

    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)

    if p.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written by format version {p.get('format_version')}, "
            f"this code reads version {FORMAT_VERSION}. Refit rather than guess.")

    s = p["spec"]
    spec = features.FeatureSpec(
        numeric=s["numeric"], means=s["means"], stds=s["stds"],
        medians=s["medians"], algo_levels=s["algo_levels"],
        market_levels=s["market_levels"], has_side=s["has_side"])

    taus = tuple(p["taus"])
    model = cost_model.ModelFit(
        spec=spec, taus=taus,
        coefs={t: np.array(p["coefficients"][str(t)], dtype=float)
               for t in taus} if p["backend"] == "quantreg" else {},
        backend=p["backend"],
        n_train=p.get("n_train", 0),
        pseudo_r1={float(k): v for k, v in p.get("pseudo_r1", {}).items()},
        empirical=(pd.DataFrame(p["empirical"])
                   if p.get("empirical") is not None else None),
    )

    # Rebuild the scoring config, so fields added since the model was saved
    # take today's defaults rather than exploding.
    cfg = dataclasses.replace(base_cfg, **p["scoring_config"])
    return model, cfg, p.get("training_reference", {})


def drift_report(df: pd.DataFrame, scored: pd.DataFrame, reference: dict,
                 cfg) -> tuple[pd.DataFrame, list[str]]:
    """Has the new book moved away from the one the threshold was fitted on?

    A frozen threshold decays silently. This separates the two reasons the flag
    rate can move:

      the market changed   -> feature medians shifted (bigger orders, wider
                              spreads, higher volatility). Recalibrate.
      execution changed    -> features look the same but the flag rate moved.
                              That is a real finding, act on it.

    Unseen algo or market values are called out separately: they fall into the
    model's baseline category and get scored against the wrong intercept, which
    produces plausible-looking numbers with no warning.
    """
    rows, warnings = [], []

    nominal = reference.get("nominal_flag_rate_pct",
                            100 * (cfg.tau_lo + (1 - cfg.tau_hi)))
    realized = 100 * scored["flagged"].mean()
    rows.append({"check": "flag rate %", "training": round(
        reference.get("flag_rate_pct", float("nan")), 2),
        "new_data": round(realized, 2), "nominal": round(nominal, 2)})

    for col, train_med in (reference.get("feature_medians") or {}).items():
        if col not in df.columns:
            continue
        new_med = float(pd.to_numeric(df[col], errors="coerce").median())
        pct = (100 * (new_med - train_med) / train_med) if train_med else float("nan")
        rows.append({"check": f"median {col}", "training": round(train_med, 3),
                     "new_data": round(new_med, 3), "change_pct": round(pct, 1)})
        if np.isfinite(pct) and abs(pct) > 25:
            warnings.append(
                f"{col} median moved {pct:+.0f}% vs the training book -- the "
                f"threshold was not fitted on orders like these.")

    if realized > 2.5 * nominal:
        warnings.append(
            f"Flag rate {realized:.1f}% is well above the {nominal:.1f}% the gate "
            f"was set to. Either execution genuinely degraded (a finding) or the "
            f"regime moved (refit). The feature rows above tell you which.")
    elif realized < 0.4 * nominal:
        warnings.append(
            f"Flag rate {realized:.1f}% is far below the {nominal:.1f}% nominal. "
            f"The threshold has gone slack and is no longer catching much.")

    for col, key in [(schema.ALGO, "algo_levels"), (schema.MARKET, "market_levels")]:
        known = set(reference.get(key) or [])
        if not known or col not in df.columns:
            continue
        unseen = sorted(set(df[col].dropna().astype(str)) - known)
        if unseen:
            n = int(df[col].astype(str).isin(unseen).sum())
            warnings.append(
                f"{n:,} orders have a {col} not present when the model was fitted "
                f"({unseen[:5]}). They are scored against the baseline category, "
                f"so their thresholds are unreliable -- refit to include them.")

    return pd.DataFrame(rows), warnings
