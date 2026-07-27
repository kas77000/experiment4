"""Quantile regression of perf_norm on difficulty --- the conditional threshold.

The idea in one line: instead of asking "is this order outside the band for its
bucket?", fit the whole conditional distribution of performance given the
order's difficulty, and ask "is it outside the band for THIS order?".

    perf_norm ~ Q_tau( sqrt(%ADV), log(POV), log(duration), log(spread),
                       log(vol), size x urgency, algo, market, side )

Three quantiles are fitted independently -- tau_lo, tau_med, tau_hi. The median
surface is the expected cost; the outer two are the threshold. Because they are
fitted separately they can CROSS in sparse corners of feature space, which is a
known defect of independent quantile regression; we fix it by rearrangement
(sorting the predicted triple per row), the standard remedy from Chernozhukov,
Fernandez-Val & Galichon.

Two backends:
    quantreg   statsmodels QuantReg. Continuous, asymmetric, no bucket edges.
    empirical  bucketed percentiles of perf_norm, used automatically when
               statsmodels is missing or the sample is too thin to regress.
               Strictly a degradation, but it keeps the tier runnable anywhere.

Cross-fitting (`cross_fit_predict`) scores every order with a model fitted
WITHOUT it. In-sample quantile fits are optimistically tight -- a p5/p95 band
fitted and evaluated on the same rows will always look near-perfectly
calibrated. Out-of-sample coverage is the number that tells you whether the
thresholds will hold next quarter.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tca import schema
from tier3_model import features


@dataclass
class ModelFit:
    spec: features.FeatureSpec
    taus: tuple
    coefs: dict = field(default_factory=dict)      # tau -> coefficient vector
    backend: str = "quantreg"
    n_train: int = 0
    n_trimmed: int = 0
    pseudo_r1: dict = field(default_factory=dict)  # tau -> Koenker-Machado R1
    empirical: pd.DataFrame | None = None          # fallback band table

    def coef_frame(self) -> pd.DataFrame:
        """Coefficients as a tidy table -- this is the artefact you hand over."""
        if self.backend != "quantreg":
            return pd.DataFrame()
        return pd.DataFrame(
            {f"tau_{t:g}": self.coefs[t] for t in self.taus},
            index=self.spec.names,
        ).round(4)


# --------------------------------------------------------------------------
# statsmodels backend
# --------------------------------------------------------------------------

def _statsmodels_available() -> bool:
    try:
        import statsmodels.api  # noqa: F401
        return True
    except Exception:
        return False


def _rho(u: np.ndarray, tau: float) -> float:
    """Quantile check loss, summed. The objective QuantReg minimizes."""
    return float(np.sum(u * (tau - (u < 0).astype(float))))


def _fit_quantreg(X: np.ndarray, y: np.ndarray, taus) -> tuple[dict, dict]:
    import statsmodels.api as sm

    coefs, r1 = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tau in taus:
            res = sm.QuantReg(y, X).fit(q=tau, max_iter=5000)
            beta = np.asarray(res.params, dtype=float)
            coefs[tau] = beta
            # Koenker-Machado pseudo-R1: 1 - (fitted loss / intercept-only loss)
            full = _rho(y - X @ beta, tau)
            null = _rho(y - np.quantile(y, tau), tau)
            r1[tau] = float(1.0 - full / null) if null > 0 else 0.0
    return coefs, r1


# --------------------------------------------------------------------------
# empirical fallback backend
# --------------------------------------------------------------------------

_EMP_KEYS = [schema.ALGO, schema.ADV_BUCKET]


def _fit_empirical(df: pd.DataFrame, taus, min_n: int) -> pd.DataFrame:
    """Bucketed percentiles of perf_norm, with a pooled row for thin groups."""
    pct = [100 * t for t in taus]
    rows = []
    for keys, g in df.groupby(_EMP_KEYS, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        q = np.nanpercentile(g[schema.PERF_NORM].to_numpy(), pct)
        rows.append({**dict(zip(_EMP_KEYS, keys)), "n": len(g),
                     **{f"q{i}": float(v) for i, v in enumerate(q)}})
    q = np.nanpercentile(df[schema.PERF_NORM].to_numpy(), pct)
    rows.append({k: None for k in _EMP_KEYS} | {"n": len(df),
                 **{f"q{i}": float(v) for i, v in enumerate(q)}})
    out = pd.DataFrame(rows)
    out["trusted"] = out["n"] >= min_n
    return out


def _predict_empirical(table: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    trusted = table[table["trusted"]]
    lookup = {(r[schema.ALGO], r[schema.ADV_BUCKET]): r
              for _, r in trusted.iterrows()
              if r[schema.ALGO] is not None}
    pooled = table[table[schema.ALGO].isna()].iloc[-1]

    qcols = [c for c in table.columns if c.startswith("q") and c != "q"]
    out = np.empty((len(df), len(qcols)))
    for i, (_, r) in enumerate(df.iterrows()):
        band = lookup.get((r[schema.ALGO], r[schema.ADV_BUCKET]), pooled)
        out[i] = [band[c] for c in qcols]
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def fit(df: pd.DataFrame, cfg) -> ModelFit:
    """Fit the conditional quantile surfaces on `df`."""
    taus = (cfg.tau_lo, cfg.tau_med, cfg.tau_hi)

    y_all = df[schema.PERF_NORM].to_numpy(dtype=float)
    keep = np.isfinite(y_all)

    # Clamp the trim to stay well inside the fitted taus. Trimming at or near
    # tau_lo would remove the very mass that defines the lower band, biasing it
    # inwards -- the failure mode is silent except in the calibration table.
    q = float(cfg.fit_trim_quantile)
    q = min(q, min(cfg.tau_lo, 1.0 - cfg.tau_hi) / 5.0)
    if q > 0 and keep.sum() > 0:
        lo, hi = np.quantile(y_all[keep], [q, 1.0 - q])
        keep &= (y_all >= lo) & (y_all <= hi)
    train = df.loc[keep]
    n_trimmed = int(len(df) - keep.sum())

    want_reg = cfg.backend in ("auto", "quantreg")
    can_reg = _statsmodels_available() and len(train) >= cfg.min_fit_n
    if cfg.backend == "quantreg" and not _statsmodels_available():
        raise ImportError(
            "backend='quantreg' requires statsmodels. `pip install statsmodels`, "
            "or set backend='auto' to fall back to empirical bands.")

    spec = features.fit_spec(train, cfg)

    if want_reg and can_reg:
        X = features.design(train, spec, cfg)
        y = train[schema.PERF_NORM].to_numpy(dtype=float)
        coefs, r1 = _fit_quantreg(X, y, taus)
        return ModelFit(spec=spec, taus=taus, coefs=coefs, backend="quantreg",
                        n_train=len(train), n_trimmed=n_trimmed, pseudo_r1=r1)

    emp = _fit_empirical(train, taus, min_n=200)
    return ModelFit(spec=spec, taus=taus, backend="empirical",
                    n_train=len(train), n_trimmed=n_trimmed, empirical=emp)


def predict(model: ModelFit, df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Predicted (q_lo, q_med, q_hi) of perf_norm for each row, rearranged."""
    if model.backend == "quantreg":
        X = features.design(df, model.spec, cfg)
        preds = np.column_stack([X @ model.coefs[t] for t in model.taus])
    else:
        preds = _predict_empirical(model.empirical, df)

    # Rearrangement: independently fitted quantiles can cross in sparse regions.
    # Sorting each row restores monotonicity without refitting.
    preds = np.sort(preds, axis=1)
    return pd.DataFrame(preds, index=df.index, columns=["q_lo", "q_med", "q_hi"])


def cross_fit_predict(df: pd.DataFrame, cfg, seed: int = 0):
    """K-fold out-of-sample predictions plus a full-data model for future orders.

    Returns (preds, full_model). Every row in `preds` comes from a model that
    never saw that row, so the coverage numbers downstream are honest.
    """
    full_model = fit(df, cfg)

    if cfg.n_folds is None or cfg.n_folds < 2:
        return predict(full_model, df, cfg), full_model

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, cfg.n_folds, size=len(df))
    preds = pd.DataFrame(np.nan, index=df.index, columns=["q_lo", "q_med", "q_hi"])

    for k in range(cfg.n_folds):
        test_mask = fold == k
        train = df.loc[~test_mask]
        test = df.loc[test_mask]
        if not len(test):
            continue
        if len(train) < cfg.min_fit_n:
            preds.loc[test.index] = predict(full_model, test, cfg).to_numpy()
            continue
        m = fit(train, cfg)
        preds.loc[test.index] = predict(m, test, cfg).to_numpy()

    return preds, full_model


def coverage_check(df: pd.DataFrame, preds: pd.DataFrame, cfg) -> pd.DataFrame:
    """Did the fitted band actually contain what it promised?

    This is the calibration test, and on real data it is the ONLY honest check
    you have -- you never learn which orders were "really" bad, but you can
    always verify that a p5 threshold is exceeded about 5% of the time
    out-of-sample. Realized far above nominal means the model is missing a
    driver; far below means it is overfitting and will never flag anything.
    """
    y = df[schema.PERF_NORM].to_numpy(dtype=float)
    ok = np.isfinite(y) & preds["q_lo"].notna().to_numpy()
    y, p = y[ok], preds.loc[ok]

    rows = [
        {"bound": f"below q_lo (tau={cfg.tau_lo:g})",
         "nominal_pct": 100 * cfg.tau_lo,
         "realized_pct": round(100 * float(np.mean(y < p["q_lo"])), 2)},
        {"bound": f"above q_hi (tau={cfg.tau_hi:g})",
         "nominal_pct": 100 * (1 - cfg.tau_hi),
         "realized_pct": round(100 * float(np.mean(y > p["q_hi"])), 2)},
        {"bound": f"below q_med (tau={cfg.tau_med:g})",
         "nominal_pct": 100 * cfg.tau_med,
         "realized_pct": round(100 * float(np.mean(y < p["q_med"])), 2)},
    ]
    return pd.DataFrame(rows)
