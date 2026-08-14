# Tier 5 Gaussian 3-Sigma Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `tier5_gaussian/` tier that fits a normal distribution to pVWAP slippage and sets the acceptable range at `mu +/- 3*sigma`, together with the evidence needed to say whether that 3-sigma promise (0.27% flagged) actually holds; then remove Tiers 1, 2 and 4, keeping `tier3_model`.

**Architecture:** A self-contained tier folder in the same shape as the existing ones. `config.py` holds the knobs, `band.py` fits and applies the band, `normality.py` produces the evidence, `run.py` is the CLI driver. It consumes the shared `tca.dataset.load_prepared`, so it scores the identical rows as `tier3_model`. Both estimators (classical mean/sd and robust median/MAD) are always computed so they can be read side by side.

**Tech Stack:** Python 3, pandas >= 2.0, numpy >= 1.24, scipy >= 1.10 (already a hard dependency), matplotlib >= 3.7 (optional, plot only).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-14-gaussian-3sigma-threshold-design.md`. Read it before starting.
- **No test suite exists in this repository and this plan does not introduce one.** Verification is by running the tier's own `--self-check` and by running the drivers end to end. Every task states the exact command and the exact expected output.
- **`k` default is `3.0`.** Configurable, never hardcoded outside `config.py`.
- **MAD scale factor is `1.4826`** (`1 / Phi^-1(0.75)`), which makes the scaled MAD a consistent estimator of sigma under normality.
- **Default metric is `schema.SLIPPAGE_BPS`** (raw pVWAP bps). `perf_in_spreads` and `perf_norm` selectable.
- **Default scoring level is `ALL`** — one range for the year.
- **`min_group_n = 200`.** Groups below it are marked `trusted = False`, never dropped.
- **Degradation is reported, never silent.** Missing `scipy` or `matplotlib` prints a line saying what was skipped; neither is in the scoring path.
- **Folder names are not reflowed.** `tier3_model/` keeps its name and its `outputs/tier3/` paths.
- **The slippage sign is already normalized by `tca.pipeline`** before this tier sees it. Negative always means underperformance. No sign handling belongs in Tier 5.
- **Commit after every task.** Use the repo's existing commit trailer:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0151cYTCYA27xD4GkcWCMmGi
  ```

## File Structure

| file | responsibility |
|---|---|
| `tier5_gaussian/__init__.py` | package marker (empty) |
| `tier5_gaussian/config.py` | `Tier5Config` dataclass, level constants, `LEVEL_KEYS`, `ESTIMATORS` |
| `tier5_gaussian/band.py` | fit both estimators per group, classify orders, score a frame. Knows nothing about normality testing |
| `tier5_gaussian/normality.py` | coverage table, required-k, shape statistics, QQ plot. Knows nothing about scoring |
| `tier5_gaussian/run.py` | CLI driver, report layout, `--self-check` |
| `tier5_gaussian/README.md` | method, settings, known weaknesses |
| `run.py` (root) | rewritten: compares `tier3_model` vs `tier5_gaussian` |
| `README.md` (root) | rewritten: two methods instead of four |
| `requirements.txt` | comments updated |

**Deleted in Task 6:** `tier1_fixed/`, `tier2_percentile/`, `tier4_vwap/`, `synthetic_vwap.py`.

Tasks 1-5 only add files, so the repository stays fully working at every commit. Task 6 does the deletion and the `run.py` rewrite together, because deleting the tiers breaks `run.py`'s imports and the two must land in one commit.

---

### Task 1: Package skeleton and config

**Files:**
- Create: `tier5_gaussian/__init__.py`
- Create: `tier5_gaussian/config.py`

**Interfaces:**
- Consumes: `tca.schema` (`ALGO`, `ADV_BUCKET`, `SLIPPAGE_BPS`)
- Produces: `Tier5Config` dataclass; `CONFIG` instance; constants `LEVEL_ALL`, `LEVEL_ALGO`, `LEVEL_ADV`, `LEVEL_ALGO_ADV`; dicts `LEVEL_KEYS`; tuple `ESTIMATORS`

- [ ] **Step 1: Create the package marker**

Create `tier5_gaussian/__init__.py` containing exactly one line:

```python
"""Tier 5 --- Gaussian mu +/- k*sigma threshold."""
```

- [ ] **Step 2: Write the config module**

Create `tier5_gaussian/config.py`:

```python
"""Tier 5 knobs --- a Gaussian band at mu +/- k*sigma.

This is the statistical-process-control control limit: assume the metric is
normally distributed, estimate its centre and scale, and accept everything
within k scales of the centre.

k = 3 is not an arbitrary choice. Under a normal distribution 99.73% of
observations fall within +/- 3 sigma, so picking k is picking a flag rate:
0.27% of orders, about 1 in 370. Whether the book delivers that is measured
rather than assumed -- see normality.py.
"""

from dataclasses import dataclass

from tca import schema

# --- fitted grouping levels, most general first --------------------------
LEVEL_ALL = "ALL"
LEVEL_ALGO = "algo"
LEVEL_ADV = "adv_bucket"
LEVEL_ALGO_ADV = "algo_x_adv_bucket"

# level -> the columns that define a group at that level. The empty tuple is
# the whole book, which is the headline number.
LEVEL_KEYS = {
    LEVEL_ALL: (),
    LEVEL_ALGO: (schema.ALGO,),
    LEVEL_ADV: (schema.ADV_BUCKET,),
    LEVEL_ALGO_ADV: (schema.ALGO, schema.ADV_BUCKET),
}

# classical -> mean / standard deviation   (the method as requested)
# robust    -> median / 1.4826 * MAD       (the same band, tail-resistant)
ESTIMATORS = ("classical", "robust")


@dataclass(frozen=True)
class Tier5Config:
    # --- the band ---------------------------------------------------------
    # How many scales either side of the centre. 3.0 promises 0.27% flagged
    # IF the data is normal, which is the assumption the report tests.
    k_sigma: float = 3.0

    # --- which metric to band --------------------------------------------
    #   SLIPPAGE_BPS    -> raw pVWAP slippage, in bps       (the headline)
    #   PERF_IN_SPREADS -> slippage / spread
    #   PERF_NORM       -> slippage / sigma_expected
    # Raw bps is the default because "the range of performance" means a bps
    # figure to the people who asked for it.
    metric: str = schema.SLIPPAGE_BPS

    # Which estimator SCORES orders. Both are always computed and reported;
    # this only picks the one the zones are cut on.
    estimator: str = "classical"

    # Which fitted level supplies each order's band. LEVEL_ALL is the request
    # taken literally: one range for the whole year. Anything else falls back
    # to LEVEL_ALL when the matched cell is untrusted.
    score_level: str = LEVEL_ALL

    # --- robustness -------------------------------------------------------
    # Minimum orders before a group's sigma is trusted. A sigma from 44 orders
    # is not a threshold. Thin groups stay in the table marked trusted=False,
    # so they are visible rather than silently absent.
    min_group_n: int = 200

    # Levels to fit. Every level lands in band_table.csv.
    group_levels: tuple = (LEVEL_ALL, LEVEL_ALGO, LEVEL_ADV, LEVEL_ALGO_ADV)

    # --- review queue -----------------------------------------------------
    # Materiality gate. 0 = off, so review_required == flagged.
    min_notional_review: float = 0.0

    # Write outputs/tier5/qq_plot.png when matplotlib is available.
    make_qq_plot: bool = True


CONFIG = Tier5Config()
```

- [ ] **Step 3: Verify it imports and the defaults are right**

Run:

```bash
python -c "from tier5_gaussian import config as c; print(c.CONFIG); print(c.LEVEL_KEYS)"
```

Expected: prints the dataclass with `k_sigma=3.0`, `metric='slippage_bps'`, `estimator='classical'`, `score_level='ALL'`, `min_group_n=200`, then the `LEVEL_KEYS` dict with four entries.

- [ ] **Step 4: Commit**

```bash
git add tier5_gaussian/__init__.py tier5_gaussian/config.py
git commit -m "Add tier5_gaussian package skeleton and config"
```

---

### Task 2: The band --- fit and score

**Files:**
- Create: `tier5_gaussian/band.py`

**Interfaces:**
- Consumes: `tier5_gaussian.config` (`LEVEL_KEYS`, `LEVEL_ALL`, `ESTIMATORS`, `Tier5Config`), `tca.schema`
- Produces:
  - `MAD_TO_SIGMA: float = 1.4826`
  - `IN_RANGE`, `OUT_LOW`, `OUT_HIGH`, `NO_BAND`, `FLAGGED` (set)
  - `estimates(x: np.ndarray, k: float) -> dict` — the 13 keys listed in Step 1
  - `classify(x: float, lo: float, hi: float) -> str`
  - `fit(df: pd.DataFrame, cfg) -> pd.DataFrame` — the band table
  - `BandModel(table: pd.DataFrame, cfg)` with `.score_frame(df) -> pd.DataFrame`
  - `flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write the module**

Create `tier5_gaussian/band.py`:

```python
"""Fit a Gaussian band per group, and score orders against it.

Two estimators are computed for every group and both are reported:

    classical:  centre = mean(x)     scale = std(x, ddof=1)
    robust:     centre = median(x)   scale = 1.4826 * MAD(x)

The 1.4826 is 1 / Phi^-1(0.75), which makes the scaled MAD a consistent
estimator of sigma *under normality*. That is the point of showing both: on
genuinely normal data they agree, so any gap between them IS the
non-normality, expressed in the band's own units. On a fat-tailed book the
classical scale is inflated by the very outliers the band exists to catch,
and the two rows say so numerically.

The band is symmetric around the centre by construction. Slippage is not
symmetric, so the two tails are not equally well served. That is a property
of the method, not a bug; normality.required_k() quantifies it per tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca import schema
from tier5_gaussian import config as t5cfg

# 1 / Phi^-1(0.75). Makes MAD a consistent estimator of sigma under normality.
MAD_TO_SIGMA = 1.4826

# Zone labels, shared vocabulary with the other tiers.
IN_RANGE = "IN_RANGE"    # inside the band -> acceptable
OUT_LOW = "OUT_LOW"      # below the band: underperformance -> flag / justify
OUT_HIGH = "OUT_HIGH"    # above the band: suspiciously good -> flag / justify
NO_BAND = "NO_BAND"      # no trusted group, or the metric is missing
FLAGGED = {OUT_LOW, OUT_HIGH}

# Column order of the band table.
BAND_COLS = [
    "level", schema.ALGO, schema.ADV_BUCKET, "n",
    "centre_classical", "scale_classical", "lo_classical", "hi_classical",
    "centre_robust", "scale_robust", "lo_robust", "hi_robust",
]

_EMPTY = {
    "n": 0,
    "centre_classical": np.nan, "scale_classical": np.nan,
    "lo_classical": np.nan, "hi_classical": np.nan,
    "centre_robust": np.nan, "scale_robust": np.nan,
    "lo_robust": np.nan, "hi_robust": np.nan,
}


def estimates(x, k: float) -> dict:
    """Both estimators and both bands for one group's metric array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return dict(_EMPTY)

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * MAD_TO_SIGMA

    return {
        "n": n,
        "centre_classical": mean, "scale_classical": sd,
        "lo_classical": mean - k * sd, "hi_classical": mean + k * sd,
        "centre_robust": med, "scale_robust": mad,
        "lo_robust": med - k * mad, "hi_robust": med + k * mad,
    }


def fit(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return a tidy band table with one row per (level, group).

    Every group appears, including groups too thin to trust -- those are
    marked trusted=False rather than dropped, so a thin cell is visible.
    """
    metric = cfg.metric
    if metric not in df.columns:
        raise ValueError(f"Tier 5 needs column {metric!r}, which is absent.")

    rows = []
    for level in cfg.group_levels:
        keys = t5cfg.LEVEL_KEYS[level]
        if not keys:
            row = {"level": level, schema.ALGO: None, schema.ADV_BUCKET: None}
            row.update(estimates(df[metric].to_numpy(), cfg.k_sigma))
            rows.append(row)
            continue
        for gk, g in df.groupby(list(keys), dropna=False, observed=False):
            gk = gk if isinstance(gk, tuple) else (gk,)
            row = {"level": level, schema.ALGO: None, schema.ADV_BUCKET: None}
            row.update(dict(zip(keys, gk)))
            row.update(estimates(g[metric].to_numpy(), cfg.k_sigma))
            rows.append(row)

    out = pd.DataFrame(rows)[BAND_COLS]
    out["trusted"] = out["n"] >= cfg.min_group_n
    return out.reset_index(drop=True)


def classify(x: float, lo: float, hi: float) -> str:
    """Map a metric value to a zone given one band's bounds."""
    if not np.isfinite(x):
        return NO_BAND
    if x < lo:
        return OUT_LOW
    if x > hi:
        return OUT_HIGH
    return IN_RANGE


class BandModel:
    """Holds the fitted table and scores orders against one chosen estimator."""

    def __init__(self, table: pd.DataFrame, cfg):
        if cfg.estimator not in t5cfg.ESTIMATORS:
            raise ValueError(f"Unknown estimator {cfg.estimator!r}; "
                             f"pick one of {list(t5cfg.ESTIMATORS)}")
        if cfg.score_level not in t5cfg.LEVEL_KEYS:
            raise ValueError(f"Unknown score_level {cfg.score_level!r}; "
                             f"pick one of {list(t5cfg.LEVEL_KEYS)}")

        self.table = table
        self.cfg = cfg
        est = cfg.estimator
        self.centre_col = f"centre_{est}"
        self.scale_col = f"scale_{est}"
        self.lo_col = f"lo_{est}"
        self.hi_col = f"hi_{est}"

        trusted = table[table["trusted"]]
        self._by_level = {}
        for level in t5cfg.LEVEL_KEYS:
            keys = t5cfg.LEVEL_KEYS[level]
            sub = trusted[trusted["level"] == level]
            self._by_level[level] = {
                tuple(r[k] for k in keys): r for _, r in sub.iterrows()
            }

    def _lookup(self, algo, adv_bucket):
        """The chosen level if it is trusted, else the global band."""
        vals = {schema.ALGO: algo, schema.ADV_BUCKET: adv_bucket}
        for level in (self.cfg.score_level, t5cfg.LEVEL_ALL):
            keys = t5cfg.LEVEL_KEYS[level]
            key = tuple(vals[k] for k in keys)
            band = self._by_level.get(level, {}).get(key)
            if band is not None:
                return band, level
        return None, None

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a prepared frame. Adds the band, the zone and rank_stat."""
        metric = self.cfg.metric
        k = self.cfg.k_sigma
        recs = []
        for _, r in df.iterrows():
            band, level = self._lookup(r.get(schema.ALGO),
                                       r.get(schema.ADV_BUCKET))
            x = float(r[metric]) if pd.notna(r[metric]) else np.nan
            if band is None or not np.isfinite(x):
                recs.append((NO_BAND, None, np.nan, np.nan,
                             np.nan, np.nan, False, np.nan))
                continue
            centre = float(band[self.centre_col])
            scale = float(band[self.scale_col])
            lo = float(band[self.lo_col])
            hi = float(band[self.hi_col])
            zone = classify(x, lo, hi)
            # 1.0 == exactly at the limit. Same convention the other tiers use,
            # which is what lets run.py hold every method to one review budget.
            denom = k * scale
            rank = abs(x - centre) / denom if denom > 0 else np.nan
            recs.append((zone, level, centre, scale, lo, hi,
                         zone in FLAGGED, rank))

        out = df.copy()
        cols = ["zone", "band_level", "band_centre", "band_scale",
                "band_lo", "band_hi", "flagged", "rank_stat"]
        if not recs:
            for c in cols:
                out[c] = pd.Series(dtype="float64")
        else:
            for c, vals in zip(cols, zip(*recs)):
                out[c] = list(vals)

        if schema.NOTIONAL in out.columns and self.cfg.min_notional_review > 0:
            out["material"] = (out[schema.NOTIONAL].fillna(0)
                               >= self.cfg.min_notional_review)
        else:
            out["material"] = True
        out["review_required"] = out["flagged"] & out["material"]
        return out


def flag_rate_by_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """Flag rate vs order difficulty.

    A global Gaussian band is one number for the whole book, so this column
    is expected to slope the way Tier 1's did: it does not adjust for
    difficulty. Scoring at a finer level flattens it.
    """
    g = scored.groupby(schema.ADV_BUCKET, dropna=False, observed=False)
    return pd.DataFrame({
        "n": g.size(),
        "flag_rate_pct": 100.0 * g["flagged"].mean(),
        "mean_slippage_bps": g[schema.SLIPPAGE_BPS].mean(),
    }).round(2)
```

- [ ] **Step 2: Verify the estimators recover known parameters**

This is the failing-test-first step: it exercises `estimates()` against a distribution whose answer is known in closed form, before anything else depends on it.

Run:

```bash
python -c "
import numpy as np
from tier5_gaussian import band
x = np.random.default_rng(11).normal(-8.7, 18.4, 200000)
e = band.estimates(x, 3.0)
print('classical centre %.3f  scale %.3f' % (e['centre_classical'], e['scale_classical']))
print('robust    centre %.3f  scale %.3f' % (e['centre_robust'], e['scale_robust']))
print('band %.2f .. %.2f' % (e['lo_classical'], e['hi_classical']))
print('outside %.4f%%' % (100*np.mean((x < e['lo_classical']) | (x > e['hi_classical']))))
"
```

Expected: both centres near `-8.7`, both scales near `18.4` (agreeing to within about 1%), band near `-63.9 .. +46.5`, outside near `0.27%`.

- [ ] **Step 3: Verify fit and scoring on the synthetic book**

Run:

```bash
python -c "
import argparse
from tca import dataset
from tier5_gaussian import band, config as t5cfg
ap = dataset.add_common_args(argparse.ArgumentParser())
df, rep = dataset.load_prepared(ap.parse_args([]), quiet=True)
cfg = t5cfg.CONFIG
t = band.fit(df, cfg)
print(t[['level','algo','adv_bucket','n','centre_classical','scale_classical','lo_classical','hi_classical','trusted']].to_string())
m = band.BandModel(t, cfg)
s = m.score_frame(df)
print()
print(s['zone'].value_counts())
print('flag rate %.2f%%' % (100*s['flagged'].mean()))
print('rank_stat max %.2f' % s['rank_stat'].max())
"
```

Expected: a table with an `ALL` row plus per-algo, per-bucket and cross rows; thin buckets show `trusted=False`; `zone` counts are dominated by `IN_RANGE`; the flag rate is well above `0.27%` (that gap is the finding, not a bug); `rank_stat` max is above `1.0`.

- [ ] **Step 4: Commit**

```bash
git add tier5_gaussian/band.py
git commit -m "Add Tier 5 Gaussian band: fit both estimators, score orders"
```

---

### Task 3: The evidence --- normality diagnostics

**Files:**
- Create: `tier5_gaussian/normality.py`

**Interfaces:**
- Consumes: `tier5_gaussian.band` (`estimates`), `tier5_gaussian.config`, `tca.schema`
- Produces:
  - `NORMAL_K: tuple = (1.0, 2.0, 3.0, 4.0)`
  - `promised_inside(k: float) -> float`
  - `coverage_table(x, centre, scale, ks=NORMAL_K) -> pd.DataFrame`
  - `required_k(x, centre, scale, target_outside=0.0027) -> dict` with keys `k_symmetric`, `k_lo`, `k_hi`
  - `shape_stats(x) -> dict` with keys `n`, `skew`, `excess_kurtosis`, `dagostino_k2`, `p_value`, `test_note`
  - `evidence(df, cfg) -> pd.DataFrame` — one row per (level, group) for `ALL` and `algo`
  - `qq_plot(x, path, title) -> str` — the message to print

- [ ] **Step 1: Write the module**

Create `tier5_gaussian/normality.py`:

```python
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
```

- [ ] **Step 2: Verify the diagnostics agree with theory on normal data**

Run:

```bash
python -c "
import numpy as np
from tier5_gaussian import normality as nm
x = np.random.default_rng(11).normal(-8.7, 18.4, 200000)
print(nm.coverage_table(x, x.mean(), x.std(ddof=1)).round(4).to_string())
print(nm.required_k(x, x.mean(), x.std(ddof=1)))
print({k: round(v, 4) for k, v in nm.shape_stats(x).items() if k != 'test_note'})
"
```

Expected: `actual_inside_pct` tracks `promised_inside_pct` at every k (within a few hundredths), every `ratio` is near `1.0`, all three `required_k` values are near `3.0`, `skew` and `excess_kurtosis` are near `0`.

- [ ] **Step 3: Verify the diagnostics detect non-normality on the real book**

Run:

```bash
python -c "
import argparse
from tca import dataset, schema
from tier5_gaussian import config as t5cfg, normality as nm
ap = dataset.add_common_args(argparse.ArgumentParser())
df, rep = dataset.load_prepared(ap.parse_args([]), quiet=True)
x = df[schema.SLIPPAGE_BPS].to_numpy()
c, s = x.mean(), x.std(ddof=1)
print(nm.coverage_table(x, c, s).round(3).to_string())
print(nm.required_k(x, c, s))
print(nm.evidence(df, t5cfg.CONFIG).round(3).to_string())
"
```

Expected: `ratio` at `k=3` is clearly above `1.0`, `required_k` is above `3.0`, `skew` is non-zero and `excess_kurtosis` is positive. The `evidence` frame has one `ALL` row plus one row per algo.

- [ ] **Step 4: Verify the QQ plot writes**

Run:

```bash
python -c "
import argparse
from tca import dataset, schema
from tier5_gaussian import normality as nm
ap = dataset.add_common_args(argparse.ArgumentParser())
df, rep = dataset.load_prepared(ap.parse_args([]), quiet=True)
print(nm.qq_plot(df[schema.SLIPPAGE_BPS].to_numpy(), dataset.out_path('tier5', 'qq_probe.png'), 'probe'))
"
```

Expected: prints `Wrote .../outputs/tier5/qq_probe.png`. Delete the probe afterwards: `rm outputs/tier5/qq_probe.png`.

- [ ] **Step 5: Commit**

```bash
git add tier5_gaussian/normality.py
git commit -m "Add Tier 5 normality evidence: coverage, required-k, shape, QQ plot"
```

---

### Task 4: The CLI driver

**Files:**
- Create: `tier5_gaussian/run.py`

**Interfaces:**
- Consumes: `tier5_gaussian.band`, `tier5_gaussian.normality`, `tier5_gaussian.config`, `tca.dataset`, `tca.evaluate`, `tca.report`, `tca.schema`
- Produces: `self_check() -> int`; `main()`; the four files under `outputs/tier5/`

- [ ] **Step 1: Write the driver**

Create `tier5_gaussian/run.py`:

```python
"""Tier 5 driver:  python -m tier5_gaussian.run  [--csv x.csv]

    --metric        slippage_bps (default) | perf_in_spreads | perf_norm
    --k             scales either side of the centre (default 3.0)
    --estimator     classical (default) | robust
    --score-level   ALL (default) | algo | adv_bucket | algo_x_adv_bucket
    --self-check    prove the implementation on data with a known answer

Outputs: outputs/tier5/{band_table,scored_orders,normality}.csv, qq_plot.png
"""

from __future__ import annotations
import argparse
import dataclasses

import numpy as np

from tca import dataset, evaluate, report, schema
from tier5_gaussian import band, config as t5cfg, normality


def self_check(n: int = 200_000, mu: float = -8.7, sd: float = 18.4,
               k: float = 3.0, seed: int = 11) -> int:
    """Run the method on data whose answer is known in closed form.

    On genuinely normal data the estimators must recover the parameters, the
    band must match the closed form, the delivered flag rate must be 0.27%,
    and the classical and robust scales must agree. If all four hold, any
    deviation on a real book is the DATA and not a bug -- which is the claim
    the whole report rests on.
    """
    x = np.random.default_rng(seed).normal(mu, sd, n)
    e = band.estimates(x, k)
    lo, hi = e["lo_classical"], e["hi_classical"]
    outside = float(np.mean((x < lo) | (x > hi)))
    theory = 1.0 - normality.promised_inside(k)
    ratio = e["scale_robust"] / e["scale_classical"]

    checks = [
        ("recovered mean", e["centre_classical"], mu, 0.20,
         f"{e['centre_classical']:.3f}  (true {mu})"),
        ("recovered sd", e["scale_classical"], sd, 0.20,
         f"{e['scale_classical']:.3f}  (true {sd})"),
        ("band lo", lo, mu - k * sd, 0.80, f"{lo:.2f}  (closed form {mu - k*sd:.2f})"),
        ("band hi", hi, mu + k * sd, 0.80, f"{hi:.2f}  (closed form {mu + k*sd:.2f})"),
        ("flag rate", outside, theory, 0.0006,
         f"{100*outside:.3f}%  (theory {100*theory:.3f}%)"),
        ("robust/classical", ratio, 1.0, 0.02, f"{ratio:.3f}  (1.000 on normal data)"),
    ]

    print(f"  drew {n:,} samples from N({mu}, {sd}), k = {k}")
    ok = True
    for name, got, want, tol, shown in checks:
        passed = abs(got - want) <= tol
        ok = ok and passed
        print(f"  {name:<18} {shown:<40} {'OK' if passed else 'FAIL'}")
    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SELF-CHECK FAILED'}")
    return 0 if ok else 1


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM],
                    help="Override the banded metric from tier5_gaussian/config.py.")
    ap.add_argument("--k", type=float, help="Scales either side of the centre.")
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS),
                    help="Which estimator cuts the zones.")
    ap.add_argument("--score-level", choices=list(t5cfg.LEVEL_KEYS),
                    help="Which fitted level supplies each order's band.")
    ap.add_argument("--self-check", action="store_true",
                    help="Prove the implementation on data with a known answer.")
    args = ap.parse_args()

    if args.self_check:
        print(report.header("TIER 5 --- SELF-CHECK ON NORMAL DATA"))
        raise SystemExit(self_check())

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.metric:
        overrides["metric"] = args.metric
    if args.k is not None:
        overrides["k_sigma"] = args.k
    if args.estimator:
        overrides["estimator"] = args.estimator
    if args.score_level:
        overrides["score_level"] = args.score_level
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 5 --- GAUSSIAN mu +/- k*sigma BAND"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    print(f"\n=== Band ===\n  metric={cfg.metric}  k={cfg.k_sigma:g}"
          f"  estimator={cfg.estimator}  score_level={cfg.score_level}"
          f"  min_group_n={cfg.min_group_n}")

    table = band.fit(df, cfg)
    headline = table[table["level"] == t5cfg.LEVEL_ALL].iloc[0]
    c = headline[f"centre_{cfg.estimator}"]
    s = headline[f"scale_{cfg.estimator}"]
    lo = headline[f"lo_{cfg.estimator}"]
    hi = headline[f"hi_{cfg.estimator}"]
    print(f"\n  All orders (n = {int(headline['n']):,})")
    print(f"    centre  {c:>9.2f}")
    print(f"    scale   {s:>9.2f}")
    print(f"    RANGE   {lo:>9.2f} .. {hi:.2f}")

    print("\n=== Band table ===")
    show = table.copy()
    for col in show.columns:
        if show[col].dtype.kind == "f":
            show[col] = show[col].round(2)
    print(report.frame(show, max_rows=40))

    n_untrusted = int((~table["trusted"]).sum())
    if n_untrusted:
        print(f"\n  {n_untrusted} group(s) below min_group_n={cfg.min_group_n}"
              f" -> a sigma from that many orders is not a threshold. Those"
              f" orders fall back to the global band.")

    model = band.BandModel(table, cfg)
    scored = model.score_frame(df)

    print("\n=== Zone distribution ===")
    print(report.zone_summary(scored))

    print("\n=== Flag rate vs order difficulty ===")
    print(report.frame(band.flag_rate_by_bucket(scored)))
    if cfg.score_level == t5cfg.LEVEL_ALL:
        print("\n  One band for the whole book, so this column is expected to")
        print("  slope: it does not adjust for difficulty. --score-level"
              " algo_x_adv_bucket flattens it.")

    # ---------------- the evidence ----------------
    x = df[cfg.metric].to_numpy()
    print(report.header("DOES k*sigma MEAN WHAT IT SAYS ON THIS BOOK?"))
    cov = normality.coverage_table(x, c, s)
    print(report.frame(cov.round(3)))

    req = normality.required_k(x, c, s)
    print(f"\n  To actually flag {100*normality.NOMINAL_OUTSIDE:.2f}% of this book"
          f" you need k = {req['k_symmetric']:.2f}, not {cfg.k_sigma:g}.")
    print(f"  Per tail: k_lo = {req['k_lo']:.2f}, k_hi = {req['k_hi']:.2f}."
          f"  A symmetric band cannot serve both when these differ.")

    st = normality.shape_stats(x)
    print(f"\n  skew             {st['skew']:>8.3f}   (0 if normal)")
    print(f"  excess kurtosis  {st['excess_kurtosis']:>8.3f}   (0 if normal)")
    if np.isfinite(st["p_value"]):
        print(f"  D'Agostino K2    {st['dagostino_k2']:>8.1f}   p = {st['p_value']:.3g}")
        print("\n  Treat that p-value with suspicion: at this sample size EVERY")
        print("  formal normality test rejects, because it tests 'exactly normal'")
        print("  and nothing real is. The coverage table above is the evidence.")
    elif st["test_note"]:
        print(f"  {st['test_note']}")

    print("\n=== Classical vs robust ===")
    est_rows = table[table["level"] == t5cfg.LEVEL_ALL][[
        "centre_classical", "scale_classical", "lo_classical", "hi_classical",
        "centre_robust", "scale_robust", "lo_robust", "hi_robust"]].round(2)
    print(report.frame(est_rows))
    sc, sr = headline["scale_classical"], headline["scale_robust"]
    if np.isfinite(sc) and np.isfinite(sr) and sr > 0:
        print(f"\n  sd is {sc/sr:.2f}x the robust scale. On normal data that ratio")
        print("  is 1.00 -- anything above it is the standard deviation being")
        print("  inflated by the very outliers the band exists to catch.")

    if cfg.make_qq_plot:
        print("\n=== QQ plot ===")
        print(normality.qq_plot(x, dataset.out_path("tier5", "qq_plot.png"),
                                f"Tier 5 --- {cfg.metric} vs normal"))

    if evaluate.has_truth(scored):
        print("\n=== Detection vs known truth (synthetic only) ===")
        print(evaluate.format_stats(evaluate.detection_stats(scored)))
        print("\n  recall by failure type:")
        print(report.frame(evaluate.recall_by_cause(scored)))
        print("\n  NOTE: these flags are IN-SAMPLE -- the band was fitted on the")
        print("  same orders being scored, so the flag rate is partly circular.")

    # ---------------- outputs ----------------
    table.to_csv(dataset.out_path("tier5", "band_table.csv"), index=False)
    normality.evidence(df, cfg).to_csv(
        dataset.out_path("tier5", "normality.csv"), index=False)
    # cfg.metric is often SLIPPAGE_BPS, which is already in the list, so
    # dedupe -- selecting a duplicated column name raises in pandas.
    wanted = [schema.ORDER_ID, schema.ALGO, schema.ADV_BUCKET,
              schema.SPREAD_BPS, schema.SLIPPAGE_BPS, cfg.metric,
              "band_centre", "band_scale", "band_lo", "band_hi",
              "band_level", "zone", "rank_stat", "flagged", "review_required"]
    cols, seen = [], set()
    for col in wanted:
        if col in scored.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    scored[cols].to_csv(dataset.out_path("tier5", "scored_orders.csv"), index=False)
    print("\nWrote outputs/tier5/ -> band_table.csv, scored_orders.csv,"
          " normality.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the self-check and verify it passes**

Run:

```bash
python -m tier5_gaussian.run --self-check
```

Expected: six lines each ending `OK`, then `ALL CHECKS PASSED`, exit code 0. Confirm the exit code:

```bash
python -m tier5_gaussian.run --self-check; echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 3: Run the full report on the synthetic book**

Run:

```bash
python -m tier5_gaussian.run
```

Expected: the headline range prints; the band table shows four levels; the coverage table shows `actual_outside_pct` at `k=3` above `0.27%`; a `required k` line prints; `skew` and `excess kurtosis` print; the classical/robust comparison prints; `outputs/tier5/qq_plot.png` is written; three CSVs are written.

- [ ] **Step 4: Verify every CLI override works**

Run each and confirm the `=== Band ===` header line reflects the override and the run completes:

```bash
python -m tier5_gaussian.run --metric perf_norm
python -m tier5_gaussian.run --k 2.5
python -m tier5_gaussian.run --estimator robust
python -m tier5_gaussian.run --score-level algo_x_adv_bucket
```

Expected: `--metric perf_norm` prints a range in sigma units, not bps. `--k 2.5` widens the flag rate. `--estimator robust` produces a narrower range than the default. `--score-level algo_x_adv_bucket` makes the flag-rate-by-bucket column visibly flatter than the `ALL` run, and `band_level` in `scored_orders.csv` contains a mix of `algo_x_adv_bucket` and `ALL`.

- [ ] **Step 5: Verify the outputs**

Run:

```bash
python -c "
import pandas as pd
for f in ['band_table','scored_orders','normality']:
    d = pd.read_csv(f'outputs/tier5/{f}.csv')
    print(f, d.shape, list(d.columns)[:6])
"
```

Expected: `band_table` has a `trusted` column and one row per level/group; `scored_orders` has one row per order with `band_lo`, `band_hi`, `zone`, `rank_stat`; `normality` has an `ALL` row plus one per algo.

- [ ] **Step 6: Commit**

```bash
git add tier5_gaussian/run.py
git commit -m "Add Tier 5 CLI driver with self-check and evidence report"
```

---

### Task 5: Tier 5 README

**Files:**
- Create: `tier5_gaussian/README.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Run the tier and capture the real numbers**

Run:

```bash
python -m tier5_gaussian.run > /tmp/tier5_out.txt 2>&1; cat /tmp/tier5_out.txt
```

Every number written into the README in Step 2 must be copied from this output. Do not invent figures.

- [ ] **Step 2: Write the README**

Create `tier5_gaussian/README.md` with these sections, in this order:

1. **Title and usage block** — `# Tier 5 --- Gaussian 3-sigma band`, then a bash block with the six invocations from `run.py`'s docstring.
2. **The method** — the four numbered steps from the spec (take the slippage, estimate mu and sigma, band at `mu +/- 3 sigma`, flag outside), then the sentence explaining that `k = 3` is a promise about the flag rate: 99.73% inside, so 0.27% flagged, about 1 in 370.
3. **Why anyone wants this** — two numbers, one formula, no model, checkable by hand, and it is what most statistical-process-control practice says in writing. State this fairly; it is a legitimate choice.
4. **Does 3 sigma mean what it says here?** — paste the real coverage table from Step 1, the real `required k` line, and the real skew/excess-kurtosis figures. Then the caveat that at this sample size every formal normality test rejects, so the coverage table and the QQ plot are the evidence, not the p-value.
5. **Classical vs robust** — paste the real side-by-side block from Step 1 and the real `sd is N.NNx the robust scale` line. Explain that `1.4826 * MAD` is a consistent estimator of sigma under normality, so on normal data the two agree and any gap is the non-normality.
6. **Known weaknesses** — four bullets, each stated plainly:
   - *The band is symmetric by construction.* Slippage is not, so the two tails are not equally served. `k_lo` and `k_hi` in the report quantify the gap.
   - *One band for the whole book does not adjust for difficulty.* The flag rate slopes with %ADV, the same failure Tier 1 had. `--score-level algo_x_adv_bucket` reduces it but does not remove it.
   - *Fitted and scored on the same orders.* The flag rate is therefore partly circular. `tier3_model` is where the out-of-sample number lives.
   - *The standard deviation is inflated by the outliers it is hunting.* That is what the robust row measures.
7. **Knobs** — one line naming `tier5_gaussian/config.py` and listing `k_sigma`, `metric`, `estimator`, `score_level`, `min_group_n`, `min_notional_review`, `make_qq_plot`.

Match the house style of `tier4_vwap/README.md`: ASCII only (`-->` not arrows, `+/-` not the sign), tables for comparisons, plain declarative sentences.

- [ ] **Step 3: Verify every number in the README appears in the captured output**

Run:

```bash
cat tier5_gaussian/README.md
```

Read it against `/tmp/tier5_out.txt` and confirm each figure matches. Fix any that do not.

- [ ] **Step 4: Commit**

```bash
git add tier5_gaussian/README.md
git commit -m "Add Tier 5 README with measured results"
```

---

### Task 6: Remove Tiers 1, 2 and 4; rewrite the top-level runner

**Files:**
- Delete: `tier1_fixed/`, `tier2_percentile/`, `tier4_vwap/`, `synthetic_vwap.py`
- Modify: `run.py` (full rewrite)

**Interfaces:**
- Consumes: `tier3_model` (`aggregate`, `config`, `cost_model`, `diagnostics`, `scoring`), `tier5_gaussian` (`band`, `config`), `tca.dataset`, `tca.evaluate`, `tca.report`, `tca.schema`
- Produces: `outputs/tier3/scored_orders.csv`, `outputs/tier5/scored_orders.csv`

The deletion and the rewrite land in **one commit** because deleting the tiers breaks `run.py`'s imports.

- [ ] **Step 1: Confirm nothing else depends on the doomed folders**

Run:

```bash
grep -rn "tier1_fixed\|tier2_percentile\|tier4_vwap\|synthetic_vwap" --include="*.py" .
```

Expected: hits only in `run.py` (lines 32-33) and inside the folders being deleted. If anything else appears, stop and report it before deleting.

- [ ] **Step 2: Delete**

```bash
git rm -r tier1_fixed tier2_percentile tier4_vwap
git rm synthetic_vwap.py
```

- [ ] **Step 3: Rewrite the top-level runner**

Replace the entire contents of `run.py` with:

```python
"""Run both methods on the SAME orders and compare them head to head.

    python run.py                    # synthetic HK VWAP demo
    python run.py --csv your.csv     # your real extract
    python run.py --budget 3         # hold both methods to a 3% review queue

For a single method with its full report, run it directly:

    python -m tier3_model.run
    python -m tier5_gaussian.run

The comparison has two halves, and the second is the one that matters.

  "At each method's own threshold" is how they would actually behave in
  production -- but it is not a fair test of METHOD, because they flag
  different fractions of the book and recall rises trivially with queue size.

  "At a matched review budget" fixes that: each method ranks the entire book
  by its own severity statistic, the top N% go to the queue, and we ask who
  filled that fixed-size queue with real problems. That is a comparison of
  ranking quality, independent of where anyone set their limit.
"""

from __future__ import annotations
import argparse

import pandas as pd

import config as root_config
from tca import dataset, evaluate, report, schema
from tier3_model import (aggregate, config as t3cfg, cost_model, diagnostics,
                         scoring)
from tier5_gaussian import band, config as t5cfg, normality


def run_tier3(df, seed: int):
    cfg = t3cfg.CONFIG
    preds, model = cost_model.cross_fit_predict(df, cfg, seed=seed)
    scored = scoring.add_scores(df, preds, cfg)
    causes = diagnostics.fit_causes(scored, cfg)
    attributed = diagnostics.attribute(scored, causes)
    desc = (f"quantile regression tau={cfg.tau_lo:g}/{cfg.tau_hi:g}, "
            f"{cfg.n_folds}-fold out-of-sample")
    return attributed, desc, preds, cfg


def run_tier5(df):
    cfg = t5cfg.CONFIG
    table = band.fit(df, cfg)
    model = band.BandModel(table, cfg)
    desc = (f"mu +/- {cfg.k_sigma:g} sigma on {cfg.metric}, "
            f"{cfg.estimator}, {cfg.score_level}, in-sample")
    return model.score_frame(df), desc, table, cfg


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--budget", type=float, default=3.0,
                    help="Matched review-queue size, in %% of orders.")
    args = ap.parse_args()

    df, clean_report = dataset.load_prepared(args)
    print("\n=== Cleaning (shared by both methods) ===")
    print(clean_report.as_text())

    t3, d3, preds3, cfg3 = run_tier3(df, seed=args.seed)
    t5, d5, table5, cfg5 = run_tier5(df)
    methods = [("Tier 3 model", t3, d3), ("Tier 5 gaussian", t5, d5)]

    # ---------------- behaviour at each method's own threshold ----------------
    print(report.header("1. AT EACH METHOD'S OWN THRESHOLD"))
    rows = []
    for name, scored, desc in methods:
        r = {"method": name, "rule": desc,
             "flag_rate_pct": round(100 * scored["flagged"].mean(), 2),
             "review_pct": round(100 * scored["review_required"].mean(), 2)}
        r.update({k: round(v, 1) for k, v in
                  evaluate.detection_stats(scored).items()
                  if k in ("precision_pct", "recall_pct", "f1_pct")})
        rows.append(r)
    print(report.frame(pd.DataFrame(rows)))

    # ---------------- calibration across difficulty ----------------
    print(report.header("2. IS THE THRESHOLD CALIBRATED? (flag rate by %ADV bucket)"))
    order = [b for b in ["<1%", "1-5%", "5-10%", "10-20%", ">20%", "unknown"]
             if b in set(df[schema.ADV_BUCKET])]
    cal = pd.DataFrame({
        name: (100 * scored.groupby(schema.ADV_BUCKET, observed=False)["flagged"].mean())
        for name, scored, _ in methods
    }).reindex(order).round(2)
    cal.insert(0, "n", df.groupby(schema.ADV_BUCKET, observed=False).size().reindex(order))
    print(report.frame(cal))
    cols = [c for c in cal.columns if c != "n"]
    spread = (cal[cols].max() - cal[cols].min()).round(2)
    print("\n  flag-rate spread across buckets (lower = better calibrated):")
    for k, v in spread.items():
        print(f"    {k:<20} {v:>6.2f} pp")
    print("\n  A calibrated threshold flags roughly the same share of easy and hard")
    print("  orders. Anything else means you are measuring difficulty, not quality.")

    # ---------------- matched-budget ranking quality ----------------
    if evaluate.has_truth(df):
        print(report.header(f"3. AT A MATCHED {args.budget:g}% REVIEW BUDGET "
                            f"(the fair comparison)"))
        rows = []
        for name, scored, _ in methods:
            s = evaluate.precision_at_budget(scored, args.budget)
            if s:
                rows.append({"method": name, "queue": s["queue"],
                             "caught": s["caught"],
                             "precision_pct": round(s["precision_pct"], 1),
                             "recall_pct": round(s["recall_pct"], 1)})
        print(report.frame(pd.DataFrame(rows)))
        print(f"\n  {int(df[schema.TRUE_OUTLIER].sum()):,} orders in this book were "
              f"genuinely broken. Each method got the same")
        print("  size queue; the difference is purely how well it ranked.")

        print("\n  recall by failure type, at the matched budget:")
        rec = {}
        for name, scored, _ in methods:
            k = max(int(round(args.budget / 100 * len(scored))), 1)
            top = scored["rank_stat"].fillna(-1e18).nlargest(k).index
            pick = pd.Series(False, index=scored.index)
            pick.loc[top] = True
            real = scored[scored[schema.TRUE_OUTLIER]]
            rec[name] = (100 * pick.loc[real.index]
                         .groupby(real[schema.TRUE_CAUSE]).mean()).round(1)
        print(report.frame(pd.DataFrame(rec)))

    # ---------------- does the Gaussian assumption hold? ----------------
    print(report.header("4. DOES THE 3-SIGMA PROMISE HOLD?"))
    headline = table5[table5["level"] == t5cfg.LEVEL_ALL].iloc[0]
    c = headline[f"centre_{cfg5.estimator}"]
    s = headline[f"scale_{cfg5.estimator}"]
    x = df[cfg5.metric].to_numpy()
    print(report.frame(normality.coverage_table(x, c, s).round(3)))
    req = normality.required_k(x, c, s)
    print(f"\n  To actually flag {100*normality.NOMINAL_OUTSIDE:.2f}% of this book"
          f" you need k = {req['k_symmetric']:.2f}, not {cfg5.k_sigma:g}.")
    print("  That gap is the cost of assuming a shape the data does not have.")

    # ---------------- what only Tier 3 gives you ----------------
    print(report.header("5. WHAT ONLY TIER 3 PRODUCES"))

    print("\n--- Out-of-sample calibration (the check that works on real data) ---")
    print(report.frame(cost_model.coverage_check(df, preds3, cfg3)))

    print("\n--- Systematic effects: mean-z t-tests, BH-corrected ---")
    sig = aggregate.significant(aggregate.slice_report(t3))
    if len(sig):
        cols = [c for c in ["dimension", schema.ALGO, schema.BROKER,
                            schema.ADV_BUCKET, "n", "mean_z", "t_stat",
                            "q_value", "verdict"] if c in sig.columns]
        print(report.frame(sig[cols].head(12)))
        print("\n  Tier 5 cannot produce this table at all: without an expected")
        print("  cost there is no residual to average, so a broker that is")
        print("  consistently 0.2 sigma worse is indistinguishable from one")
        print("  that simply got handed the harder orders.")
    else:
        print("  none significant")

    print("\n--- Cause attribution across the review queue ---")
    print(report.frame(diagnostics.cause_summary(
        t3, currency=root_config.DATA.notional_currency)))
    conf = diagnostics.cause_confusion(t3)
    if len(conf):
        print("\n  attributed vs KNOWN cause (synthetic only):")
        print(report.frame(conf))
        acc = diagnostics.cause_accuracy(t3)
        print("\n  attribution accuracy:")
        print(report.frame(acc))
        hit, tot = acc["attributed_correctly"].sum(), acc["flagged"].sum()
        print(f"\n  {hit}/{tot} flagged true failures got the right diagnosis "
              f"({100*hit/tot:.0f}%).")
        print("  That is the deliverable: not 'this order was bad', but which of")
        print("  four different problems it was, each with a different remedy.")

    for name, scored, _ in methods:
        folder = {"Tier 3 model": "tier3", "Tier 5 gaussian": "tier5"}[name]
        scored.to_csv(dataset.out_path(folder, "scored_orders.csv"), index=False)
    print("\nWrote outputs/tier{3,5}/scored_orders.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the comparison runs end to end**

Run:

```bash
python run.py
```

Expected: five numbered sections print; section 1 has two rows; section 2 shows the flag-rate spread for both methods (Tier 5 at `score_level=ALL` should have the larger spread, since one band for the whole book does not adjust for difficulty); section 3 shows both at the same queue size; section 4 shows the coverage table and the required-k line; section 5 is unchanged Tier 3 output. Both CSVs are written.

- [ ] **Step 5: Verify nothing else broke**

Run:

```bash
python -m tier3_model.run
python -m tier5_gaussian.run --self-check
python check_extract.py --help
python distribution.py --help
```

Expected: all four succeed. `tier3_model.run` must still write `outputs/tier3/model.json`.

- [ ] **Step 6: Verify score_new.py still works against the frozen model**

Run:

```bash
python -c "
import synthetic_data
synthetic_data.generate(n=2000, seed=99).to_csv('outputs/_probe.csv', index=False)
"
python score_new.py outputs/_probe.csv
rm outputs/_probe.csv
```

Expected: it loads `outputs/tier3/model.json`, scores the probe file and prints the drift block. If `model.json` is missing, run `python -m tier3_model.run` first.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Remove tiers 1, 2 and 4; compare tier3_model vs tier5_gaussian"
```

---

### Task 7: Update the root README and requirements

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Capture the real numbers**

Run:

```bash
python run.py > /tmp/run_out.txt 2>&1; cat /tmp/run_out.txt
```

Every figure that goes into the README must come from this output.

- [ ] **Step 2: Delete the sections that no longer have code behind them**

In `README.md`, delete these headed sections in full:

- `## Tier 1 --- Fixed thresholds`
- `## Tier 2 --- Percentile bands within peer groups`

Delete the blockquote near the top that begins `> **Measuring VWAP orders against interval VWAP? Start at` and ends `mean-z tests in Tier 3 structurally cannot see.` — it points at the deleted `tier4_vwap/README.md`.

- [ ] **Step 3: Replace the tier table**

Replace the table under `The same problem is solved three ways...` and its lead-in paragraph with:

```markdown
The same problem is solved two ways, and they answer different questions.
Each is self-contained in its own folder, and both run on the identical set of
orders so the comparison is honest.

| | question it answers | folder |
|---|---|---|
| **Tier 3** | "did this order cost more than expected *for an order like it* --- and what went wrong?" | `tier3_model/` |
| **Tier 5** | "is this order beyond 3 standard deviations of the book?" | `tier5_gaussian/` |

> The tier numbers are historical. Tiers 1, 2 and 4 were removed; the numbering
> was not reflowed, so git history and the frozen `outputs/tier3/model.json`
> stay valid. Tier 5 is not more sophisticated than Tier 3 --- it is the
> parametric alternative, and Section "Tier 5" below says where each one wins.
```

- [ ] **Step 4: Add the Tier 5 section**

Insert a `## Tier 5 --- Gaussian 3-sigma band` section immediately after the `## Tier 3 --- Expected-cost model and residual scoring` section ends. Content, using real numbers from Step 1:

1. The method in four steps, and the sentence that `k = 3` promises 0.27% flagged.
2. The real coverage table from section 4 of `/tmp/run_out.txt`.
3. The real required-k line.
4. The classical-vs-robust point: `1.4826 * MAD` is a consistent estimator of sigma under normality, so on normal data the two agree and any gap is the non-normality.
5. Four "where it breaks" bullets, matching the tier README: symmetric by construction, one band does not adjust for difficulty, fitted and scored in-sample, sd inflated by its own outliers.
6. A pointer to `tier5_gaussian/README.md`.

- [ ] **Step 5: Update the running instructions**

In the `## Running it` section, replace the Step 2 block with:

```bash
python run.py --csv your_file.csv               # both methods, side by side
python -m tier3_model.run --csv your_file.csv   # the full Tier 3 report
python -m tier5_gaussian.run --csv your_file.csv  # the full Tier 5 report
```

Delete the block listing `python -m tier1_fixed.run` and `python -m tier2_percentile.run`. In the `distribution.py` block, change `outputs/tier4/scored_orders.csv` to `outputs/tier5/scored_orders.csv`.

- [ ] **Step 6: Update the Evidence section**

The comparison tables under `## Evidence` list Tier 1, Tier 2 and Tier 3 columns. Rebuild both tables from `/tmp/run_out.txt`:

- the calibration table (flag rate by %ADV bucket) keeps its row labels and gets two columns, `Tier 3` and `Tier 5`, plus the `spread` row from section 2's output
- the matched-budget table keeps its columns and gets two rows, from section 3's output

Delete the subsection `### An honest result worth reading twice` — it compares two Tier 2 invocations that no longer exist.

- [ ] **Step 7: Update the Layout section**

Replace the layout tree with:

```
check_extract.py       preflight: validate a file and infer its settings
distribution.py        seaborn plots of the performance distribution
score_new.py           apply a frozen threshold to future orders
config.py              shared data contract --- the only file you edit for real data
synthetic_data.py      impact-shaped demo book (arrival-price DGP) + truth labels
run.py                 both methods, side by side
tca/                   shared infrastructure, no thresholds of its own
  schema.py              canonical column names
  pipeline.py            load -> units -> clean -> metrics -> buckets
  dataset.py             CLI to prepared frame (so both methods score the same rows)
  evaluate.py            precision/recall, including the matched-budget comparison
  report.py              shared formatting
tier3_model/           Tier 3: cost model, z-scores, slice tests, cause attribution
  features.py            the design matrix (square-root law)
  cost_model.py          quantile regression, cross-fitting, calibration
  scoring.py             residual z, zones, severity tiers, review gate
  persist.py             freeze/load the threshold + drift detection
  aggregate.py           slice tests: bias (t-test) and consistency (Levene)
  diagnostics.py         cause attribution + single-order narratives
tier5_gaussian/        Tier 5: Gaussian mu +/- 3 sigma band
  band.py                fit both estimators per group, score orders
  normality.py           coverage, required-k, shape statistics, QQ plot
```

Change the sentence after it from "Each tier folder has its own README" to "Both tier folders have their own README with the method, the settings and their known weaknesses."

- [ ] **Step 8: Sweep for stale references**

Run:

```bash
grep -n "Tier 1\|Tier 2\|Tier 4\|tier1_fixed\|tier2_percentile\|tier4_vwap\|synthetic_vwap\|three tiers\|three ways\|all three" README.md
```

Expected after fixing: no hits. Every hit is a stale reference — rewrite or delete it. Pay attention to `## Why this is harder than it looks` (it ends by naming Tier 2 and Tier 3 — keep the Tier 3 half, drop the Tier 2 half), `### Configuration` (says "Each tier's thresholds live in its own folder's `config.py`" — still true, leave it), and `## Known limits`.

- [ ] **Step 9: Update requirements.txt**

Replace the first comment block. The file becomes:

```
# Shared pipeline + Tier 5 (Gaussian band)
pandas>=2.0
numpy>=1.24

# Tier 3: quantile regression cost model, t-tests and BH correction
# Tier 5: D'Agostino K2 normality test and the QQ plot
scipy>=1.10
statsmodels>=0.14
# statsmodels is optional in practice -- tier3_model falls back to bucketed
# empirical quantiles when it is absent (backend="auto"). scipy is required.

# distribution.py and the Tier 5 QQ plot only. Nothing in the scoring path
# imports these, so both methods still run in an environment without them.
seaborn>=0.13
matplotlib>=3.7
```

- [ ] **Step 10: Verify the README against reality**

Run:

```bash
grep -c "Tier 5" README.md
python run.py > /tmp/run_out2.txt 2>&1; echo "exit=$?"
```

Expected: `Tier 5` appears several times; `exit=0`. Read the README start to finish once and confirm no sentence promises a tier that no longer exists.

- [ ] **Step 11: Commit**

```bash
git add README.md requirements.txt
git commit -m "Rewrite README and requirements for the two surviving methods"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task:

| spec section | task |
|---|---|
| `config.py` knobs | 1 |
| `band.py` two estimators, four levels, scoring, `rank_stat`, NaN handling | 2 |
| `normality.py` coverage, required-k, shape, QQ, graceful degradation | 3 |
| `run.py` CLI, `--metric`/`--k`/`--estimator`/`--score-level`, report layout | 4 |
| outputs: `band_table.csv`, `scored_orders.csv`, `normality.csv`, `qq_plot.png` | 4 |
| `--self-check` verification | 4 |
| tier README | 5 |
| deletions of tiers 1, 2, 4 and `synthetic_vwap.py` | 6 |
| top-level `run.py` rewrite | 6 |
| root README rewrite, `requirements.txt` | 7 |
| naming: folders not reflowed | 6, 7 (README note) |

**Type consistency.** `estimates()` returns the same 13 keys everywhere it is called (Task 2 defines it; Tasks 3 and 4 consume it). Column names `centre_{est}` / `scale_{est}` / `lo_{est}` / `hi_{est}` are constructed identically in `band.BandModel.__init__`, `normality.evidence`, `run.py` `main()` and the root `run.py`. `LEVEL_ALL` is referenced as `t5cfg.LEVEL_ALL` in all four modules. `NOMINAL_OUTSIDE` lives in `normality.py` and is read from there by both runners.

**Known risk, flagged rather than hidden.** Task 6 Step 6 depends on `outputs/tier3/model.json` already existing. If it does not, the step says to run `python -m tier3_model.run` first. This is the only ordering dependency outside the task sequence.
