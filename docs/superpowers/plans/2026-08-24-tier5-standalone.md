# Tier 5 Standalone Band-Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a copyable `tier5_standalone/` folder that fits a Gaussian band on a one-year extract, freezes it to JSON, and scores a later period against that frozen band — split by region and strategy, one folder per cell.

**Architecture:** `fit.py` is the only module that computes a centre or a scale; it writes `bands/<REGION>/<STRATEGY>.json`. `score.py` loads those bands and applies `lo`/`hi` unchanged, never refitting. Region comes from the `Sym` suffix, strategy from the `Strategy` column, and the period label from the `Date` column — all derived from the data, so the same command works whether the twelve cells arrive as one file or twelve. `cells.py` is the shared derivation both sides use, which is what guarantees a band written for `HK/VWAP` is the one `score` looks up for those rows.

**Tech Stack:** Python 3.13, pandas, numpy, scipy, matplotlib, seaborn, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-tier5-standalone-band-freeze-design.md`

## Global Constraints

- Everything lives under `tier5_standalone/`. Nothing outside that folder may be imported.
- The package is named `tier5`, not `tier5_gaussian`. All copied modules have their imports rewritten.
- `score.py` must never call `band.estimates`, `band.fit`, or otherwise compute a centre or scale. It reads them from JSON only.
- `tca/evaluate.py` is NOT copied. Any call into it is removed.
- Region codes are `AU`, `HK`, `JP`, `IN`. No suffix aliasing. Unrecognised suffixes are fitted and reported, never dropped.
- The date source column is `Date`, mapped to `schema.ORDER_DATE`. It is optional in code: absent dates fall back to `--label` and skip the overlap check.
- Bands are immutable once written. Nothing but `fit` writes a band file.
- `FORMAT_VERSION = 1`. A mismatch on load raises, never guesses.
- `min_group_n = 200`. Below that, `fit` refuses without `--force`.
- Default `k_sigma = 3.0`, `metric = slippage_bps`, `estimator = classical`.
- No module in the scoring path may import matplotlib or seaborn at module level.
- Tests run with `pytest tier5_standalone/tests/ -v` from the repo root.

---

## File Structure

**Copied unchanged except for import rewrites (`tier5_gaussian` → `tier5`):**

| File | Responsibility |
|---|---|
| `tier5_standalone/tca/schema.py` | Canonical column names. **Modified:** adds `ORDER_DATE`. |
| `tier5_standalone/tca/pipeline.py` | load → clean → derive metrics → bucket |
| `tier5_standalone/tca/dataset.py` | CLI args → prepared frame |
| `tier5_standalone/tca/report.py` | `header()`, `frame()`, `zone_summary()` |
| `tier5_standalone/config.py` | COLUMN_MAP / units / sign. **Modified:** adds `Date` mapping and `REGION_NAMES`. |
| `tier5_standalone/synthetic_data.py` | Demo book. **Modified:** emits a `Date` column. |
| `tier5_standalone/check_extract.py` | Pre-flight inspection of a new extract |
| `tier5_standalone/tier5/config.py` | `Tier5Config` dataclass |
| `tier5_standalone/tier5/band.py` | `estimates()`, `classify()`, `BandModel` |
| `tier5_standalone/tier5/normality.py` | coverage, `required_k`, shape stats, QQ plot |
| `tier5_standalone/tier5/run.py` | Existing single-run explore driver. **Modified:** evaluate call removed. |

**New:**

| File | Responsibility |
|---|---|
| `tier5_standalone/tier5/cells.py` | Derive (region, strategy) cells, period label, and every output path |
| `tier5_standalone/tier5/persist.py` | Freeze a band to JSON, load it back, drift report |
| `tier5_standalone/tier5/curve.py` | Seaborn KDE + fitted normal + band lines |
| `tier5_standalone/tier5/fit.py` | Year extract → band files |
| `tier5_standalone/tier5/score.py` | Band files + later extract → outliers |
| `tier5_standalone/tier5/batch.py` | Walk a directory of extracts |
| `tier5_standalone/README.md` | Usage doc |
| `tier5_standalone/tests/test_*.py` | Test suite |

---

## Task 1: Scaffold the standalone folder

**Files:**
- Create: `tier5_standalone/` with the copied tree described above
- Create: `tier5_standalone/tier5/__init__.py`, `tier5_standalone/tca/__init__.py`
- Create: `tier5_standalone/tests/__init__.py`, `tier5_standalone/tests/conftest.py`
- Modify: `tier5_standalone/tca/schema.py` (add `ORDER_DATE`)
- Modify: `tier5_standalone/config.py` (add `Date` mapping, `REGION_NAMES`)
- Modify: `tier5_standalone/synthetic_data.py` (emit `Date`)
- Modify: `tier5_standalone/tier5/run.py` (drop the evaluate import and its block)
- Test: `tier5_standalone/tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing
- Produces: `schema.ORDER_DATE == "order_date"`; `config.REGION_NAMES` dict; `synthetic_data.generate(n, seed, start_date="2025-06-02", end_date="2026-05-29")` emitting a `Date` column of `YYYY-MM-DD` strings; the importable package `tier5`

- [ ] **Step 1: Copy the tree**

```bash
cd C:/Users/user/Desktop/Projects/Threshold
mkdir -p tier5_standalone/tca tier5_standalone/tier5 tier5_standalone/tests
cp config.py synthetic_data.py check_extract.py requirements.txt tier5_standalone/
cp tca/__init__.py tca/schema.py tca/pipeline.py tca/dataset.py tca/report.py tier5_standalone/tca/
cp tier5_gaussian/__init__.py tier5_gaussian/config.py tier5_gaussian/band.py \
   tier5_gaussian/normality.py tier5_gaussian/run.py tier5_standalone/tier5/
touch tier5_standalone/tests/__init__.py
```

Note `tca/evaluate.py` is deliberately not copied.

- [ ] **Step 2: Rewrite imports**

In `tier5_standalone/tier5/*.py`, replace every `tier5_gaussian` with `tier5`:

```bash
cd C:/Users/user/Desktop/Projects/Threshold/tier5_standalone
sed -i 's/tier5_gaussian/tier5/g' tier5/*.py
```

- [ ] **Step 3: Add the date column to the schema**

In `tier5_standalone/tca/schema.py`, after the `SIDE = "side"` line in the identity block:

```python
ORDER_DATE = "order_date"  # order start date. Optional: used only to label
                           # output folders, stamp the fit window into a band
                           # file, and refuse a scoring window that overlaps
                           # the fit window. Never enters the metric or band.
```

Do NOT add it to `schema.NUMERIC` (it is not numeric) or `schema.ESSENTIAL` (a
missing date must not drop the row).

- [ ] **Step 4: Map it in config.py**

In `tier5_standalone/config.py`, inside `COLUMN_MAP`, after the `schema.SIDE` line:

```python
    schema.ORDER_DATE:    "Date",           # optional; labels periods only
```

And after the `REVERSION_SIGN` block, add:

```python
# ---------------------------------------------------------------------------
# 3c) REGIONS  ---  the Sym suffix, already derived into schema.MARKET
# ---------------------------------------------------------------------------
# Presentation only. Output folders use the two-letter code itself, so a venue
# missing from this map still fits and scores normally -- it is reported as an
# unrecognised region rather than dropped.
REGION_NAMES = {
    "AU": "Australia",
    "HK": "Hong Kong",
    "JP": "Japan",
    "IN": "India",
}
```

- [ ] **Step 5: Emit a Date column from synthetic_data**

In `tier5_standalone/synthetic_data.py`, change the `generate` signature and add
the column. Replace the signature line:

```python
def generate(n: int = 12000, market: str = "HK", seed: int = 7,
             with_diagnostics: bool = True,
             start_date: str = "2025-06-02",
             end_date: str = "2026-05-29") -> pd.DataFrame:
```

Then immediately before the `out = pd.DataFrame({` line, add:

```python
    # Trading dates spread uniformly across the window, so the demo exercises
    # period labelling and the overlap check rather than only the fallback.
    span = pd.bdate_range(start_date, end_date)
    order_date = pd.Series(span[rng.integers(0, len(span), size=n)]).dt.strftime("%Y-%m-%d")
```

and inside the `pd.DataFrame({...})` literal, after the `"market": market,` line:

```python
        "Date": order_date.to_numpy(),
```

The key is literally `"Date"`, matching `COLUMN_MAP`, because `synthetic_data`
stands in for a raw extract.

- [ ] **Step 6: Remove the evaluate dependency from run.py**

In `tier5_standalone/tier5/run.py`, change the import line:

```python
from tca import dataset, report, schema
```

and delete the entire block beginning `if evaluate.has_truth(scored):` through
the line ending `print("  same orders being scored, so the flag rate is partly circular.")`.

- [ ] **Step 7: Write the scaffold test**

Create `tier5_standalone/tests/conftest.py`:

```python
import os
import sys

# The standalone folder is its own root: `python -m tier5.fit` is always run
# from inside it, so tests must import the same way.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

Create `tier5_standalone/tests/test_scaffold.py`:

```python
import pandas as pd

import config
import synthetic_data
from tca import schema
from tier5 import band, normality


def test_order_date_in_schema():
    assert schema.ORDER_DATE == "order_date"
    assert schema.ORDER_DATE not in schema.NUMERIC
    assert schema.ORDER_DATE not in schema.ESSENTIAL


def test_date_is_mapped():
    assert config.COLUMN_MAP[schema.ORDER_DATE] == "Date"


def test_region_names():
    assert set(config.REGION_NAMES) == {"AU", "HK", "JP", "IN"}


def test_synthetic_data_emits_dates():
    df = synthetic_data.generate(n=500, seed=1)
    assert "Date" in df.columns
    d = pd.to_datetime(df["Date"])
    assert d.min().year == 2025
    assert d.max().year == 2026


def test_date_survives_prepare():
    raw = synthetic_data.generate(n=500, seed=1)
    from tca import pipeline
    df, _ = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                             config.SLIPPAGE_SIGN, pre_transform=config.PRE_TRANSFORM)
    assert schema.ORDER_DATE in df.columns
    assert df[schema.ORDER_DATE].notna().all()


def test_no_tier3_or_evaluate_imported():
    import tier5.run
    assert not hasattr(tier5.run, "evaluate")


def test_band_self_check_still_passes():
    import numpy as np
    x = np.random.default_rng(11).normal(-8.7, 18.4, 200_000)
    e = band.estimates(x, 3.0)
    assert abs(e["centre_classical"] - (-8.7)) < 0.20
    assert abs(e["scale_classical"] - 18.4) < 0.20
    outside = float(np.mean((x < e["lo_classical"]) | (x > e["hi_classical"])))
    assert abs(outside - (1 - normality.promised_inside(3.0))) < 0.0006
```

- [ ] **Step 8: Run the tests**

Run: `python -m pytest tier5_standalone/tests/test_scaffold.py -v`
Expected: all 7 PASS.

- [ ] **Step 9: Confirm the folder runs standalone**

```bash
cd tier5_standalone && python -m tier5.run --self-check && cd ..
```

Expected: `ALL CHECKS PASSED`.

- [ ] **Step 10: Commit**

```bash
git add tier5_standalone/
git commit -m "Scaffold tier5_standalone: copy tier 5, add Date column and regions"
```

---

## Task 2: `tier5/cells.py` — cell and path derivation

**Files:**
- Create: `tier5_standalone/tier5/cells.py`
- Test: `tier5_standalone/tests/test_cells.py`

**Interfaces:**
- Consumes: `schema.MARKET`, `schema.ALGO`, `schema.ORDER_DATE` from Task 1
- Produces:
  - `cells(df) -> list[tuple[str, str, pd.DataFrame]]` sorted by (region, strategy)
  - `date_range(df) -> tuple[pd.Timestamp|None, pd.Timestamp|None]`
  - `period_label(df) -> str|None`
  - `safe(name) -> str`
  - `band_path(bands_dir, region, strategy) -> str`
  - `out_dir(root, kind, period, region, strategy) -> str`
  - `windows_overlap(a_lo, a_hi, b_lo, b_hi) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_cells.py`:

```python
import os

import pandas as pd
import pytest

from tca import schema
from tier5 import cells


def _df(rows):
    return pd.DataFrame(rows)


def test_cells_splits_by_region_and_strategy():
    df = _df([
        {schema.MARKET: "HK", schema.ALGO: "VWAP"},
        {schema.MARKET: "HK", schema.ALGO: "VWAP"},
        {schema.MARKET: "HK", schema.ALGO: "TWAP"},
        {schema.MARKET: "JP", schema.ALGO: "VWAP"},
    ])
    got = [(r, s, len(g)) for r, s, g in cells.cells(df)]
    assert got == [("HK", "TWAP", 1), ("HK", "VWAP", 2), ("JP", "VWAP", 1)]


def test_cells_is_sorted():
    df = _df([
        {schema.MARKET: "JP", schema.ALGO: "VWAP"},
        {schema.MARKET: "AU", schema.ALGO: "TWAP"},
    ])
    assert [(r, s) for r, s, _ in cells.cells(df)] == [("AU", "TWAP"), ("JP", "VWAP")]


def test_cells_uppercases_and_strips_region():
    df = _df([{schema.MARKET: " hk ", schema.ALGO: " VWAP "}])
    r, s, _ = cells.cells(df)[0]
    assert (r, s) == ("HK", "VWAP")


def test_period_label_single_month():
    df = _df([{schema.ORDER_DATE: "2026-07-03"}, {schema.ORDER_DATE: "2026-07-28"}])
    assert cells.period_label(df) == "2026-07"


def test_period_label_range():
    df = _df([{schema.ORDER_DATE: "2025-06-02"}, {schema.ORDER_DATE: "2026-05-29"}])
    assert cells.period_label(df) == "2025-06_2026-05"


def test_period_label_none_without_dates():
    assert cells.period_label(_df([{schema.MARKET: "HK"}])) is None


def test_period_label_none_when_all_dates_unparseable():
    df = _df([{schema.ORDER_DATE: "not a date"}])
    assert cells.period_label(df) is None


def test_safe_sanitises():
    assert cells.safe("VWAP/Passive") == "VWAP_Passive"
    assert cells.safe("") == "UNKNOWN"


def test_band_path_nests_by_region_then_strategy():
    p = cells.band_path("bands", "HK", "VWAP")
    assert p == os.path.join("bands", "HK", "VWAP.json")


def test_out_dir_nests():
    p = cells.out_dir("outputs", "score", "2026-07", "HK", "VWAP")
    assert p == os.path.join("outputs", "score", "2026-07", "HK", "VWAP")


@pytest.mark.parametrize("a_lo,a_hi,b_lo,b_hi,expected", [
    ("2025-06-01", "2026-05-31", "2026-07-01", "2026-07-31", False),
    ("2025-06-01", "2026-05-31", "2026-05-15", "2026-07-31", True),
    ("2025-06-01", "2026-05-31", "2025-06-01", "2026-05-31", True),
])
def test_windows_overlap(a_lo, a_hi, b_lo, b_hi, expected):
    ts = pd.Timestamp
    assert cells.windows_overlap(ts(a_lo), ts(a_hi), ts(b_lo), ts(b_hi)) is expected


def test_windows_overlap_false_when_any_bound_missing():
    assert cells.windows_overlap(None, None, pd.Timestamp("2026-07-01"),
                                 pd.Timestamp("2026-07-31")) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_cells.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tier5.cells'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/cells.py`:

```python
"""Where a row belongs, and where its outputs go.

Both fit and score derive cells through this module and nothing else. That is
what guarantees the band written for HK/VWAP is the one score looks up for
those same rows -- if the two sides derived cells independently they would
drift apart the first time a strategy name changed case.
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
            .replace({"": UNKNOWN_REGION, "NAN": UNKNOWN_REGION}))


def _strategy_series(df: pd.DataFrame) -> pd.Series:
    if schema.ALGO not in df.columns:
        return pd.Series(UNKNOWN_STRATEGY, index=df.index, dtype="object")
    return (df[schema.ALGO].astype(str).str.strip()
            .replace({"": UNKNOWN_STRATEGY, "nan": UNKNOWN_STRATEGY}))


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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_cells.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tier5_standalone/tier5/cells.py tier5_standalone/tests/test_cells.py
git commit -m "Add tier5.cells: cell derivation, period labels and output paths"
```

---

## Task 3: `tier5/persist.py` — freeze and reload a band

**Files:**
- Create: `tier5_standalone/tier5/persist.py`
- Test: `tier5_standalone/tests/test_persist.py`

**Interfaces:**
- Consumes: `band.estimates()`, `normality.required_k()`, `normality.shape_stats()`, `cells.date_range()`
- Produces:
  - `FORMAT_VERSION = 1`
  - `save(est, cfg, path, *, region, strategy, source_csv, period, df, flag_rate_pct) -> str`
  - `load(path, base_cfg) -> tuple[dict, Tier5Config, dict]`
  - `drift_report(df, scored, reference, cfg) -> tuple[pd.DataFrame, list[str]]`

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_persist.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from tca import schema
from tier5 import band, config as t5cfg, persist


def _fixture(tmp_path, n=2000, seed=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        schema.SLIPPAGE_BPS: rng.normal(-10.0, 20.0, n),
        schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n),
        schema.PCT_ADV: rng.uniform(0.1, 5.0, n),
        schema.VOLATILITY: rng.uniform(100.0, 250.0, n),
        schema.DURATION_MIN: rng.uniform(10.0, 300.0, n),
        schema.ORDER_DATE: pd.bdate_range("2025-06-02", periods=n).astype(str),
    })
    cfg = t5cfg.CONFIG
    est = band.estimates(df[cfg.metric].to_numpy(), cfg.k_sigma)
    path = str(tmp_path / "HK" / "VWAP.json")
    persist.save(est, cfg, path, region="HK", strategy="VWAP",
                 source_csv="year.csv", period="2025-06_2033-01",
                 df=df, flag_rate_pct=1.45)
    return path, est, cfg, df


def test_save_writes_nested_file(tmp_path):
    path, _, _, _ = _fixture(tmp_path)
    assert path.endswith("VWAP.json")
    with open(path) as fh:
        assert json.load(fh)["region"] == "HK"


def test_roundtrip_preserves_bounds_exactly(tmp_path):
    path, est, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["lo"] == est["lo_classical"]
    assert loaded["hi"] == est["hi_classical"]
    assert loaded["centre"] == est["centre_classical"]
    assert loaded["scale"] == est["scale_classical"]


def test_both_estimators_stored(tmp_path):
    path, est, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["lo_robust"] == est["lo_robust"]
    assert loaded["hi_robust"] == est["hi_robust"]


def test_reference_carries_shape_and_required_k(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    assert ref["flag_rate_pct"] == 1.45
    assert np.isfinite(ref["k_required"])
    assert "spread_bps" in ref["feature_medians"]


def test_fit_window_stamped(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    loaded, _, _ = persist.load(path, cfg)
    assert loaded["fit_date_min"] == "2025-06-02"
    assert loaded["fit_date_max"] is not None


def test_scoring_config_travels(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    _, loaded_cfg, _ = persist.load(path, cfg)
    assert loaded_cfg.k_sigma == cfg.k_sigma
    assert loaded_cfg.metric == cfg.metric
    assert loaded_cfg.estimator == cfg.estimator


def test_format_version_mismatch_raises(tmp_path):
    path, _, cfg, _ = _fixture(tmp_path)
    with open(path) as fh:
        payload = json.load(fh)
    payload["format_version"] = 999
    with open(path, "w") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="format version"):
        persist.load(path, cfg)


def test_drift_report_flags_moved_median(tmp_path):
    path, _, cfg, df = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    moved = df.copy()
    moved[schema.SPREAD_BPS] = moved[schema.SPREAD_BPS] * 2.0
    scored = moved.assign(flagged=False)
    table, warnings = persist.drift_report(moved, scored, ref, cfg)
    assert len(table)
    assert any("spread_bps" in w for w in warnings)


def test_drift_report_quiet_when_nothing_moved(tmp_path):
    path, _, cfg, df = _fixture(tmp_path)
    _, _, ref = persist.load(path, cfg)
    scored = df.assign(flagged=np.zeros(len(df), dtype=bool))
    scored.loc[scored.index[:int(0.0145 * len(df))], "flagged"] = True
    _, warnings = persist.drift_report(df, scored, ref, cfg)
    assert warnings == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_persist.py -v`
Expected: FAIL — `No module named 'tier5.persist'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/persist.py`:

```python
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
from tier5 import cells, normality

FORMAT_VERSION = 1

# Config fields that affect scoring and must travel with the band. Anything not
# listed falls back to the current default on load.
SCORING_FIELDS = ["k_sigma", "metric", "estimator", "min_notional_review"]

# Features whose medians are snapshotted for the drift report.
REFERENCE_FEATURES = [schema.SPREAD_BPS, schema.PCT_ADV,
                      schema.VOLATILITY, schema.DURATION_MIN]

DRIFT_MEDIAN_PCT = 25.0     # a feature median moving more than this is called out


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


def _f(v):
    """JSON cannot hold NaN portably. None is honest about a missing number."""
    v = float(v)
    return v if np.isfinite(v) else None


def save(est: dict, cfg, path: str, *, region: str, strategy: str,
         source_csv: str, period: str | None, df: pd.DataFrame,
         flag_rate_pct: float) -> str:
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
        "estimator": cfg.estimator,
        "k_sigma": float(cfg.k_sigma),
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
                 "change_pct": round(100.0 * (realized - fitted) / fitted, 1)
                 if fitted else float("nan")})

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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_persist.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tier5_standalone/tier5/persist.py tier5_standalone/tests/test_persist.py
git commit -m "Add tier5.persist: freeze a band to JSON, reload it, report drift"
```

---

## Task 4: `tier5/curve.py` — the Gaussian curve picture

**Files:**
- Create: `tier5_standalone/tier5/curve.py`
- Test: `tier5_standalone/tests/test_curve.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: `plot(x, *, centre, scale, lo, hi, path, title, subtitle=None, k=3.0) -> str`

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_curve.py`:

```python
import os

import numpy as np
import pytest

from tier5 import curve


@pytest.fixture
def sample():
    return np.random.default_rng(3).normal(-10.0, 20.0, 5000)


def test_writes_a_png(tmp_path, sample):
    path = str(tmp_path / "curve.png")
    msg = curve.plot(sample, centre=-10.0, scale=20.0, lo=-70.0, hi=50.0,
                     path=path, title="HK / VWAP")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000
    assert path in msg


def test_creates_missing_directories(tmp_path, sample):
    path = str(tmp_path / "a" / "b" / "curve.png")
    curve.plot(sample, centre=-10.0, scale=20.0, lo=-70.0, hi=50.0,
               path=path, title="t")
    assert os.path.exists(path)


def test_empty_input_skips_cleanly(tmp_path):
    msg = curve.plot(np.array([]), centre=0.0, scale=1.0, lo=-3.0, hi=3.0,
                     path=str(tmp_path / "c.png"), title="t")
    assert "skipped" in msg.lower()
    assert not os.path.exists(str(tmp_path / "c.png"))


def test_non_finite_scale_skips_cleanly(tmp_path, sample):
    msg = curve.plot(sample, centre=0.0, scale=0.0, lo=-3.0, hi=3.0,
                     path=str(tmp_path / "c.png"), title="t")
    assert "skipped" in msg.lower()


def test_not_imported_at_module_level():
    """The scoring path must run without matplotlib or seaborn installed."""
    import ast
    import pathlib
    src = pathlib.Path(curve.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level_imports = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    names = []
    for node in top_level_imports:
        if isinstance(node, ast.Import):
            names += [a.name.split(".")[0] for a in node.names]
        else:
            names.append((node.module or "").split(".")[0])
    assert "matplotlib" not in names
    assert "seaborn" not in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_curve.py -v`
Expected: FAIL — `No module named 'tier5.curve'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/curve.py`:

```python
"""The band, drawn.

One picture with two shapes on it: what the data actually looks like (a KDE of
the observed metric) and what the band assumes it looks like (the fitted normal
PDF). The gap between them IS the non-normality that the coverage table reports
numerically -- a reader who will not read a kurtosis figure can see a peaked
middle and fat ends immediately.

The band bounds are drawn as vertical lines with the out-of-band regions
shaded, so `lo` and `hi` are located on the same axes as the distribution they
came from.

matplotlib and seaborn are imported INSIDE the function on purpose. Nothing in
the scoring path may depend on a plotting library being installed.
"""

from __future__ import annotations

import os

import numpy as np


def plot(x, *, centre: float, scale: float, lo: float, hi: float,
         path: str, title: str, subtitle: str | None = None,
         k: float = 3.0) -> str:
    """Write the curve. Returns the line to print (never raises on a missing lib)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "  Curve skipped (no finite values)."
    if not np.isfinite(scale) or scale <= 0:
        return "  Curve skipped (scale is not positive)."

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        return f"  Curve skipped ({exc.name} not installed)."

    outside = float(np.mean((x < lo) | (x > hi)))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Observed shape.
    sns.kdeplot(x=x, ax=ax, fill=True, color="#2c7fb8", alpha=0.35,
                linewidth=1.6, label="observed (KDE)")

    # Assumed shape: the normal the band is built on. Plain numpy so scipy is
    # not required just to draw a Gaussian.
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 512)
    pdf = np.exp(-0.5 * ((grid - centre) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    ax.plot(grid, pdf, linestyle="--", linewidth=1.8, color="#d95f0e",
            label=f"fitted normal  N({centre:.2f}, {scale:.2f}$^2$)")

    # The band.
    ax.axvline(lo, color="#b2182b", linewidth=1.5)
    ax.axvline(hi, color="#b2182b", linewidth=1.5)
    ax.axvspan(float(np.min(x)), lo, color="#b2182b", alpha=0.07)
    ax.axvspan(hi, float(np.max(x)), color="#b2182b", alpha=0.07)

    top = ax.get_ylim()[1]
    ax.text(lo, top * 0.97, f" lo {lo:.2f}", color="#b2182b",
            ha="left", va="top", fontsize=9)
    ax.text(hi, top * 0.97, f"hi {hi:.2f} ", color="#b2182b",
            ha="right", va="top", fontsize=9)

    ax.set_title(title, fontsize=13)
    caption = (f"n = {x.size:,}    k = {k:g}    "
               f"outside the band: {100 * outside:.2f}%")
    if subtitle:
        caption = f"{subtitle}\n{caption}"
    ax.set_xlabel(caption)
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return (f"  Wrote {path}\n"
            f"  The dashed line is what the band assumes; the filled shape is "
            f"what the data is. The gap between them is the non-normality.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_curve.py -v`
Expected: all PASS.

- [ ] **Step 5: Eyeball the output**

```bash
cd tier5_standalone && python -c "
import numpy as np
from tier5 import curve
x = np.random.default_rng(1).standard_t(4, 8000) * 20 - 10
print(curve.plot(x, centre=-10, scale=20, lo=-70, hi=50,
                 path='outputs/_curve_check.png', title='curve check'))
" && cd ..
```

Open the PNG. The KDE should be visibly more peaked with fatter ends than the
dashed normal. If they overlap exactly, the fitted-normal overlay is wrong.

- [ ] **Step 6: Commit**

```bash
git add tier5_standalone/tier5/curve.py tier5_standalone/tests/test_curve.py
git commit -m "Add tier5.curve: seaborn KDE against the fitted normal, with band bounds"
```

---

## Task 5: `tier5/fit.py` — year extract to band files

**Files:**
- Create: `tier5_standalone/tier5/fit.py`
- Test: `tier5_standalone/tests/test_fit.py`

**Interfaces:**
- Consumes: `cells.cells()`, `cells.period_label()`, `cells.band_path()`, `cells.out_dir()`, `persist.save()`, `curve.plot()`, `band.estimates()`, `band.classify()`
- Produces:
  - `fit_frame(df, cfg, *, bands_dir, out_dir, source_csv, force=False) -> list[dict]` — one result dict per cell with keys `region, strategy, n, lo, hi, centre, scale, flag_rate_pct, band_path, skipped, reason`
  - `main()` — CLI entry point

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_fit.py`:

```python
import json
import os

import numpy as np
import pandas as pd
import pytest

from tca import schema
from tier5 import cells, config as t5cfg, fit


def _book(n_per_cell=600, seed=4):
    rng = np.random.default_rng(seed)
    frames = []
    for region, strategy, mu, sd in [("HK", "VWAP", -10.0, 20.0),
                                     ("HK", "TWAP", -14.0, 24.0),
                                     ("JP", "VWAP", -8.0, 18.0)]:
        frames.append(pd.DataFrame({
            schema.MARKET: region,
            schema.ALGO: strategy,
            schema.SLIPPAGE_BPS: rng.normal(mu, sd, n_per_cell),
            schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n_per_cell),
            schema.PCT_ADV: rng.uniform(0.1, 5.0, n_per_cell),
            schema.VOLATILITY: rng.uniform(100.0, 250.0, n_per_cell),
            schema.DURATION_MIN: rng.uniform(10.0, 300.0, n_per_cell),
            schema.ORDER_DATE: pd.bdate_range("2025-06-02",
                                              periods=n_per_cell).astype(str),
        }))
    return pd.concat(frames, ignore_index=True)


def test_writes_one_band_per_cell(tmp_path):
    res = fit.fit_frame(_book(), t5cfg.CONFIG,
                        bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"),
                        source_csv="year.csv")
    written = {(r["region"], r["strategy"]) for r in res if not r["skipped"]}
    assert written == {("HK", "VWAP"), ("HK", "TWAP"), ("JP", "VWAP")}
    for region, strategy in written:
        assert os.path.exists(cells.band_path(str(tmp_path / "bands"),
                                              region, strategy))


def test_bands_differ_between_cells(tmp_path):
    res = fit.fit_frame(_book(), t5cfg.CONFIG,
                        bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"),
                        source_csv="year.csv")
    los = {(r["region"], r["strategy"]): r["lo"] for r in res}
    assert los[("HK", "TWAP")] != los[("HK", "VWAP")]


def test_band_file_records_region_strategy_and_window(tmp_path):
    fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    with open(cells.band_path(str(tmp_path / "bands"), "HK", "VWAP")) as fh:
        p = json.load(fh)
    assert p["region"] == "HK"
    assert p["strategy"] == "VWAP"
    assert p["fit_date_min"] == "2025-06-02"


def test_thin_cell_is_skipped_not_written(tmp_path):
    book = pd.concat([_book(), pd.DataFrame({
        schema.MARKET: "AU", schema.ALGO: "IS",
        schema.SLIPPAGE_BPS: np.random.default_rng(9).normal(-9, 19, 40),
        schema.SPREAD_BPS: 10.0, schema.PCT_ADV: 1.0,
        schema.VOLATILITY: 180.0, schema.DURATION_MIN: 60.0,
        schema.ORDER_DATE: "2025-07-01",
    })], ignore_index=True)
    res = fit.fit_frame(book, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    thin = [r for r in res if r["region"] == "AU"][0]
    assert thin["skipped"] is True
    assert "min_group_n" in thin["reason"]
    assert not os.path.exists(cells.band_path(str(tmp_path / "bands"), "AU", "IS"))


def test_force_writes_thin_cell(tmp_path):
    book = pd.DataFrame({
        schema.MARKET: "AU", schema.ALGO: "IS",
        schema.SLIPPAGE_BPS: np.random.default_rng(9).normal(-9, 19, 40),
        schema.SPREAD_BPS: 10.0, schema.PCT_ADV: 1.0,
        schema.VOLATILITY: 180.0, schema.DURATION_MIN: 60.0,
        schema.ORDER_DATE: "2025-07-01",
    })
    res = fit.fit_frame(book, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv",
                        force=True)
    assert res[0]["skipped"] is False
    assert os.path.exists(cells.band_path(str(tmp_path / "bands"), "AU", "IS"))


def test_flag_rate_is_in_sample_and_recorded(tmp_path):
    res = fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    for r in res:
        assert 0.0 <= r["flag_rate_pct"] <= 5.0


def test_curve_written_per_cell(tmp_path):
    fit.fit_frame(_book(), t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    d = cells.out_dir(str(tmp_path / "outputs"), "fit",
                      "2025-06_2027-09", "HK", "VWAP")
    # period is derived from the data; just assert some curve landed under fit/
    hits = []
    for root, _, files in os.walk(str(tmp_path / "outputs" / "fit")):
        hits += [f for f in files if f == "curve.png"]
    assert len(hits) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_fit.py -v`
Expected: FAIL — `No module named 'tier5.fit'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/fit.py`:

```python
"""Fit a Gaussian band per (region, strategy) and freeze it.

    python -m tier5.fit --csv extracts/year.csv

Region comes from the Sym suffix, strategy from the Strategy column and the
period from the Date column, so the same command works whether the twelve cells
arrive as one file or twelve.

This is the only module that computes a centre or a scale. Everything the
scoring side needs travels in the band file.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

from tca import dataset, report, schema
from tier5 import band, cells, config as t5cfg, curve, normality, persist


def fit_frame(df, cfg, *, bands_dir: str, out_dir: str, source_csv: str,
              force: bool = False) -> list[dict]:
    """Fit and freeze every cell in `df`. Returns one result dict per cell."""
    if cfg.metric not in df.columns:
        raise ValueError(f"Tier 5 needs column {cfg.metric!r}, which is absent.")

    results = []
    for region, strategy, g in cells.cells(df):
        period = cells.period_label(g) or "unknown-period"
        x = g[cfg.metric].to_numpy()
        est = band.estimates(x, cfg.k_sigma)
        e = cfg.estimator
        lo, hi = est[f"lo_{e}"], est[f"hi_{e}"]

        row = {"region": region, "strategy": strategy, "n": int(est["n"]),
               "period": period,
               "centre": est[f"centre_{e}"], "scale": est[f"scale_{e}"],
               "lo": lo, "hi": hi, "flag_rate_pct": float("nan"),
               "band_path": None, "skipped": False, "reason": ""}

        if est["n"] < cfg.min_group_n and not force:
            row["skipped"] = True
            row["reason"] = (f"n={est['n']} is below min_group_n="
                             f"{cfg.min_group_n}; a sigma from that many orders "
                             f"is not a threshold. Use --force to override.")
            results.append(row)
            continue

        finite = x[np.isfinite(x)]
        flag_rate = (100.0 * float(np.mean((finite < lo) | (finite > hi)))
                     if finite.size else float("nan"))
        row["flag_rate_pct"] = flag_rate

        path = cells.band_path(bands_dir, region, strategy)
        persist.save(est, cfg, path, region=region, strategy=strategy,
                     source_csv=source_csv, period=period, df=g,
                     flag_rate_pct=flag_rate)
        row["band_path"] = path

        cell_out = cells.out_dir(out_dir, "fit", period, region, strategy)
        os.makedirs(cell_out, exist_ok=True)
        normality.evidence(g, cfg).to_csv(
            os.path.join(cell_out, "normality.csv"), index=False)
        row["curve_msg"] = curve.plot(
            x, centre=row["centre"], scale=row["scale"], lo=lo, hi=hi,
            path=os.path.join(cell_out, "curve.png"),
            title=f"{region} / {strategy}  --  {cfg.metric}",
            subtitle=f"fitted on {period}", k=cfg.k_sigma)

        results.append(row)
    return results


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM])
    ap.add_argument("--k", type=float, help="Scales either side of the centre.")
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS))
    ap.add_argument("--bands-dir", default="bands",
                    help="Where band JSON files are written.")
    ap.add_argument("--out-dir", default="outputs",
                    help="Where curves and evidence are written.")
    ap.add_argument("--force", action="store_true",
                    help="Fit cells below min_group_n anyway.")
    args = ap.parse_args()

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.metric:
        overrides["metric"] = args.metric
    if args.k is not None:
        overrides["k_sigma"] = args.k
    if args.estimator:
        overrides["estimator"] = args.estimator
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 5 --- FIT AND FREEZE"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())
    print(f"\n  metric={cfg.metric}  k={cfg.k_sigma:g}  estimator={cfg.estimator}"
          f"  min_group_n={cfg.min_group_n}")

    results = fit_frame(df, cfg, bands_dir=args.bands_dir,
                        out_dir=args.out_dir,
                        source_csv=args.csv or "synthetic",
                        force=args.force)

    for r in results:
        print(f"\n  {r['region']} / {r['strategy']}   n = {r['n']:,}   {r['period']}")
        if r["skipped"]:
            print(f"    SKIPPED: {r['reason']}")
            continue
        print(f"    centre  {r['centre']:>9.2f}")
        print(f"    scale   {r['scale']:>9.2f}")
        print(f"    RANGE   {r['lo']:>9.2f} .. {r['hi']:.2f}"
              f"      <- frozen to {r['band_path']}")
        print(f"    in-sample flagged: {r['flag_rate_pct']:.2f}%")
        if r.get("curve_msg"):
            print(r["curve_msg"])

    n_ok = sum(1 for r in results if not r["skipped"])
    n_skip = len(results) - n_ok
    print(f"\nFroze {n_ok} band(s) to {args.bands_dir}/"
          + (f", skipped {n_skip}." if n_skip else "."))

    unknown = sorted({r["region"] for r in results
                      if r["region"] not in __import__("config").REGION_NAMES})
    if unknown:
        print(f"\n  Unrecognised region code(s): {unknown}. They were fitted "
              f"normally -- check the Sym suffix if that is unexpected.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_fit.py -v`
Expected: all PASS.

- [ ] **Step 5: Run it against the demo book**

```bash
cd tier5_standalone && python -m tier5.fit --n 6000 && cd ..
```

Expected: three cells (`HK/VWAP`, `HK/VWAP_Aggressive`, `HK/VWAP_Passive`), each
printing a RANGE and a band path, with a period label like `2025-06_2026-05`.

- [ ] **Step 6: Commit**

```bash
git add tier5_standalone/tier5/fit.py tier5_standalone/tests/test_fit.py
git commit -m "Add tier5.fit: freeze one Gaussian band per region and strategy"
```

---

## Task 6: `tier5/score.py` — apply frozen bands to a later period

**Files:**
- Create: `tier5_standalone/tier5/score.py`
- Test: `tier5_standalone/tests/test_score.py`

**Interfaces:**
- Consumes: `persist.load()`, `persist.drift_report()`, `cells.*`, `band.classify()`, `curve.plot()`
- Produces:
  - `score_frame(df, base_cfg, *, bands_dir, out_dir, label=None) -> list[dict]` — one result dict per cell with keys `region, strategy, n, n_flagged, flag_rate_pct, fit_flag_rate_pct, lo, hi, skipped, reason, out_dir`
  - `LeakageError` exception
  - `main()`

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_score.py`:

```python
import os

import numpy as np
import pandas as pd
import pytest

from tca import schema
from tier5 import cells, config as t5cfg, fit, score


def _cell(region, strategy, n, mu, sd, start, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        schema.ORDER_ID: [f"{region}{strategy}{i}" for i in range(n)],
        schema.MARKET: region,
        schema.ALGO: strategy,
        schema.SLIPPAGE_BPS: rng.normal(mu, sd, n),
        schema.SPREAD_BPS: rng.uniform(5.0, 15.0, n),
        schema.PCT_ADV: rng.uniform(0.1, 5.0, n),
        schema.VOLATILITY: rng.uniform(100.0, 250.0, n),
        schema.DURATION_MIN: rng.uniform(10.0, 300.0, n),
        schema.ORDER_DATE: pd.bdate_range(start, periods=n).astype(str),
    })


@pytest.fixture
def frozen(tmp_path):
    year = pd.concat([_cell("HK", "VWAP", 800, -10.0, 20.0, "2025-06-02", 1),
                      _cell("JP", "VWAP", 800, -8.0, 18.0, "2025-06-02", 2)],
                     ignore_index=True)
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    return tmp_path


def test_scores_against_frozen_band(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 7)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    r = res[0]
    assert r["skipped"] is False
    assert r["region"] == "HK" and r["strategy"] == "VWAP"
    assert 0 <= r["flag_rate_pct"] <= 100


def test_never_refits_bounds(frozen):
    """A far wider July must NOT widen the band."""
    import json
    with open(cells.band_path(str(frozen / "bands"), "HK", "VWAP")) as fh:
        frozen_lo = json.load(fh)["lo"]
    july = _cell("HK", "VWAP", 300, -10.0, 90.0, "2030-07-01", 8)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    assert res[0]["lo"] == frozen_lo
    assert res[0]["flag_rate_pct"] > 5.0   # wider data, same band -> more flags


def test_missing_band_is_skipped_not_borrowed(frozen):
    july = _cell("AU", "IS", 300, -10.0, 20.0, "2030-07-01", 9)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    assert res[0]["skipped"] is True
    assert "no band" in res[0]["reason"].lower()


def test_overlapping_window_is_refused(frozen):
    overlapping = _cell("HK", "VWAP", 300, -10.0, 20.0, "2025-06-02", 10)
    with pytest.raises(score.LeakageError, match="overlap"):
        score.score_frame(overlapping, t5cfg.CONFIG,
                          bands_dir=str(frozen / "bands"),
                          out_dir=str(frozen / "outputs"))


def test_outliers_csv_written_and_sorted(frozen):
    july = _cell("HK", "VWAP", 400, -10.0, 45.0, "2030-07-01", 11)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    path = os.path.join(res[0]["out_dir"], "outliers.csv")
    assert os.path.exists(path)
    out = pd.read_csv(path)
    assert len(out) == res[0]["n_flagged"]
    assert (out["n_sigma_outside"] > 0).all()
    assert out["n_sigma_outside"].is_monotonic_decreasing


def test_scored_csv_has_every_row(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 12)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"))
    scored = pd.read_csv(os.path.join(res[0]["out_dir"], "scored.csv"))
    assert len(scored) == 300


def test_label_overrides_period_folder(frozen):
    july = _cell("HK", "VWAP", 300, -10.0, 20.0, "2030-07-01", 13)
    res = score.score_frame(july, t5cfg.CONFIG,
                            bands_dir=str(frozen / "bands"),
                            out_dir=str(frozen / "outputs"),
                            label="my-label")
    assert "my-label" in res[0]["out_dir"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_score.py -v`
Expected: FAIL — `No module named 'tier5.score'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/score.py`:

```python
"""Apply frozen bands to a later period.

    python -m tier5.score --csv extracts/july.csv

This module NEVER fits. It reads lo/hi out of a band file and classifies against
them unchanged, which is what turns the flag rate from a definition into a
measurement. If a cell has no band file it is skipped and reported -- scoring
HK VWAP orders against the Japan TWAP band would produce plausible-looking
numbers with no warning, which is worse than no answer.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from tca import dataset, report, schema
from tier5 import band, cells, config as t5cfg, curve, persist


class LeakageError(RuntimeError):
    """The scoring window overlaps the window the band was fitted on."""


# Written to outliers.csv when present in the extract. These are the inputs to
# the analyst's explanation -- this module deliberately does not attribute cause.
DIAGNOSTIC_COLS = [schema.NOTIONAL, schema.SPREAD_BPS, schema.PCT_ADV,
                   schema.PARTICIPATION, schema.DURATION_MIN,
                   schema.PASSIVE_FILL_PCT, schema.AUCTION_PCT,
                   schema.REVERSION_BPS, schema.MOMENTUM_BPS]

IDENT_COLS = [schema.ORDER_ID, schema.SYMBOL, schema.SIDE, schema.ORDER_DATE]


def _n_sigma_outside(x, lo, hi, scale):
    """How far outside, in scales. Always positive; 0 inside the band."""
    above = (x - hi) / scale
    below = (lo - x) / scale
    return np.maximum(0.0, np.maximum(above, below))


def score_frame(df, base_cfg, *, bands_dir: str, out_dir: str,
                label: str | None = None) -> list[dict]:
    """Score every cell in `df` against its frozen band."""
    results = []
    for region, strategy, g in cells.cells(df):
        period = label or cells.period_label(g) or "score"
        path = cells.band_path(bands_dir, region, strategy)
        row = {"region": region, "strategy": strategy, "n": int(len(g)),
               "period": period, "n_flagged": 0,
               "flag_rate_pct": float("nan"), "fit_flag_rate_pct": float("nan"),
               "lo": float("nan"), "hi": float("nan"),
               "skipped": False, "reason": "", "out_dir": None,
               "drift_table": None, "drift_warnings": []}

        if not os.path.exists(path):
            row["skipped"] = True
            row["reason"] = (f"no band at {path} -- fit this cell first. "
                             f"Not scored against another cell's band.")
            results.append(row)
            continue

        frozen, cfg, reference = persist.load(path, base_cfg)

        b_lo, b_hi = cells.date_range(g)
        f_lo = pd.Timestamp(frozen["fit_date_min"]) if frozen["fit_date_min"] else None
        f_hi = pd.Timestamp(frozen["fit_date_max"]) if frozen["fit_date_max"] else None
        if cells.windows_overlap(f_lo, f_hi, b_lo, b_hi):
            raise LeakageError(
                f"{region}/{strategy}: the scoring window "
                f"{b_lo.date()}..{b_hi.date()} overlaps the window the band was "
                f"fitted on ({f_lo.date()}..{f_hi.date()}). Scoring a period the "
                f"band already saw makes the flag rate circular, which is the "
                f"one thing this workflow exists to avoid.")

        if cfg.metric not in g.columns:
            row["skipped"] = True
            row["reason"] = f"extract has no column {cfg.metric!r}"
            results.append(row)
            continue

        lo, hi = float(frozen["lo"]), float(frozen["hi"])
        centre, scale = float(frozen["centre"]), float(frozen["scale"])
        x = pd.to_numeric(g[cfg.metric], errors="coerce").to_numpy(dtype=float)

        scored = g.copy()
        scored["zone"] = [band.classify(v, lo, hi) for v in x]
        scored["band_lo"] = lo
        scored["band_hi"] = hi
        scored["band_centre"] = centre
        scored["band_scale"] = scale
        scored["n_sigma_outside"] = _n_sigma_outside(x, lo, hi, scale)
        scored["flagged"] = scored["zone"].isin(list(band.FLAGGED))

        row["lo"], row["hi"] = lo, hi
        row["n_flagged"] = int(scored["flagged"].sum())
        row["flag_rate_pct"] = 100.0 * float(scored["flagged"].mean())
        row["fit_flag_rate_pct"] = reference.get("flag_rate_pct", float("nan"))

        cell_out = cells.out_dir(out_dir, "score", period, region, strategy)
        os.makedirs(cell_out, exist_ok=True)
        row["out_dir"] = cell_out

        keep = [c for c in IDENT_COLS + ["zone", cfg.metric, schema.SLIPPAGE_BPS,
                                         "band_lo", "band_hi", "n_sigma_outside"]
                + DIAGNOSTIC_COLS if c in scored.columns]
        keep = list(dict.fromkeys(keep))

        scored.to_csv(os.path.join(cell_out, "scored.csv"), index=False)
        (scored[scored["flagged"]][keep]
         .sort_values("n_sigma_outside", ascending=False)
         .to_csv(os.path.join(cell_out, "outliers.csv"), index=False))

        table, warnings = persist.drift_report(g, scored, reference, cfg)
        row["drift_table"], row["drift_warnings"] = table, warnings

        row["curve_msg"] = curve.plot(
            x, centre=centre, scale=scale, lo=lo, hi=hi,
            path=os.path.join(cell_out, "curve.png"),
            title=f"{region} / {strategy}  --  {period} vs frozen band",
            subtitle=f"band frozen on {frozen.get('fit_period')}",
            k=float(frozen["k_sigma"]))

        results.append(row)
    return results


def main():
    ap = dataset.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--bands-dir", default="bands")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--label", default=None,
                    help="Name the period folder. Defaults to the Date range.")
    args = ap.parse_args()

    df, clean_report = dataset.load_prepared(args)

    print(report.header("TIER 5 --- SCORE AGAINST FROZEN BANDS"))
    print("\n=== Cleaning ===")
    print(clean_report.as_text())

    results = score_frame(df, t5cfg.CONFIG, bands_dir=args.bands_dir,
                          out_dir=args.out_dir, label=args.label)

    for r in results:
        print(f"\n  {r['region']} / {r['strategy']}")
        if r["skipped"]:
            print(f"    SKIPPED: {r['reason']}")
            continue
        ratio = (r["flag_rate_pct"] / r["fit_flag_rate_pct"]
                 if r["fit_flag_rate_pct"] else float("nan"))
        print(f"    band        {r['lo']:.2f} .. {r['hi']:.2f}   (frozen)")
        print(f"    fit book:   {r['fit_flag_rate_pct']:.2f}% outside")
        print(f"    {r['period']}:    {r['n']:,} orders, {r['n_flagged']} flagged, "
              f"{r['flag_rate_pct']:.2f}% outside"
              + (f"   <- {ratio:.1f}x" if np.isfinite(ratio) else ""))
        print(f"    wrote {r['out_dir']}")
        if r["drift_warnings"]:
            print("\n    Drift:")
            for w in r["drift_warnings"]:
                print(f"      - {w}")

    n_ok = sum(1 for r in results if not r["skipped"])
    print(f"\nScored {n_ok} cell(s). {len(results) - n_ok} skipped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_score.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tier5_standalone/tier5/score.py tier5_standalone/tests/test_score.py
git commit -m "Add tier5.score: apply frozen bands out-of-sample, never refit"
```

---

## Task 7: `tier5/batch.py` — walk a directory of extracts

**Files:**
- Create: `tier5_standalone/tier5/batch.py`
- Test: `tier5_standalone/tests/test_batch.py`

**Interfaces:**
- Consumes: `fit.fit_frame()`, `score.score_frame()`
- Produces:
  - `run(mode, directory, *, bands_dir, out_dir, cfg, label=None) -> tuple[list[dict], list[dict]]` — `(results, failures)`
  - `summary_frame(results, mode) -> pd.DataFrame`
  - `main()`

- [ ] **Step 1: Write the failing test**

Create `tier5_standalone/tests/test_batch.py`:

```python
import os

import numpy as np
import pandas as pd

from tca import schema
from tier5 import batch, config as t5cfg


def _write(path, region, strategy, n, mu, sd, start, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    pd.DataFrame({
        "aggrTgtId": [f"{region}{i}" for i in range(n)],
        "Sym": f"0001 {region}",
        "Strategy": strategy,
        "Pvwap": rng.normal(mu, sd, n),
        "Sprd": rng.uniform(5.0, 15.0, n),
        "%Adv": rng.uniform(0.1, 5.0, n),
        "Vol": rng.uniform(1.0, 2.5, n),
        "PR": rng.uniform(1.0, 20.0, n),
        "Dur": rng.uniform(10.0, 300.0, n),
        "Date": pd.bdate_range(start, periods=n).astype(str),
    }).to_csv(path, index=False)


def test_fit_then_score_across_a_directory(tmp_path):
    for region in ("HK", "JP"):
        _write(str(tmp_path / "year" / region / "VWAP.csv"),
               region, "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    results, failures = batch.run("fit", str(tmp_path / "year"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert failures == []
    assert len({(r["region"], r["strategy"]) for r in results}) == 2

    for region in ("HK", "JP"):
        _write(str(tmp_path / "july" / region / "VWAP.csv"),
               region, "VWAP", 200, -10.0, 20.0, "2030-07-01", 2)
    results, failures = batch.run("score", str(tmp_path / "july"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert failures == []
    assert all(not r["skipped"] for r in results)


def test_one_broken_file_does_not_abort_the_rest(tmp_path):
    _write(str(tmp_path / "year" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    broken = tmp_path / "year" / "JP" / "VWAP.csv"
    os.makedirs(os.path.dirname(str(broken)), exist_ok=True)
    broken.write_text("this,is,not\na,valid,extract\n", encoding="utf-8")

    results, failures = batch.run("fit", str(tmp_path / "year"),
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"),
                                  cfg=t5cfg.CONFIG)
    assert len(failures) == 1
    assert "VWAP.csv" in failures[0]["file"]
    assert any(r["region"] == "HK" for r in results)


def test_summary_frame_has_a_row_per_cell(tmp_path):
    for region in ("HK", "JP"):
        _write(str(tmp_path / "year" / region / "VWAP.csv"),
               region, "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    results, _ = batch.run("fit", str(tmp_path / "year"),
                           bands_dir=str(tmp_path / "bands"),
                           out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    summary = batch.summary_frame(results, "fit")
    assert len(summary) == 2
    assert {"region", "strategy", "n", "lo", "hi"} <= set(summary.columns)


def test_summary_written_for_score(tmp_path):
    _write(str(tmp_path / "year" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 600, -10.0, 20.0, "2025-06-02", 1)
    batch.run("fit", str(tmp_path / "year"), bands_dir=str(tmp_path / "bands"),
              out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG)
    _write(str(tmp_path / "july" / "HK" / "VWAP.csv"),
           "HK", "VWAP", 200, -10.0, 20.0, "2030-07-01", 2)
    batch.run("score", str(tmp_path / "july"), bands_dir=str(tmp_path / "bands"),
              out_dir=str(tmp_path / "outputs"), cfg=t5cfg.CONFIG,
              label="2030-07")
    assert os.path.exists(str(tmp_path / "outputs" / "score" / "2030-07" /
                              "_summary.csv"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tier5_standalone/tests/test_batch.py -v`
Expected: FAIL — `No module named 'tier5.batch'`

- [ ] **Step 3: Implement**

Create `tier5_standalone/tier5/batch.py`:

```python
"""Run fit or score over a whole directory of extracts.

    python -m tier5.batch fit   --dir extracts/year
    python -m tier5.batch score --dir extracts/2026-07 --label 2026-07

Only needed when the cells arrive as separate files. Region, strategy and
period still come from the data rather than the filename, so the directory can
be organised however you like -- this just walks it.

One file failing does not abort the rest. With twelve cells, a single malformed
export should cost you that cell, not the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os

import pandas as pd

import config
from tca import pipeline, report, schema
from tier5 import config as t5cfg, fit, score


def _load(path):
    raw = pd.read_csv(path)
    df, _ = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                             config.SLIPPAGE_SIGN,
                             pre_transform=getattr(config, "PRE_TRANSFORM", None))
    return df


def run(mode: str, directory: str, *, bands_dir: str, out_dir: str, cfg,
        label: str | None = None, force: bool = False):
    """Walk `directory` for CSVs and fit or score each. Returns (results, failures)."""
    if mode not in ("fit", "score"):
        raise ValueError(f"mode must be 'fit' or 'score', got {mode!r}")

    paths = sorted(glob.glob(os.path.join(directory, "**", "*.csv"),
                             recursive=True))
    results, failures = [], []
    for path in paths:
        try:
            df = _load(path)
            if mode == "fit":
                results += fit.fit_frame(df, cfg, bands_dir=bands_dir,
                                         out_dir=out_dir, source_csv=path,
                                         force=force)
            else:
                results += score.score_frame(df, cfg, bands_dir=bands_dir,
                                             out_dir=out_dir, label=label)
        except Exception as exc:                      # noqa: BLE001
            failures.append({"file": path, "error": f"{type(exc).__name__}: {exc}"})
    return results, failures


def summary_frame(results: list[dict], mode: str) -> pd.DataFrame:
    """One row per cell -- the only place all twelve are comparable."""
    rows = []
    for r in results:
        row = {"region": r["region"], "strategy": r["strategy"],
               "n": r["n"], "lo": r["lo"], "hi": r["hi"],
               "skipped": r["skipped"], "reason": r["reason"]}
        if mode == "fit":
            row["flag_pct"] = r.get("flag_rate_pct")
        else:
            row["n_flagged"] = r.get("n_flagged")
            row["flag_pct"] = r.get("flag_rate_pct")
            row["fit_flag_pct"] = r.get("fit_flag_rate_pct")
            fit_pct = r.get("fit_flag_rate_pct") or 0.0
            row["vs_fit"] = (round(r["flag_rate_pct"] / fit_pct, 2)
                             if fit_pct else None)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["region", "strategy"]).reset_index(drop=True)
        for col in ("lo", "hi", "flag_pct", "fit_flag_pct"):
            if col in df.columns:
                df[col] = df[col].round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "score"])
    ap.add_argument("--dir", required=True, help="Directory of CSV extracts.")
    ap.add_argument("--bands-dir", default="bands")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--label", default=None)
    ap.add_argument("--k", type=float)
    ap.add_argument("--metric", choices=[schema.SLIPPAGE_BPS,
                                         schema.PERF_IN_SPREADS,
                                         schema.PERF_NORM])
    ap.add_argument("--estimator", choices=list(t5cfg.ESTIMATORS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = t5cfg.CONFIG
    overrides = {}
    if args.k is not None:
        overrides["k_sigma"] = args.k
    if args.metric:
        overrides["metric"] = args.metric
    if args.estimator:
        overrides["estimator"] = args.estimator
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    print(report.header(f"TIER 5 --- BATCH {args.mode.upper()}"))
    results, failures = run(args.mode, args.dir, bands_dir=args.bands_dir,
                            out_dir=args.out_dir, cfg=cfg, label=args.label,
                            force=args.force)

    summary = summary_frame(results, args.mode)
    print("\n=== Summary ===")
    print(report.frame(summary, max_rows=60))

    period = args.label
    if period is None:
        periods = {r.get("period") for r in results if r.get("period")}
        period = periods.pop() if len(periods) == 1 else "mixed"
    dest = os.path.join(args.out_dir, args.mode, period, "_summary.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    summary.to_csv(dest, index=False)
    print(f"\nWrote {dest}")

    if failures:
        print(f"\n=== {len(failures)} file(s) failed ===")
        for f in failures:
            print(f"  {f['file']}\n    {f['error']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tier5_standalone/tests/test_batch.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tier5_standalone/tier5/batch.py tier5_standalone/tests/test_batch.py
git commit -m "Add tier5.batch: fit or score a whole directory, one failure at a time"
```

---

## Task 8: End-to-end proof and the README

**Files:**
- Create: `tier5_standalone/README.md`
- Create: `tier5_standalone/tests/test_end_to_end.py`
- Modify: `tier5_standalone/requirements.txt` (drop statsmodels, keep seaborn)

**Interfaces:**
- Consumes: everything
- Produces: a documented, verified folder

- [ ] **Step 1: Write the end-to-end test**

Create `tier5_standalone/tests/test_end_to_end.py`:

```python
"""The round trip: freeze on one book, score another, prove nothing leaked."""

import json
import os

import numpy as np
import pandas as pd

import config
import synthetic_data
from tca import pipeline
from tier5 import cells, config as t5cfg, fit, persist, score


def _prep(raw):
    df, _ = pipeline.prepare(raw, config.COLUMN_MAP, config.DATA,
                             config.SLIPPAGE_SIGN,
                             pre_transform=config.PRE_TRANSFORM)
    return df


def test_full_round_trip(tmp_path):
    year = _prep(synthetic_data.generate(n=6000, seed=7,
                                         start_date="2025-06-02",
                                         end_date="2026-05-29"))
    later = _prep(synthetic_data.generate(n=1200, seed=8,
                                          start_date="2026-07-01",
                                          end_date="2026-07-31"))

    fit_res = fit.fit_frame(year, t5cfg.CONFIG,
                            bands_dir=str(tmp_path / "bands"),
                            out_dir=str(tmp_path / "outputs"),
                            source_csv="year.csv")
    assert any(not r["skipped"] for r in fit_res)

    score_res = score.score_frame(later, t5cfg.CONFIG,
                                  bands_dir=str(tmp_path / "bands"),
                                  out_dir=str(tmp_path / "outputs"))
    scored_cells = [r for r in score_res if not r["skipped"]]
    assert scored_cells

    # The band that scored July is byte-identical to the one fit froze.
    for r in scored_cells:
        with open(cells.band_path(str(tmp_path / "bands"),
                                  r["region"], r["strategy"])) as fh:
            frozen = json.load(fh)
        assert r["lo"] == frozen["lo"]
        assert r["hi"] == frozen["hi"]


def test_scoring_the_fit_book_itself_is_refused(tmp_path):
    year = _prep(synthetic_data.generate(n=3000, seed=7))
    fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                  out_dir=str(tmp_path / "outputs"), source_csv="year.csv")
    import pytest
    with pytest.raises(score.LeakageError):
        score.score_frame(year, t5cfg.CONFIG,
                          bands_dir=str(tmp_path / "bands"),
                          out_dir=str(tmp_path / "outputs"))


def test_in_sample_flag_rate_matches_when_leakage_check_is_bypassed(tmp_path):
    """fit's recorded rate and a manual classify of the same rows agree."""
    year = _prep(synthetic_data.generate(n=4000, seed=7))
    res = fit.fit_frame(year, t5cfg.CONFIG, bands_dir=str(tmp_path / "bands"),
                        out_dir=str(tmp_path / "outputs"), source_csv="y.csv")
    from tca import schema
    for r in res:
        if r["skipped"]:
            continue
        g = year[(year[schema.MARKET] == r["region"])
                 & (year[schema.ALGO] == r["strategy"])]
        x = g[t5cfg.CONFIG.metric].to_numpy()
        manual = 100.0 * float(np.mean((x < r["lo"]) | (x > r["hi"])))
        assert abs(manual - r["flag_rate_pct"]) < 1e-9
```

- [ ] **Step 2: Run to verify it fails, then passes**

Run: `python -m pytest tier5_standalone/tests/test_end_to_end.py -v`

If `test_scoring_the_fit_book_itself_is_refused` does not raise, the leakage
check is broken — fix `score.score_frame` before continuing. This test is the
whole point of the design.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest tier5_standalone/tests/ -v`
Expected: every test passes.

- [ ] **Step 4: Trim requirements.txt**

Replace `tier5_standalone/requirements.txt` with:

```
# Shared pipeline + the Gaussian band
pandas>=2.0
numpy>=1.24

# D'Agostino K2 normality test and the QQ plot
scipy>=1.10

# The Gaussian curve (tier5/curve.py) and the QQ plot. Nothing in the scoring
# path imports either, so the folder still fits and scores without them.
matplotlib>=3.7
seaborn>=0.13
```

statsmodels is dropped: it was tier3's quantile-regression backend and nothing
here uses it.

- [ ] **Step 5: Write the README**

Create `tier5_standalone/README.md` covering, in this order:

1. **What this folder is** — one paragraph: a Gaussian band frozen on a year of
   one region and strategy, applied unchanged to a later month.
2. **Install** — `pip install -r requirements.txt`.
3. **First run, no data needed** — `python -m tier5.run --self-check`, then
   `python -m tier5.fit --n 6000` to see it work on the synthetic book.
4. **Pointing it at real data** — run `python check_extract.py your.csv`, then
   edit `config.py`'s `COLUMN_MAP`. State that `Date` must be present for period
   labelling and the leakage check, and that `Sym`'s last two characters become
   the region.
5. **The two-step workflow** — the exact commands:

```bash
# 1. Fit the year and freeze the bands
python -m tier5.fit --csv extracts/year.csv

# 2. Score July against them
python -m tier5.score --csv extracts/july.csv --label 2026-07

# Or, if the cells arrive as separate files:
python -m tier5.batch fit   --dir extracts/year/
python -m tier5.batch score --dir extracts/july/ --label 2026-07
```

6. **What you get** — the `bands/` and `outputs/` trees from the spec, and what
   each file contains. Name `outliers.csv` as the one to open first.
7. **Reading `outliers.csv`** — explain `n_sigma_outside`, and that the
   diagnostic columns are there to support the analyst's explanation; this tool
   does not attribute cause.
8. **The flags** — a table of every flag on `fit`, `score` and `batch`.
9. **Choosing `k`** — reproduce the guidance from `tier5_gaussian/USAGE.md`:
   `k=3` promises 0.27% and delivers roughly 6x that on a real book; `k_required`
   in each band file says what would have delivered the nominal rate.
10. **Known limits** — symmetric band vs skewed slippage; the band decays, so
    refit when the drift report says the feature medians moved; thin cells are
    skipped rather than guessed at.

- [ ] **Step 6: Verify the README's commands actually run**

```bash
cd tier5_standalone
python -m tier5.run --self-check
python -m tier5.fit --n 6000 --bands-dir /tmp/b --out-dir /tmp/o
cd ..
```

Every command in the README must have been executed once before commit.

- [ ] **Step 7: Commit**

```bash
git add tier5_standalone/
git commit -m "Add tier5_standalone README, end-to-end tests and trimmed requirements"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Standalone folder, tier5 only, no tier3 | 1 |
| Package renamed `tier5`, imports rewritten | 1 |
| `evaluate.py` not copied, run.py block removed | 1 |
| `schema.ORDER_DATE` added, optional | 1 |
| `Date` mapped in COLUMN_MAP | 1 |
| `REGION_NAMES`, no aliasing | 1 |
| synthetic_data emits `Date` | 1 |
| Region from Sym suffix, strategy from Strategy | 2 |
| Period label from date min/max | 2 |
| Nested `<REGION>/<STRATEGY>/` paths | 2 |
| Overlap detection | 2 (helper), 6 (enforcement) |
| Band JSON payload, both estimators, FORMAT_VERSION | 3 |
| `k_required` in the reference | 3 |
| drift_report | 3 |
| Seaborn KDE + fitted normal + band lines + shading | 4 |
| Graceful degradation without matplotlib/seaborn | 4 |
| fit groups by cell, one band each | 5 |
| `min_group_n` refusal with `--force` | 5 |
| Unrecognised regions reported not dropped | 5 |
| score never refits | 6 |
| Missing band skipped, not borrowed | 6 |
| Leakage refusal | 6, 8 |
| `outliers.csv` sorted by `n_sigma_outside`, diagnostics carried | 6 |
| `--label` overrides period | 6 |
| batch walks a directory, one failure isolated | 7 |
| `_summary.csv` cross-cell table | 7 |
| README | 8 |
| Round-trip test | 8 |

**Placeholder scan:** none — every code step carries the actual implementation.

**Type consistency:** `fit_frame` and `score_frame` both return `list[dict]`;
`batch.summary_frame` reads only keys both produce (`region`, `strategy`, `n`,
`lo`, `hi`, `skipped`, `reason`) plus mode-specific keys guarded by the `mode`
branch. `cells.band_path` is the single constructor of a band path, used
identically in Tasks 5, 6 and 7. `curve.plot` has one signature, called from
Tasks 5 and 6 with the same keywords.
