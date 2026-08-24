# Tier 5 standalone: freeze a band on a year, score the next month

**Date:** 2026-08-24
**Status:** approved (revised after feedback), not yet implemented

## Problem

Tier 5 today fits a Gaussian band and scores orders in a single run, on the same
book. That makes the flag rate a definition rather than a measurement: 1.7% of
orders are outside the band partly because the band was dragged toward the very
outliers it then counts. The current README already says so.

The intended operational use is different, and out-of-sample:

1. Take one year of one strategy in one region (e.g. HK VWAP, Jun 2025 - Jun 2026).
2. Fit the acceptable range from that year: `lo`, `hi`.
3. Freeze it.
4. In July, flag the orders that fall outside the frozen range, and explain them.

Four regions (Australia, Hong Kong, Japan, India) x three strategies = 12 cells.

This must also run as a **standalone folder** the user copies to a target
machine, containing tier 5 only -- no tier3, no comparison driver.

## Decisions

| Question | Decision | Why |
|---|---|---|
| How the two windows are split | **Two separate CSV files.** `fit` reads the year file, `score` reads the July file. | The user exports per-period already. Dates are read for *labelling and verification*, never for slicing. |
| Region | **Derived from the `Sym` suffix.** | `config.PRE_TRANSFORM` already computes `_market = Sym[-2:]` -> `schema.MARKET`. It is free. |
| Strategy | **Derived from the `Strategy` column** (`schema.ALGO`). | Already mapped. |
| Period label | **Derived from the date column's min/max.** | User requirement: read the date to know the interval. |
| How the 12 cells arrive | **Either way.** `fit` groups by (region, strategy) found in the file. | Since both are read from the data, one combined file and twelve separate files both work with the same command. The question dissolves. |
| Cause attribution in `score` | **Not built.** Emit diagnostic columns, let the analyst explain. | Automated cause attribution is `tier3_model/diagnostics.py`. Pulling it in would defeat "standalone". |
| Default `k` | **3.0** | The user's stated method. Out-of-sample the resulting flag rate is an honest measurement. `--k` tunes it. |
| Output organisation | **Nested by region then strategy.** | User requirement: bands and outputs must not mix into one flat pile. |

### Superseded

An earlier draft passed `--region` and `--strategy` as CLI labels and required
one file per cell. Both are replaced by deriving the values from the data. The
CLI flags survive only as overrides for extracts whose columns are unusable.

## Schema change: a date column

`tca/schema.py` has no date field today and `COLUMN_MAP` maps none. One is added:

```python
ORDER_DATE = "order_date"     # order start date; drives period labelling only
```

Mapped in `config.py`, parsed with `pd.to_datetime(errors="coerce")`, and used
for three things and nothing else:

1. **Naming the period folder** -- `2025-06_2026-05` from min/max.
2. **Stamping the fit window into the band file**, so a band is self-describing.
3. **Refusing to score a file that overlaps the fit window**, which is leakage.

It is **optional**. If the column is absent or unmapped, the period falls back
to `--label` (default `score`) and the overlap check is skipped with a printed
notice. The metric, the band and the scoring path never touch it, so a missing
date degrades labelling and nothing else.

The source column name is extract-specific. `config.py` ships with a documented
candidate list (`starttime`, `StartTime`, `TradeDate`, `Date`) and
`check_extract.py` prints which of them it found, so the user sets it once.

## Region aliasing

Bloomberg uses two suffixes for India: `IN` (NSE) and `IS` (BSE). Left alone,
that silently splits India into two half-sized cells, each with a different band
and neither matching what the user asked for. `config.py` gains:

```python
REGION_ALIASES = {"IS": "IN"}     # fold BSE into India
REGION_NAMES = {"AU": "Australia", "HK": "Hong Kong",
                "JT": "Japan", "IN": "India"}
```

`REGION_ALIASES` is applied to `schema.MARKET` during preparation.
`REGION_NAMES` is presentation only -- folders use the two-letter code.

Any suffix not in `REGION_NAMES` still fits normally; it is reported in the
summary as an unrecognised region rather than dropped, so a stray venue shows up
instead of vanishing.

## Folder layout

The deliverable is a directory copied wholesale. Nothing outside it is imported.

```
tier5_standalone/
  README.md              usage doc
  requirements.txt       pandas, numpy, scipy, matplotlib, seaborn
  config.py              COLUMN_MAP / units / SLIPPAGE_SIGN / region aliases
  check_extract.py       run first on any new extract
  synthetic_data.py      lets the folder prove itself before real data goes in
  tca/
    __init__.py  schema.py  pipeline.py  report.py  dataset.py
  tier5/
    __init__.py  config.py  band.py  normality.py   (copied, imports rewritten)
    curve.py      NEW
    persist.py    NEW
    fit.py        NEW
    score.py      NEW
    batch.py      NEW
    run.py              existing single-run driver, kept for exploration
```

The package is renamed `tier5_gaussian` -> `tier5` inside the standalone folder.

`tca/evaluate.py` is **not** copied: it only scores synthetic ground-truth
labels, which real extracts never carry. `run.py`'s call into it is removed.

## Directory conventions

Everything nests `<REGION>/<STRATEGY>/` so twelve cells never mix.

```
bands/
  HK/VWAP.json                      the frozen range
  HK/VWAP.json is written by fit, read by score, never modified

outputs/
  fit/2025-06_2026-05/HK/VWAP/      curve.png, normality.csv
  score/2026-07/HK/VWAP/            outliers.csv, scored.csv, curve.png
  score/2026-07/_summary.csv        cross-cell comparison
```

Both period segments are derived from the data's date range. Successive months
accumulate side by side instead of overwriting.

## Data flow

```
year.csv --> fit.py --> for each (region, strategy) present:
                          bands/<R>/<S>.json
                          outputs/fit/<period>/<R>/<S>/curve.png
                          outputs/fit/<period>/<R>/<S>/normality.csv
                                 |
july.csv ------------------------+--> score.py --> outputs/score/<period>/<R>/<S>/outliers.csv
                                                   outputs/score/<period>/<R>/<S>/scored.csv
                                                   outputs/score/<period>/<R>/<S>/curve.png
                                                   drift report (stdout)
```

`fit.py` is the only module that computes a centre or a scale. `score.py` loads
`lo`/`hi` and applies them unchanged; it must never refit. This is the single
invariant the whole design exists to enforce.

## Module specs

### `tier5/cells.py`

Small shared helper, so fit and score derive cells identically:

```python
def cells(df) -> list[tuple[str, str, pd.DataFrame]]   # (region, strategy, rows)
def period_label(df) -> str        # "2025-06_2026-05", or None if no dates
def band_path(bands_dir, region, strategy) -> str
def out_dir(root, kind, period, region, strategy) -> str
```

Keeping this in one place is what guarantees a band written for `HK/VWAP` is the
one `score` looks up for those same rows.

### `tier5/persist.py`

Mirrors `tier3_model/persist.py`, which already solves this problem for the cost
model. Same shape, much smaller payload.

```python
FORMAT_VERSION = 1

def save(estimates, cfg, path, *, region, strategy, source_csv, period, df) -> str
def load(path, base_cfg) -> tuple[dict, Tier5Config, dict]   # band, cfg, reference
def drift_report(df, scored, reference, cfg) -> tuple[pd.DataFrame, list[str]]
```

Band file payload:

```json
{
  "format_version": 1,
  "fitted_at": "2026-08-24T10:31:00",
  "region": "HK",
  "strategy": "VWAP",
  "source_csv": "extracts/year.csv",
  "fit_period": "2025-06_2026-05",
  "fit_date_min": "2025-06-02",
  "fit_date_max": "2026-05-29",
  "metric": "slippage_bps",
  "estimator": "classical",
  "k_sigma": 3.0,
  "n": 8957,
  "centre": -9.74, "scale": 18.67, "lo": -65.74, "hi": 46.27,
  "centre_robust": -8.60, "scale_robust": 14.37,
  "lo_robust": -51.70, "hi_robust": 34.50,
  "reference": {
    "flag_rate_pct": 1.45,
    "skew": -0.39, "excess_kurtosis": 3.90,
    "k_required": 4.61,
    "feature_medians": {"spread_bps": 9.8, "pct_adv": 1.4,
                        "volatility": 181.0, "duration_min": 54.0}
  }
}
```

Both estimators are stored even though only one scores, so switching estimator
later does not require a refit. `k_required` is `normality.required_k`'s
`k_symmetric` on the fit year -- at score time it tells the user what k would
have delivered the nominal rate on that book.

`load` raises on a `format_version` mismatch rather than guessing, matching
tier3's behaviour.

### `tier5/fit.py`

```
python -m tier5.fit --csv extracts/year.csv
                    [--k 3.0] [--metric slippage_bps] [--estimator classical]
                    [--bands-dir bands/] [--out-dir outputs/]
                    [--region HK] [--strategy VWAP]   # overrides, rarely needed
                    [--force]
```

Steps: `dataset.load_prepared` -> `cells.cells(df)` -> for each cell,
`band.estimates` -> `normality` evidence -> `curve.plot` -> `persist.save`.

Prints one block per cell, then a summary:

```
  HK / VWAP   n = 8,957   2025-06 .. 2026-05
    centre    -9.74
    scale     18.67
    RANGE    -65.74 .. 46.27      <- frozen to bands/HK/VWAP.json
```

Refuses to write a band when `n < cfg.min_group_n` (200) unless `--force`, and
says why. A sigma from 44 orders is not a threshold. Skipped cells are listed,
not silently dropped.

### `tier5/score.py`

```
python -m tier5.score --csv extracts/july.csv
                      [--bands-dir bands/] [--label 2026-07] [--out-dir outputs/]
```

Steps: derive cells from the new file -> for each, load `bands/<R>/<S>.json` ->
classify against the frozen `lo`/`hi` via `band.classify` -> write scored +
outliers -> `drift_report` -> curve of the new period against the frozen band.

A cell present in the extract with no matching band file is **reported and
skipped**, never scored against some other cell's band.

If both files carry dates and the score window overlaps the band's fit window,
`score` refuses with the overlapping range printed. That is leakage, and it is
the mistake that makes the whole exercise meaningless.

`outliers.csv` is sorted by `n_sigma_outside` descending and carries every
diagnostic column present in the extract:

`order_id, symbol, side, order_date, zone, slippage_bps, band_lo, band_hi,
n_sigma_outside, notional, spread_bps, pct_adv, participation, duration_min,
passive_fill_pct, auction_pct, reversion_bps`

`n_sigma_outside` is `(x - hi)/scale` above the band and `(lo - x)/scale` below,
so it is always positive and directly comparable across cells and regions.

Stdout reports the headline comparison, which is the actual finding:

```
  HK / VWAP
    fitted on 8,957 orders (2025-06..2026-05): 1.45% outside
    2026-07:    812 orders,                    2.34% outside   <- 1.6x
```

Drift warnings fire when a feature median moves more than 25% or the flag rate
moves far from the fitted rate, distinguishing "the market changed" from
"execution changed" -- reusing tier3's thresholds and wording.

### `tier5/curve.py`

```python
def plot(x, *, centre, scale, lo, hi, path, title, subtitle=None) -> str
```

One figure:

- seaborn KDE of the observed metric (the real shape)
- the fitted `N(centre, scale^2)` PDF overlaid as a dashed line (the assumed shape)
- vertical lines at `lo` and `hi`, labelled with their values
- the two out-of-band tails shaded, annotated with the observed % outside
- a caption giving `n`, `k`, and the % outside

The visible gap between the KDE and the dashed normal *is* the non-normality
that the coverage table reports numerically. That is the point of the picture.

Degrades gracefully: if seaborn or matplotlib is missing, return a message and
skip, exactly as `normality.qq_plot` already does. Nothing in the scoring path
imports it at module level.

### `tier5/batch.py`

Only needed when the twelve cells arrive as twelve files rather than one.

```
python -m tier5.batch fit   --dir extracts/year/    --bands-dir bands/
python -m tier5.batch score --dir extracts/2026-07/ --bands-dir bands/
```

Walks `*.csv` under `--dir` (recursively) and runs the corresponding module on
each. Region, strategy and period still come from the data, not the filename, so
the directory may be organised however the user likes.

One file failing does not abort the rest: failures are collected and reported in
the summary with their error text.

Ends with a cross-cell summary written to `outputs/score/<period>/_summary.csv`
-- the one place all twelve cells are comparable:

```
  region  strategy   n_fit   lo       hi      n_new  flagged  flag_%  vs_fit
  AU      VWAP       6,204  -71.20   52.10     498      11    2.21%   1.4x
  HK      VWAP       8,957  -65.74   46.27     812      19    2.34%   1.6x
  ...
```

## Testing

- `--self-check` still passes inside the standalone folder (proves the copy is intact).
- `fit` then `score` on the *same* synthetic file reproduces the in-sample flag
  rate `run.py` reports. Round-trip test proving freeze/load loses nothing.
- `persist.save` -> `load` returns identical `lo`/`hi` to float equality.
- A band file with a bumped `format_version` raises on load.
- A multi-region, multi-strategy synthetic file produces one band per cell, in
  the right folders.
- India rows suffixed `IN` and `IS` land in a single `IN` cell.
- An extract with no date column still fits and scores, labelled from `--label`.
- Overlapping fit and score windows are refused.
- A cell with no band file is skipped with a message, not scored.
- `score` on a file whose metric column is absent fails with a clear message.
- `batch` with one deliberately broken file still processes the rest.

## Out of scope

- Slicing one file into windows by date. Dates label and verify; they do not slice.
- Automated cause attribution (tier3's job).
- Refitting or updating a band from `score`. Bands are immutable once written.
- Any tier3 code in the standalone folder.
